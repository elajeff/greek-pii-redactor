import re
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from greek_pii_redactor.proxy import create_app
from greek_pii_redactor.rules import Span

NAME = "Νίκος Παπαδόπουλος"
AFM = "123456783"
EMAIL = "nikos@example.gr"
MESSAGE = f"Ο {NAME}, ΑΦΜ {AFM}, email {EMAIL}."
USER_HI = [{"role": "user", "content": "hi"}]


def find_name(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), "PERSON") for m in re.finditer(NAME, text)]


def fake_upstream(status: int = 200) -> tuple[FastAPI, list[dict]]:
    """An upstream that records what it receives and answers with the last message it got."""
    app = FastAPI()
    received: list[dict] = []

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        received.append({"authorization": request.headers.get("authorization"), "body": body})
        if status != 200:
            return JSONResponse({"error": {"message": "upstream failed"}}, status_code=status)
        reply = f"Noted: {body['messages'][-1]['content']}"
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": reply}}]})

    return app, received


@contextmanager
def proxy_to(transport: httpx.AsyncBaseTransport) -> Iterator[TestClient]:
    # The with block runs the app's lifespan, which opens the client for the upstream.
    with TestClient(create_app("http://upstream/v1", find_name, transport)) as client:
        yield client


def reply_with(content: bytes) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(200, content=content))


def test_upstream_sees_placeholders_and_client_gets_the_originals() -> None:
    upstream, received = fake_upstream()
    request = {"model": "m", "temperature": 0.2, "messages": [{"role": "user", "content": MESSAGE}]}
    with proxy_to(httpx.ASGITransport(app=upstream)) as proxy:
        response = proxy.post("/v1/chat/completions", json=request)
    assert response.status_code == 200
    sent = received[0]["body"]
    for original in (NAME, AFM, EMAIL):
        assert original not in str(sent)
    assert sent["messages"][0]["content"] == "Ο [PERSON_1], ΑΦΜ [AFM_1], email [EMAIL_1]."
    assert (sent["model"], sent["temperature"]) == ("m", 0.2)
    assert response.json()["choices"][0]["message"]["content"] == f"Noted: {MESSAGE}"


@pytest.mark.parametrize("stop, sent_stop", [(NAME, "[PERSON_1]"), ([NAME], ["[PERSON_1]"])])
def test_stop_is_redacted_and_stream_false_and_null_are_forwarded(
    stop: str | list[str], sent_stop: str | list[str]
) -> None:
    upstream, received = fake_upstream()
    request = {"messages": USER_HI, "stop": stop, "stream": False, "temperature": None}
    with proxy_to(httpx.ASGITransport(app=upstream)) as proxy:
        response = proxy.post("/v1/chat/completions", json=request)
    assert response.status_code == 200
    sent = received[0]["body"]
    assert (sent["stop"], sent["stream"], sent["temperature"]) == (sent_stop, False, None)


def test_messages_of_one_request_share_placeholders() -> None:
    upstream, received = fake_upstream()
    messages = [
        {"role": "user", "content": f"Mail {EMAIL}"},
        {"role": "assistant", "content": [{"type": "text", "text": "Mail maria@example.gr"}]},
        {"role": "user", "content": f"Again {EMAIL}"},
    ]
    with proxy_to(httpx.ASGITransport(app=upstream)) as proxy:
        proxy.post("/v1/chat/completions", json={"messages": messages})
    sent = received[0]["body"]["messages"]
    assert sent[0]["content"] == "Mail [EMAIL_1]"
    assert sent[1]["content"] == [{"type": "text", "text": "Mail [EMAIL_2]"}]
    assert sent[2]["content"] == "Again [EMAIL_1]"


def test_unreachable_upstream_gives_502_without_the_request_text() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with proxy_to(httpx.MockTransport(refuse)) as proxy:
        response = proxy.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": MESSAGE}]}
        )
    assert response.status_code == 502
    assert "message" in response.json()["error"]
    for original in (NAME, AFM, EMAIL):
        assert original not in response.text


@pytest.mark.parametrize(
    "extra, messages",
    [
        ({"stream": True}, USER_HI),
        ({"temperature": {"x": NAME}}, USER_HI),
        ({"n": True}, USER_HI),
        ({"stop": [NAME, 5]}, USER_HI),
        ({"model": 5}, USER_HI),
        ({}, [{"role": "user", "content": [{"type": "text", "text": "hi", "note": NAME}]}]),
        ({"tools": [{"type": "function"}]}, USER_HI),
        ({"prediction": {"type": "content", "content": NAME}}, USER_HI),
        ({"user": NAME}, USER_HI),
        ({}, [{"role": "user", "name": NAME, "content": "hi"}]),
        ({}, [{"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}]),
        ({}, [{"role": "tool", "content": "result"}]),
        ({}, [{"content": "no role"}]),
        ({}, [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]),
        ({}, [{"role": "user", "content": "Send it to [PERSON_1]."}]),
    ],
)
def test_unsupported_requests_are_refused_and_not_forwarded(
    extra: dict, messages: list[dict]
) -> None:
    upstream, received = fake_upstream()
    with proxy_to(httpx.ASGITransport(app=upstream)) as proxy:
        response = proxy.post("/v1/chat/completions", json={"messages": messages, **extra})
    assert response.status_code == 400
    assert "message" in response.json()["error"]
    assert received == []


@pytest.mark.parametrize(
    "raw_body",
    [
        b'{"messages": [{"role": "user", "content": [{"type": "text"}]}]}',
        rb'{"messages": [{"role": "user", "content": "x \ud800"}]}',
        b'{"messages": [{"role": "user", "content": "hi"}], "temperature": NaN}',
        rb'{"\ud800": 1, "messages": []}',
        rb'{"messages": [{"\ud800": 1, "role": "user", "content": "hi"}]}',
        rb'{"messages": [{"role": "user", "content": "hi"}], "model": "\ud800"}',
        b"[]",
        b"not json",
    ],
)
def test_malformed_requests_give_400_and_are_not_forwarded(raw_body: bytes) -> None:
    upstream, received = fake_upstream()
    with proxy_to(httpx.ASGITransport(app=upstream)) as proxy:
        response = proxy.post(
            "/v1/chat/completions", content=raw_body, headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 400
    assert received == []


@pytest.mark.parametrize(
    "reply",
    [
        b"not json",
        b"[]",
        b'{"choices": null}',
        b'{"choices": [{"message": null}]}',
        b'{"choices": [{"message": {"content": NaN}}]}',
    ],
)
def test_malformed_upstream_reply_gives_502(reply: bytes) -> None:
    with proxy_to(reply_with(reply)) as proxy:
        response = proxy.post("/v1/chat/completions", json={"messages": USER_HI})
    assert response.status_code == 502


def test_upstream_reply_with_a_lone_surrogate_still_gives_200() -> None:
    with proxy_to(reply_with(rb'{"choices": [{"message": {"content": "x \ud800"}}]}')) as proxy:
        response = proxy.post("/v1/chat/completions", json={"messages": USER_HI})
    assert response.status_code == 200


def test_authorization_header_reaches_the_upstream() -> None:
    upstream, received = fake_upstream()
    with proxy_to(httpx.ASGITransport(app=upstream)) as proxy:
        proxy.post(
            "/v1/chat/completions",
            json={"messages": USER_HI},
            headers={"Authorization": "Bearer sk-test"},
        )
    assert received[0]["authorization"] == "Bearer sk-test"


@pytest.mark.parametrize("status", [429, 500])
def test_upstream_errors_keep_their_status(status: int) -> None:
    upstream, _ = fake_upstream(status)
    with proxy_to(httpx.ASGITransport(app=upstream)) as proxy:
        response = proxy.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": MESSAGE}]}
        )
    assert response.status_code == status
    assert response.json() == {"error": {"message": "upstream failed"}}

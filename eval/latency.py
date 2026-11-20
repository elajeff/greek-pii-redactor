"""Measures the time the proxy adds to a request, not the time the model takes.

Takes a seeded sample of data/synthetic_test.jsonl (python eval/make_synthetic.py --seed 7
--split test --count 2000). The proxy runs in this process in front of a fake upstream that
answers at once, so the numbers are redaction, restore and the proxy's own handling. For each
sentence the same request is also sent straight to the fake upstream; the added time is the
difference of the two round trips. There is no network and no uvicorn in between, so a real
deployment adds a little more. Reports IDs only, and with --model-dir the PERSON tagger too.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from greek_pii_redactor.proxy import create_app
from greek_pii_redactor.rules import Span

DATA_FILE = Path("data/synthetic_test.jsonl")
ENDPOINT = "http://proxy/v1/chat/completions"
# The first requests load kernels and warm caches; they are not what a running proxy costs.
WARMUP_REQUESTS = 5


def make_fake_upstream() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        reply = body["messages"][-1]["content"]
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": reply}}]})

    return app


async def added_milliseconds(
    texts: list[str], detect_names: Callable[[str], list[Span]] | None
) -> list[float]:
    upstream = make_fake_upstream()
    proxy = create_app("http://upstream/v1", detect_names, httpx.ASGITransport(app=upstream))
    added = []
    # ASGITransport does not run the proxy's lifespan, so it is entered here.
    async with (
        proxy.router.lifespan_context(proxy),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy)) as through_proxy,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream)) as direct,
    ):
        for number, text in enumerate(WARMUP_REQUESTS * texts[:1] + texts):
            request = {"messages": [{"role": "user", "content": text}]}
            start = time.perf_counter()
            reply = await through_proxy.post(ENDPOINT, json=request)
            proxied = time.perf_counter() - start
            start = time.perf_counter()
            await direct.post(ENDPOINT, json=request)
            straight = time.perf_counter() - start
            reply.raise_for_status()
            if number >= WARMUP_REQUESTS:
                added.append((proxied - straight) * 1000)
    return added


def report(label: str, added: list[float]) -> None:
    p95 = statistics.quantiles(added, n=20)[-1]
    print(f"{label:<14} median {statistics.median(added):7.2f} ms   p95 {p95:7.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-dir", type=Path, help="weights from train/train_tagger.py")
    args = parser.parse_args()
    if not DATA_FILE.exists():
        raise SystemExit(
            f"{DATA_FILE} not found. Create it with: "
            "python eval/make_synthetic.py --seed 7 --split test --count 2000"
        )
    if args.model_dir and not args.model_dir.is_dir():
        raise SystemExit(f"{args.model_dir} not found. Train it with: python train/train_tagger.py")

    with DATA_FILE.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    if args.count > len(records):
        raise SystemExit(f"--count {args.count} exceeds the {len(records)} records in {DATA_FILE}")
    texts = [record["text"] for record in random.Random(args.seed).sample(records, args.count)]

    print(f"{args.count} requests, seed {args.seed}, added time per request (proxy minus direct)")
    report("IDs only", asyncio.run(added_milliseconds(texts, None)))
    if args.model_dir:
        # Imported here so the script runs without torch and transformers installed.
        from greek_pii_redactor.tagger import load_tagger

        report("with tagger", asyncio.run(added_milliseconds(texts, load_tagger(args.model_dir))))
    print("single run on this machine, not a benchmark")


if __name__ == "__main__":
    main()

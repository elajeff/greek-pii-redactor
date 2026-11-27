import subprocess
import sys
from pathlib import Path

import pytest

AFM = "123456783"
PHONE = "6912345678"
EMAIL = "nikos@example.gr"
TEXT = f"Ο πελάτης έδωσε ΑΦΜ {AFM}.\r\nΤηλέφωνο {PHONE}, email {EMAIL}.\n".encode()
REPLY = "Ο [PERSON_1] έστειλε [EMAIL_1].".encode()


def run(cwd: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    # A real process with piped stdout, so the UTF-8 and LF handling in main() is what runs.
    command = [sys.executable, "-c", "from greek_pii_redactor.cli import main; main()", *args]
    return subprocess.run(command, cwd=cwd, capture_output=True)


def test_redact_then_restore_gives_the_file_back(tmp_path: Path) -> None:
    (tmp_path / "message.txt").write_bytes(TEXT)

    redacted = run(tmp_path, "redact", "message.txt")
    assert redacted.returncode == 0
    for personal in (AFM, PHONE, EMAIL):
        assert personal.encode() not in redacted.stdout
    for placeholder in (b"[AFM_1]", b"[PHONE_1]", b"[EMAIL_1]"):
        assert placeholder in redacted.stdout
    assert (tmp_path / "message.txt.mapping.json").is_file()

    (tmp_path / "redacted.txt").write_bytes(redacted.stdout)
    restored = run(tmp_path, "restore", "redacted.txt", "--mapping", "message.txt.mapping.json")
    assert restored.returncode == 0
    assert restored.stdout == TEXT


@pytest.mark.parametrize(
    "args, files, named_file",
    [
        (
            ["redact", "message.txt"],
            {"message.txt": TEXT, "message.txt.mapping.json": b'{"[EMAIL_1]": ["a@b.gr"]}'},
            "message.txt.mapping.json",
        ),
        (
            ["restore", "message.txt", "--mapping", "mapping.json"],
            {"message.txt": TEXT, "mapping.json": b"not json"},
            "mapping.json",
        ),
        (
            ["restore", "message.txt", "--mapping", "mapping.json"],
            {"message.txt": TEXT, "mapping.json": b'["[EMAIL_1]", "a@b.gr"]'},
            "mapping.json",
        ),
        *[
            (
                ["restore", "reply.txt", "--mapping", "mapping.json"],
                {"reply.txt": REPLY, "mapping.json": mapping.encode()},
                "mapping.json",
            )
            for mapping in [
                '{"[EMAIL_1]": "a@b.gr"}',
                '{"[EMAIL_1]": ["a@b.gr"]}',
                '{"[EMAIL_1]": []}',
                '{"[PERSON_1]": [{"text": "Νίκος"}]}',
            ]
        ],
    ],
    ids=[
        "mapping exists",
        "mapping not json",
        "mapping is a list",
        "forms are a string",
        "forms are strings",
        "no forms",
        "person form without article",
    ],
)
def test_bad_input_is_refused_with_a_message(
    tmp_path: Path, args: list[str], files: dict[str, bytes], named_file: str
) -> None:
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)

    refused = run(tmp_path, *args)

    stderr = refused.stderr.decode()
    assert refused.returncode != 0
    assert len(stderr.splitlines()) == 1
    assert named_file in stderr
    assert "Traceback" not in stderr
    for name, content in files.items():
        assert (tmp_path / name).read_bytes() == content

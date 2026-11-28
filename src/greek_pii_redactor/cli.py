"""Command line: redact a file, restore it with the saved mapping, or run the proxy."""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from greek_pii_redactor.names import LEXICON_FILE, MODELS_FILE, Names, load_names
from greek_pii_redactor.redact import redact, restore
from greek_pii_redactor.rules import Span

NAMES_NOT_REDACTED = "Names are not redacted: pass --model-dir to redact them."


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", newline="")
    except OSError as error:
        raise SystemExit(f"Cannot read {path}: {error.strerror}") from error
    except UnicodeDecodeError as error:
        raise SystemExit(f"{path} is not UTF-8 text") from error


def load_name_detector(model_dir: Path | None) -> Callable[[str], list[Span]] | None:
    if model_dir is None:
        print(NAMES_NOT_REDACTED, file=sys.stderr)
        return None
    if not model_dir.is_dir():
        raise SystemExit(
            f"{model_dir} not found. Download it with: "
            f"hf download elajeff/greek-pii-redactor-ner --local-dir {model_dir}"
        )
    # Imported here so the commands that don't need names work without torch installed.
    from greek_pii_redactor.tagger import load_tagger

    return load_tagger(model_dir)


def load_names_dir(names_dir: Path | None) -> Names | None:
    if names_dir is None:
        return None
    build = f"Build it with: python train/train_inflection.py --holdout 0 --out {names_dir}"
    if not names_dir.is_dir():
        raise SystemExit(f"{names_dir} not found. {build}")
    for file_name in (LEXICON_FILE, MODELS_FILE):
        if not (names_dir / file_name).is_file():
            raise SystemExit(f"{names_dir} has no {file_name}. {build}")
    try:
        return load_names(names_dir)
    except ImportError as error:
        raise SystemExit(
            f"{error.msg}. Install the names dependencies: pip install greek-pii-redactor[names]"
        ) from error


def run_redact(args: argparse.Namespace) -> None:
    mapping_path = args.mapping or args.file.with_name(f"{args.file.name}.mapping.json")
    mapping_exists = f"{mapping_path} already exists: it is not overwritten. Use --mapping."
    if mapping_path.exists():
        raise SystemExit(mapping_exists)
    text = read_text(args.file)
    detect_names = load_name_detector(args.model_dir)
    try:
        redacted, placeholders = redact(text, detect_names)
    except ValueError as error:
        raise SystemExit(f"{args.file}: {error}") from error
    mapping = json.dumps(placeholders, ensure_ascii=False, indent=2)
    try:
        with mapping_path.open("x", encoding="utf-8") as file:
            file.write(mapping)
    except FileExistsError as error:
        raise SystemExit(mapping_exists) from error
    print(redacted, end="")


def run_restore(args: argparse.Namespace) -> None:
    names = load_names_dir(args.names_dir)
    try:
        placeholders = json.loads(read_text(args.mapping))
    except json.JSONDecodeError as error:
        raise SystemExit(f"{args.mapping} is not valid JSON: {error}") from error
    # restore reads the first form's text, and each form of a PERSON with its article; a blank
    # PERSON text fails when the gender is read from it.
    if not isinstance(placeholders, dict) or not all(
        isinstance(forms, list)
        and forms
        and all(
            isinstance(form, dict)
            and isinstance(form.get("text"), str)
            and (
                not placeholder.startswith("[PERSON_")
                or (isinstance(form.get("article"), str) and form["text"].strip())
            )
            for form in forms
        )
        for placeholder, forms in placeholders.items()
    ):
        raise SystemExit(f"{args.mapping} is not a mapping written by the redact command")
    print(restore(read_text(args.file), placeholders, names), end="")


def run_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn

        from greek_pii_redactor.proxy import create_app
    except ImportError as error:
        raise SystemExit(
            f"{error.msg}. Install the server dependencies: pip install greek-pii-redactor[serve]"
        ) from error
    names = load_names_dir(args.names_dir)
    app = create_app(args.upstream, load_name_detector(args.model_dir), names=names)
    # The access log prints the request path and query string, which can hold personal data.
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


def main() -> None:
    parser = argparse.ArgumentParser(prog="greek-pii-redactor", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    redact_command = commands.add_parser("redact", help="print FILE with personal data replaced")
    redact_command.add_argument("file", type=Path)
    redact_command.add_argument("--model-dir", type=Path, help="PERSON tagger weights")
    redact_command.add_argument(
        "--mapping", type=Path, help="where to write the mapping (default: FILE.mapping.json)"
    )
    redact_command.set_defaults(run=run_redact)

    restore_command = commands.add_parser("restore", help="print FILE with the originals back")
    restore_command.add_argument("file", type=Path)
    restore_command.add_argument("--mapping", type=Path, required=True)
    restore_command.add_argument(
        "--names-dir", type=Path, help="name forms built by train/train_inflection.py"
    )
    restore_command.set_defaults(run=run_restore)

    serve_command = commands.add_parser("serve", help="run the OpenAI-compatible proxy")
    serve_command.add_argument("--upstream", required=True, help="e.g. http://localhost:11434/v1")
    serve_command.add_argument("--model-dir", type=Path, help="PERSON tagger weights")
    serve_command.add_argument(
        "--names-dir", type=Path, help="name forms built by train/train_inflection.py"
    )
    serve_command.add_argument("--host", default="127.0.0.1")
    serve_command.add_argument("--port", type=int, default=8080)
    serve_command.set_defaults(run=run_serve)

    args = parser.parse_args()
    # Redirected output would be cp1252 on Windows and turn LF into CRLF.
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    args.run(args)

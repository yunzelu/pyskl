"""Create a temporary S2 training config that resumes from a checkpoint."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def set_assignment(text: str, name: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(name)}\s*=.*$", flags=re.MULTILINE)
    line = f"{name} = {value}"
    if pattern.search(text):
        return pattern.sub(line, text)
    return text.rstrip() + "\n" + line + "\n"


def make_resume_config(base_config: Path, resume_from: Path, output: Path) -> Path:
    if not base_config.exists():
        raise FileNotFoundError(f"Missing base config: {base_config}")
    if not resume_from.exists():
        raise FileNotFoundError(f"Missing resume checkpoint: {resume_from}")

    checkpoint_value = repr(resume_from.as_posix())
    text = base_config.read_text(encoding="utf-8")
    text = set_assignment(text, "load_from", "None")
    text = set_assignment(text, "resume_from", checkpoint_value)
    text = set_assignment(text, "auto_resume", "False")
    compile(text, str(output), "exec")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = make_resume_config(args.base_config, args.resume_from, args.output)
    print(f"[DONE] wrote resume config to {output}")


if __name__ == "__main__":
    main()

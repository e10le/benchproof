from pathlib import Path


def read_lines(path: Path) -> list[str]:
    # str.splitlines() also splits on U+2028, U+0085 and similar, which JSON strings may hold unescaped.
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line]

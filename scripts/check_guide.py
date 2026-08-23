#!/usr/bin/env python3

import re
import sys
from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def github_anchor(title: str) -> str:
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"[`*_~]", "", title).strip().lower()
    title = re.sub(r"[^\w\- ]", "", title)
    return re.sub(r"[ ]+", "-", title)


def main() -> int:
    text = README.read_text(encoding="utf-8")
    lines = text.splitlines()
    errors: list[str] = []

    anchors: set[str] = set()
    anchor_counts: dict[str, int] = {}
    for line in lines:
        match = HEADING.match(line)
        if not match:
            continue
        base = github_anchor(match.group(2))
        count = anchor_counts.get(base, 0)
        anchor_counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")

    for line_number, line in enumerate(lines, 1):
        for target in LINK.findall(line):
            if target.startswith("#") and target[1:] not in anchors:
                errors.append(f"README.md:{line_number}: broken internal anchor {target}")

    current_section = "before first section"
    section_urls: set[str] = set()
    section_line = 0

    def finish_resource_section() -> None:
        if section_line and not section_urls:
            errors.append(f"README.md:{section_line}: {current_section!r} has no external resources")

    for line_number, line in enumerate(lines, 1):
        heading = HEADING.match(line)
        if heading and len(heading.group(1)) == 3:
            finish_resource_section()
            current_section = heading.group(2)
            section_urls = set()
            section_line = line_number
            continue

        for target in LINK.findall(line):
            if target.startswith(("#", "mailto:")) or "://" not in target:
                continue
            normalized = target.rstrip("/")
            if normalized in section_urls:
                errors.append(
                    f"README.md:{line_number}: duplicate link in {current_section!r}: {target}"
                )
            section_urls.add(normalized)

    finish_resource_section()

    if re.search(r"^#{2,6}\s+Tier\s+\d", text, re.MULTILINE):
        errors.append("README.md: tier headings are no longer part of the guide")

    if "**Checkpoint:**" in text:
        errors.append("README.md: checkpoints are not part of the resource-list format")

    if errors:
        print("Guide checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Guide checks passed for {len(anchors)} headings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

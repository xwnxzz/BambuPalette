"""Decode a GitHub contents-API JSON response into the file it carries.

The API returns file bytes base64-encoded inside a JSON envelope. ``web_fetch``
spills large responses to a temporary file, so this script takes that spill path
and writes the decoded bytes out. Reference material only -- it is never part of
the application.

Usage:
    python tools/decode_github_content.py <spill.json> <output-path>
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2

    source = Path(argv[1])
    target = Path(argv[2])

    text = source.read_text(encoding="utf-8", errors="replace")
    # The spill may carry a short prefix before the JSON object.
    start = text.find("{")
    if start < 0:
        print("no JSON object found in the spill file")
        return 1

    envelope = json.loads(text[start:])
    if "content" not in envelope:
        print(f"envelope has no content field: {sorted(envelope)}")
        return 1

    payload = base64.b64decode(envelope["content"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)

    print(f"name   {envelope.get('name')}")
    print(f"sha    {envelope.get('sha')}")
    print(f"size   {envelope.get('size')} (decoded {len(payload)})")
    print(f"wrote  {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Read-only command line audits and a clearly marked synthetic walkthrough."""

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys

from .coco_labels import inspect_coco
from .demo import synthetic_demo
from .provenance import validate_run_record
from .splits import inspect_splits
from .summaries import summarize_runs
from .yolo_labels import inspect_yolo_labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="run synthetic geometry/reliability arithmetic")
    for name in ("coco", "splits", "run-record", "summarize", "yolo"):
        command = subparsers.add_parser(name)
        command.add_argument("path", type=Path)
        if name == "yolo":
            command.add_argument("--classes", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = synthetic_demo()
        else:
            content = args.path.read_text(encoding="utf-8-sig")
            if args.command == "yolo":
                result = inspect_yolo_labels(content, args.classes)
            else:
                document = json.loads(content)
                result = {"coco": inspect_coco, "splits": inspect_splits,
                          "run-record": validate_run_record, "summarize": summarize_runs}[args.command](document)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False,
                         default=lambda value: asdict(value) if is_dataclass(value) else str(value)))
        return 0 if result.get("valid", True) else 1
    except (OSError, UnicodeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

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
from .local_data import hash_splits
from .image_headers import image_dimensions
from .yolo_dataset import inspect_yolo_dataset
from .coco_assets import inspect_coco_assets
from .conversion import coco_to_yolo
from .box_svg import box_svg
from .protocols import protocol_fingerprint
from .experiment_matrix import experiment_matrix
from .paired_results import paired_results
from .reference_pipeline import reference_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="run synthetic geometry/reliability arithmetic")
    for name in ("coco", "splits", "run-record", "summarize", "yolo", "hash-splits", "image-info", "yolo-dataset", "coco-assets", "convert-coco", "box-svg", "protocol", "matrix", "paired", "reference"):
        command = subparsers.add_parser(name)
        command.add_argument("path", type=Path)
        if name in ("yolo", "yolo-dataset"):
            command.add_argument("--classes", type=int, required=True)
        if name in ("hash-splits", "yolo-dataset", "coco-assets"):
            command.add_argument("--root", type=Path, required=True)
        if name == "convert-coco":
            command.add_argument("--ignored-policy", choices=("reject", "skip"), default="reject")
        if name == "paired":
            for option in ("baseline", "candidate", "metric"):
                command.add_argument("--" + option, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = synthetic_demo()
        elif args.command == "image-info":
            result = image_dimensions(args.path)
        else:
            content = args.path.read_text(encoding="utf-8-sig")
            if args.command == "yolo":
                result = inspect_yolo_labels(content, args.classes)
            else:
                document = json.loads(content)
                readers = {"coco": inspect_coco, "splits": inspect_splits, "run-record": validate_run_record, "summarize": summarize_runs, "reference": reference_pipeline}
                if args.command in readers:
                    result = readers[args.command](document)
                elif args.command == "hash-splits":
                    result = hash_splits(args.root, document)
                elif args.command == "yolo-dataset":
                    result = inspect_yolo_dataset(args.root, document, args.classes)
                elif args.command == "coco-assets":
                    result = inspect_coco_assets(args.root, document)
                elif args.command == "convert-coco":
                    result = coco_to_yolo(document, args.ignored_policy)
                elif args.command == "protocol":
                    result = {"protocol_fingerprint": protocol_fingerprint(document)}
                elif args.command == "paired":
                    result = paired_results(document, args.baseline, args.candidate, args.metric)
                elif args.command == "matrix":
                    if not isinstance(document, dict):
                        raise ValueError("matrix input must be an object")
                    result = experiment_matrix(document.get("protocol"), document.get("methods"), document.get("seeds"))
                elif args.command == "box-svg":
                    if not isinstance(document, dict):
                        raise ValueError("SVG input must be an object")
                    print(box_svg(document.get("width"), document.get("height"), document.get("annotations", [])))
                    return 0
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False,
                         default=lambda value: asdict(value) if is_dataclass(value) else str(value)))
        return 0 if result.get("valid", True) else 1
    except (OSError, UnicodeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

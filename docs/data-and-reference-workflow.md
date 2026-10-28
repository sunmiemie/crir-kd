# Local data and reference workflow

These tools were implemented and tested on 2026-10-21. Historical author dates do not imply that this implementation or experimental evidence existed earlier. All examples below are preparation or arithmetic references; none train or evaluate a detector.

Run from the repository root with Python 3.10+:

```console
python -m crir_kd reference examples/reference.synthetic.json
python -m crir_kd image-info path/to/image.jpg
python -m crir_kd hash-splits path/to/splits.json --root path/to/dataset
python -m crir_kd yolo-dataset path/to/pairs.json --root path/to/dataset --classes 10
python -m crir_kd coco-assets path/to/annotations.json --root path/to/images
python -m crir_kd convert-coco path/to/annotations.json
python -m crir_kd box-svg path/to/boxes.json
python -m crir_kd protocol path/to/protocol.json
python -m crir_kd matrix path/to/matrix.json
python -m crir_kd paired path/to/results.json --baseline quality --candidate joint --metric AP
```

Commands write results to stdout. `box-svg` emits SVG text suitable for redirecting to a file; other commands emit JSON. No command downloads data or overwrites source labels. The existing 0/1/2 exit status convention applies.

`hash-splits` consumes the documented split manifest and calculates SHA-256 from actual local bytes. Paths and symlinks must resolve to files inside the explicit dataset root. `yolo-dataset` consumes an array of `{"image":"images/a.jpg","label":"labels/a.txt"}` objects. Missing label files are errors; existing empty labels are negative images. It reports pixel boxes and class counts, while retaining line errors. `coco-assets` compares declared dimensions to local PNG/JPEG headers. Header validation does not prove a full image can be decoded.

`convert-coco` emits a deterministic category-to-class mapping and label text by image ID, including negative images. Crowd/ignored objects fail by default. Explicit `--ignored-policy skip` is lossy and reports every skipped annotation; original official evaluation annotations must be preserved. This generic utility does not implement VisDrone ignore-region rules or replace dataset-specific conversion review.

`box-svg` takes `width`, `height`, and `annotations` with pixel `xywh` and optional `label`. It escapes labels, uses no scripts or external resources, and rejects outside-canvas boxes.

Input-container validation added on 2026-10-28 requires JSON `annotations` to be an array of objects. An empty array or omitted field produces an unannotated canvas; `null`, booleans, numbers, strings, and objects are input errors. These errors return exit status `2` with a JSON error on stderr and no SVG on stdout. Direct Python calls accept lists and tuples. Invalid later rows also produce no partial SVG output.

The reference input schema is demonstrated completely in `examples/reference.synthetic.json`. GT and reference predictions use original-pixel `xyxy`; each transformed view specifies an original-coordinate crop, output `[width,height]`, optional horizontal flip, and predictions in view pixels. By default only fully visible targets count in perturbation views. Teacher misses remain in coverage denominators. Feature boxes use each feature grid's own coordinates. The sampler uses one bilinear sample per output bin, with pixel centers at `i+0.5`; it is a small reference, not optimized RoIAlign or autograd. Student and teacher channel counts must agree. Optional whole-region L2 normalization and MSE have explicit scalar behavior; no learnable projection is implemented.

The reference report includes every GT, aligned local feature losses, quality/stability controls, an equal-view-budget mean-quality control, and a joint mean-weight control. `permutation.permute_weights` additionally provides seeded within-size/quality-bin weight shuffling; group boundaries and seed are explicit. No weight operation deletes detection supervision.

Protocols require `dataset`, `student`, `teacher`, `optimization`, `augmentation`, `evaluation`, and `kd_features` objects. They must identify dataset version/split, student implementation/input size, teacher checkpoint/input size, and evaluation implementation. Method and seed are run-level fields and are excluded by schema. Other controlled settings are retained in the fingerprint. `protocols.compare_protocols` reports differing fields. Fingerprints describe supplied metadata, not independent proof of experimental equivalence.

Matrix input contains `protocol`, distinct `seeds`, and `methods` objects with `name` and `teacher_forwards_per_step`. Runs are explicitly marked planned and contain no results; joint must include quality-only. Teacher budget differences are exposed. Paired result reports require the existing run-summary schema, one protocol and synthetic/real status, and pair only shared seeds while listing exclusions. Deltas are descriptive; no significance or method-superiority claim is made.

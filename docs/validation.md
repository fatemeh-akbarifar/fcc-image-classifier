# Reproducibility report

Verified on 26 September 2026 with Python 3.9.13, TensorFlow 2.16.2, Keras 3.7.0, NumPy 1.23.5, and macOS ARM64 CPU.

## Functional validation

`python -m pytest -q`: **4 passed** in both the existing runtime and a newly installed isolated environment. Tests cover a real model training step, save/load prediction consistency, numeric test-image ordering, missing-image rejection, and archive traversal rejection.

The pinned requirements installed successfully in a clean virtual environment. `pip check` reported no broken requirements.

## Full-dataset evaluation

```bash
python image_classifier.py --epochs 15 --seed 42
```

The default configuration trains on the 2,000 supplied training images, selects weights using 1,000 validation images, and evaluates all 50 numbered test images. It writes measured metrics, predictions, a training plot, and the saved model under `artifacts/`. Full retraining was stopped because it was unnecessary for documenting the original achievement. No new final test accuracy is claimed. The original notebook records 79.58% validation accuracy after 35 epochs; see [the historical evidence](original-work.md).

[Dataset checksums](data-manifest.json) identify the validation input. Model quality is distinct from a passing software test suite; do not interpret the functional tests as an accuracy claim.

## First-run downloads

The CDN rejected Python's default HTTP user agent (403). The downloader now supplies an explicit client header; a fresh real HTTPS download matched the recorded checksum. A focused test verifies that header, complete file writing, and cache reuse.

## Hosted checks

[The CNN implementation passed GitHub Actions on Python 3.11](https://github.com/fatemeh-akbarifar/fcc-image-classifier/actions/runs/36243170999). Test-image resizing in the maintained code now explicitly uses bilinear interpolation, matching the training/validation resize method.

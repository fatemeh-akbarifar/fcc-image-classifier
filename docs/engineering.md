# Engineering notes

## Original project

Cat and dog classification with a convolutional neural network by Fatemeh Akbarifar, developed using a freeCodeCamp starter. The existing Git history and original Colab source establish the project provenance.

## Reproducibility work (September 2026)

- Replace notebook-only shell/magic commands in Python entry points with explicit download helpers and command-line interfaces.
- Pin dependencies and add focused tests plus a GitHub Actions workflow.
- Make imports free of downloads and training side effects.
- Save measured outputs separately from source code.
- Provide a notebook entry point that runs the same Python implementation.

## Method-specific changes

1. Use the supplied splits: 2,000 training images, 1,000 validation images, and 50 numbered test images.
2. Resize to 150 × 150 RGB pixels. Apply random flips, rotations, zooms, and translations only during training.
3. Rescale pixels inside the model, then apply four convolution/max-pooling blocks with 32, 64, 128, and 256 filters.
4. Use a 512-unit dense layer, 50% dropout, and a sigmoid output, trained with Adam and binary cross-entropy.
5. Restore the best validation-loss weights. Evaluate every test image in numeric filename order (`1.jpg` through `50.jpg`) against the original challenge labels. The challenge threshold is 63%.

Class mapping: **0 = cat, 1 = dog**. Input to the saved model is RGB pixels in the 0–255 range, resized to 150 × 150. Rescaling is already included. All training and validation batches are consumed, including the final partial batch.

## Reading the evidence

The validation report describes newly executed runs. It does not retroactively claim that historical notebook outputs used the corrected evaluation pipeline. Unit tests verify behavior; they are not model-quality benchmarks.

## Data provenance

[Dataset checksums](data-manifest.json) identify the exact downloaded inputs used for validation. These are hashes of public dataset files, not private Drive content.

The current Git tree excludes the original duplicate dataset ZIP, extracted images, and `__MACOSX` files. Download caching replaces those tracked copies; prior Git history is preserved.

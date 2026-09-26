"""Train the project's four-block convolutional cat/dog classifier."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from data_utils import download, extract_zip

ANSWERS = [1,0,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,0,1,1,0,0,1,1,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,0,1,0,1,1,0,0,0,0,0,0]
SIZE = (150, 150)


def ordered_test_files(directory):
    files = list(Path(directory).glob("*.jpg"))
    expected = {f"{i}.jpg" for i in range(1, 51)}
    if {p.name for p in files} != expected:
        raise ValueError("FCC evaluation requires exactly test/1.jpg through test/50.jpg")
    return sorted(files, key=lambda p: int(p.stem))


def build_model():
    layers = tf.keras.layers
    model = tf.keras.Sequential([layers.Input(shape=(*SIZE, 3)), layers.Rescaling(1/255.)])
    for filters in (32, 64, 128, 256):
        model.add(layers.Conv2D(filters, 3, activation="relu"))
        model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def make_dataset(directory, batch_size, training, seed):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory, class_names=["cats", "dogs"], label_mode="binary", image_size=SIZE,
        batch_size=batch_size, shuffle=training, seed=seed)
    if training:
        augment = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(.1, seed=seed+1),
            tf.keras.layers.RandomZoom(.2, seed=seed+2),
            tf.keras.layers.RandomTranslation(.2, .2, seed=seed+3)])
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=2)
    return ds.prefetch(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("cats_and_dogs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    if not args.data_dir.exists():
        archive = download("https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip", args.data_dir.parent / "cats_and_dogs.zip")
        extract_zip(archive, args.data_dir.parent)
    for split in ("train", "validation", "test"):
        if not (args.data_dir / split).is_dir():
            raise ValueError(f"Missing dataset split: {args.data_dir / split}")
    files = ordered_test_files(args.data_dir / "test")
    train = make_dataset(args.data_dir / "train", args.batch_size, True, args.seed)
    validation = make_dataset(args.data_dir / "validation", args.batch_size, False, args.seed)
    model = build_model()
    history = model.fit(train, validation_data=validation, epochs=args.epochs, verbose=2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    test_images = np.stack([tf.keras.utils.img_to_array(tf.keras.utils.load_img(p, target_size=SIZE, interpolation="bilinear")) for p in files])
    probabilities = model.predict(test_images, batch_size=args.batch_size, verbose=0).ravel()
    accuracy = float(np.mean((probabilities >= .5) == np.array(ANSWERS)))
    val_metrics = model.evaluate(validation, verbose=0, return_dict=True)
    metrics = {"test_accuracy": accuracy, "test_images": len(files), "challenge_passed": accuracy >= .63,
               "validation_accuracy": float(val_metrics["accuracy"]), "epochs_run": len(history.history["loss"]), "seed": args.seed}
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "model.keras")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output / "history.json").write_text(json.dumps(history.history, indent=2))
    pd.DataFrame({"filename": [p.name for p in files], "dog_probability": probabilities,
                  "predicted": (probabilities >= .5).astype(int), "actual": ANSWERS}).to_csv(args.output / "predictions.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(history.history[metric], label="Training")
        ax.plot(history.history["val_" + metric], label="Validation")
        ax.set(xlabel="Epoch (zero-based)", ylabel=metric); ax.legend()
    fig.tight_layout(); fig.savefig(args.output / "training.png"); plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

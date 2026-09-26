import numpy as np
import pytest
import tensorflow as tf
from image_classifier import ordered_test_files, build_model, SIZE


def test_numeric_test_order_and_missing_image(tmp_path):
    for i in reversed(range(1,51)): (tmp_path / f"{i}.jpg").touch()
    assert [p.name for p in ordered_test_files(tmp_path)][:3] == ["1.jpg", "2.jpg", "3.jpg"]
    (tmp_path / "17.jpg").unlink()
    with pytest.raises(ValueError): ordered_test_files(tmp_path)


def test_training_and_saved_prediction(tmp_path):
    model = build_model()
    images = np.full((2, *SIZE, 3), 128, dtype=np.float32)
    assert np.isfinite(model.train_on_batch(images, np.array([[0.],[1.]]))).all()
    before = model(images, training=False).numpy()
    path = tmp_path / "model.keras"; model.save(path)
    loaded = tf.keras.models.load_model(path)
    np.testing.assert_allclose(before, loaded(images, training=False).numpy(), atol=1e-6)
    assert np.all((before >= 0) & (before <= 1))

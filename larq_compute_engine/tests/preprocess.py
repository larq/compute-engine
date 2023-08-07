"""From larq_zoo/training/data.py"""
import tensorflow as tf

IMAGE_SIZE = 224
CROP_PADDING = 32

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def _center_crop(image, image_size):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        (
            (image_size / (image_size + CROP_PADDING))
            * tf.cast(tf.minimum(image_height, image_width), tf.float32)
        ),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2

    image = tf.image.crop_to_bounding_box(
        image,
        offset_height,
        offset_width,
        padded_center_crop_size,
        padded_center_crop_size,
    )
    image = tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0]
    return image


def _normalize(image, mean_rgb=MEAN_RGB, stddev_rgb=STDDEV_RGB):
    """Normalizes images to variance 1 and mean 0 over the whole dataset"""

    image -= tf.broadcast_to(mean_rgb, tf.shape(image))
    image /= tf.broadcast_to(stddev_rgb, tf.shape(image))

    return image


def preprocess_image_tensor(image_tensor, image_size=IMAGE_SIZE):
    """Preprocesses the given image Tensor.

    Args:
      image_tensor: `Tensor` representing an image array arbitrary size.
      image_size: image size.

    Returns:
      A preprocessed and normalized image `Tensor`.
    """
    image_tensor = _center_crop(image_tensor, image_size)
    image_tensor = tf.reshape(image_tensor, [image_size, image_size, 3])
    image_tensor = tf.cast(image_tensor, dtype=tf.float32)
    image_tensor = _normalize(image_tensor, mean_rgb=MEAN_RGB, stddev_rgb=STDDEV_RGB)
    return image_tensor

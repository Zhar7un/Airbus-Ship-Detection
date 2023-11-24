import tensorflow as tf
import os


target_size = (120, 120)


def preprocess_image(image):
    image = tf.image.resize(image, target_size)
    image = image / 255.0
    return image


def preprocess_mask(mask):
    mask = tf.image.resize(mask, target_size)
    mask = tf.cast(mask, dtype=tf.uint8)
    mask = mask // 255
    return mask


def preprocess_data(image, mask):
    image = preprocess_image(image)
    mask = preprocess_mask(mask)
    return image, mask


def load_image_and_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    return image, mask


def create_dataset(data_folder, batch_size, shuffle=True):
    images_folder = os.path.join(data_folder, 'images')
    masks_folder = os.path.join(data_folder, 'masks')

    image_filenames = os.listdir(images_folder)
    image_paths = [os.path.join(images_folder, fname) for fname in image_filenames]
    mask_paths = [os.path.join(masks_folder, fname.replace('.jpg', '.png')) for fname in image_filenames]

    # Create a dataset of file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    # Load and preprocess images and masks
    dataset = dataset.map(load_image_and_mask)
    dataset = dataset.map(preprocess_data)

    # Shuffle and batch the dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    dataset = dataset.batch(batch_size)

    return dataset

from unet_model import dice_loss, dice_score
from data_preprocessing import create_dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32


# Function to evaluate the model and visualize predicted masks
def evaluate(model, data_generator):
    random_index = tf.random.uniform(shape=[], minval=0, maxval=BATCH_SIZE, dtype=tf.int32).numpy()

    # Unpack the data generator
    batch = data_generator.take(1).get_single_element()
    images = batch[0]
    true_masks = batch[1]

    # Perform inference
    prediction = model.predict(images)

    selected_mask = prediction[random_index]

    # Convert the selected prediction to a binary mask (thresholding at 0.5, assuming sigmoid activation)
    binary_mask = tf.cast(selected_mask > 0.5, dtype=tf.uint8)

    # Squeeze dimensions with size 1
    images_squeezed = tf.squeeze(images[random_index])
    true_masks_squeezed = tf.squeeze(true_masks[random_index])
    binary_mask_squeezed = tf.squeeze(binary_mask)

    return images_squeezed.numpy(), true_masks_squeezed.numpy(), binary_mask_squeezed.numpy()


if __name__ == "__main__":
    trained_model = tf.keras.models.load_model('airship_model.keras', custom_objects={'dice_loss': dice_loss, 'dice_score': dice_score})

    test_folder = "../test/"

    test_generator = create_dataset(test_folder, batch_size=BATCH_SIZE, shuffle=True)

    for i in range(10):
        # Make predictions
        image, true_mask, predicted_mask = evaluate(trained_model, test_generator)

        # Visualize the results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title('Original Image')

        axes[1].imshow(true_mask, cmap='gray')
        axes[1].set_title('True Mask')

        axes[2].imshow(predicted_mask, cmap='gray')
        axes[2].set_title('Predicted Mask')

        plt.show()

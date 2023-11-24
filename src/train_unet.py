import tensorflow as tf
from keras.optimizers import Adam
from data_preprocessing import create_dataset
from unet_model import build_unet, dice_loss, dice_score
import pickle
import keras
import random
import matplotlib as plt


BATCH_SIZE = 64
NUM_OF_EPOCHS = 15
test_folder = "../test/"
test_generator = create_dataset(test_folder, batch_size=BATCH_SIZE, shuffle=True)


class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, epoch_interval=None):
        self.epoch_interval = epoch_interval

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_interval and epoch % self.epoch_interval == 0:
            pred_masks = self.model.predict(test_generator)
            pred_masks = tf.math.argmax(pred_masks, axis=-1)
            pred_masks = pred_masks[..., tf.newaxis]

            # Randomly select an image from the test batch
            random_index = random.randint(0, BATCH_SIZE - 1)
            random_image = test_generator[random_index]
            random_pred_mask = pred_masks[random_index]
            random_true_mask = test_generator[random_index]

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
            ax[0].imshow(random_image)
            ax[0].set_title(f"Image: {epoch:03d}")

            ax[1].imshow(random_true_mask)
            ax[1].set_title(f"Ground Truth Mask: {epoch:03d}")

            ax[2].imshow(random_pred_mask)
            ax[2].set_title(
                f"Predicted Mask: {epoch:03d}",
            )

            plt.show()
            plt.close()


callbacks = [DisplayCallback(1)]


def train_unet(train_folder,
               val_folder,
               batch_size,
               epochs=10,
               model_save_path='airship_model.h5',
               history_save_path='training_history.pkl'):
    # Create data generators for training and validation
    train_generator = create_dataset(train_folder, batch_size=batch_size, shuffle=True)
    val_generator = create_dataset(val_folder, batch_size=batch_size, shuffle=False)

    # Build the U-Net model
    model = build_unet()

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=[dice_score])

    # Create a callback to save the model weights
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)

    # Train the model
    history = model.fit(
        x=train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )

    # Save the training history
    with open(history_save_path, 'wb') as file:
        pickle.dump(history.history, file)


if __name__ == "__main__":
    train_folder = "../train/"
    val_folder = "../validation/"
    train_unet(train_folder, val_folder, epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE)

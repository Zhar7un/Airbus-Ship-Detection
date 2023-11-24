import tensorflow as tf
from keras.optimizers import Adam
from data_preprocessing import create_dataset
from unet_model import build_unet, dice_loss, dice_score
import pickle


BATCH_SIZE = 64
NUM_OF_EPOCHS = 30
test_folder = "../test/"
test_generator = create_dataset(test_folder, batch_size=BATCH_SIZE, shuffle=True)


def train_unet(train_folder,
               val_folder,
               batch_size,
               epochs=10,
               model_save_path='airship_model.keras',
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

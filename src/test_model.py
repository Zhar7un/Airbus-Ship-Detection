import keras
from unet_model import dice_loss, dice_score
from data_preprocessing import create_dataset


BATCH_SIZE = 32

model = keras.models.load_model('airship_model.h5', custom_objects={'dice_loss': dice_loss, 'dice_score': dice_score})
test_folder = "../test/"

test_generator = create_dataset(test_folder, batch_size=BATCH_SIZE, shuffle=True)
predictions = model.predict(test_generator)

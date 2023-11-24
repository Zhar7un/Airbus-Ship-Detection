import tensorflow as tf
from keras import layers


def build_unet(input_shape=(128, 128, 3), num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for filters in [64, 128, 256]:
        residual = x
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(residual)
        x = layers.add([x, residual])

    # Decoder
    for filters in [256, 128, 64, 32]:
        residual = x
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(residual)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])

    # Output layer
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def dice_score(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    return 1 - dice_score(y_true, y_pred)

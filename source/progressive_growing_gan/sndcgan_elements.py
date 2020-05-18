import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from .config import gen_batch_norm_decay, epsilon
from .spectral_normalization import SpectralNormalization


def make_generator(h=128, w=128, channels=3, noise_dims=128):
    '''
    Creates the generator model for SNDCGAN.
    :return: the generator model
    '''

    model = Sequential()
    model.add(layers.Dense(h // 8 * w // 8 * 512, use_bias=False, input_shape=(noise_dims,)))
    model.add(layers.BatchNormalization(momentum=gen_batch_norm_decay, epsilon=epsilon))
    model.add(layers.Activation('relu'))
    model.add(layers.Reshape(target_shape=(h // 8, w // 8, 512)))
    assert model.output_shape == (None, h // 8, w // 8, 512)

    model.add(
        layers.Convolution2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), use_bias=False, padding='same'))
    model.add(layers.BatchNormalization(momentum=gen_batch_norm_decay, epsilon=epsilon))
    model.add(layers.Activation('relu'))
    assert model.output_shape == (None, h // 4, w // 4, 256), \
        f'True output shape: {model.output_shape}, expected (None, {h // 4}, {w // 4}, 256)'

    model.add(
        layers.Convolution2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), use_bias=False, padding='same'))
    model.add(layers.BatchNormalization(momentum=gen_batch_norm_decay, epsilon=epsilon))
    model.add(layers.Activation('relu'))
    assert model.output_shape == (None, h // 2, w // 2, 128), \
        f'True output shape: {model.output_shape}, expected (None, {h // 2}, {w // 2}, 128)'

    model.add(layers.Convolution2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), use_bias=False, padding='same'))
    model.add(layers.BatchNormalization(momentum=gen_batch_norm_decay, epsilon=epsilon))
    model.add(layers.Activation('relu'))
    assert model.output_shape == (None, h, w, 64), \
        f'True output shape: {model.output_shape}, expected (None, {h}, {w}, 64)'

    model.add(layers.Convolution2DTranspose(filters=channels, kernel_size=(3, 3), strides=(1, 1), use_bias=False,
                                            padding='same'))
    model.add(layers.Activation('tanh'))
    assert model.output_shape == (None, h, w, channels), \
        f'True output shape: {model.output_shape}, expected (None, {h}, {w}, {channels})'

    return model


def make_discriminator(h=128, w=128, channels=3):
    '''
    Creates the discriminator model for SNDCGAN.
    :param h: height of the image
    :param w: width of the image
    :return: the discriminator model
    '''

    lrelu = lambda x: tf.nn.leaky_relu(x, 0.1)
    model = Sequential()

    model.add(SpectralNormalization(layers.Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                                   input_shape=(h, w, channels), activation=lrelu, padding='same')))

    model.add(SpectralNormalization(layers.Convolution2D(filters=128, kernel_size=(4, 4), strides=(2, 2),
                                   activation=lrelu, padding='same')))

    model.add(SpectralNormalization(layers.Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                                   activation=lrelu, padding='same')))

    model.add(SpectralNormalization(layers.Convolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2),
                                   activation=lrelu, padding='same')))

    model.add(SpectralNormalization(layers.Convolution2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                                   activation=lrelu, padding='same')))

    model.add(SpectralNormalization(layers.Convolution2D(filters=512, kernel_size=(4, 4), strides=(2, 2),
                                   activation=lrelu, padding='same')))

    model.add(SpectralNormalization(layers.Convolution2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                                   activation=lrelu, padding='same')))

    model.add(SpectralNormalization(layers.Dense(1)))  # no sigmoid activation cause logits.

    return model

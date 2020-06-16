import tensorflow as tf

from tensorflow.keras import layers, constraints
from .spectral_normalization import SpectralNormalization
from .config import gen_batch_norm_decay, epsilon

import numpy as np

lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)


# Taken from "Progressive Growing of GANs", https://github.com/tkarras/progressive_growing_of_gans
def n_filters(res, fmap_base=8192, fmap_decay=1.0, fmap_max=256):
    return min(int(fmap_base / (2.0 ** (res * fmap_decay))), fmap_max)


# TODO mydense, myconv

class MappingScale(layers.Layer):
    def __init__(self, noise_dims, map_depth, **kwargs):
        super().__init__(**kwargs)
        self.map_depth = map_depth
        self.noise_dims = noise_dims

    def build(self, input_shape):
        for i in range(self.map_depth):
            setattr(self, f'map_{i}', WeightScaleDense(self.noise_dims, name=f'mapdense_{i}'))


    def call(self, inputs, **kwargs):
        x = inputs
        for i in range(self.map_depth):
            x = getattr(self, f'map_{i}')(x)
        return x


class WeightScaleConv(layers.Layer):

    def __init__(self, filters, kernel_size, gain=np.sqrt(2), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.gain = gain
        self.kernel_size = kernel_size

    def build(self, input_shape):
        kernel_shape = [self.kernel_size, self.kernel_size, input_shape[-1], self.filters]
        self.w = self.add_weight(name='kernel', shape=kernel_shape, trainable=True,
                                 initializer='random_normal', dtype='float32')
        self.scale = tf.constant(self.gain / np.sqrt(np.prod(kernel_shape[:-1])), dtype='float32')

    def call(self, inputs, **kwargs):
        return tf.nn.conv2d(inputs, self.w * self.scale, strides=1, name='conv', padding='SAME')



class WeightScaleDense(layers.Layer):
    def __init__(self, units, gain=np.sqrt(2), **kwargs):
        super().__init__(**kwargs)
        self.filters = units
        self.gain = gain

    def build(self, input_shape):
        flatshape = np.prod(input_shape[1:].as_list())
        self.reshape = layers.Flatten()

        kernel_shape = [flatshape, self.filters]
        self.w = self.add_weight(name='kernel', shape=kernel_shape, trainable=True,
                                 initializer='random_normal', dtype='float32')
        import pdb; pdb.set_trace()
        self.scale = tf.constant(self.gain / np.sqrt(np.prod(kernel_shape[:-1])), dtype='float32')

    def call(self, inputs, **kwargs):
        x = self.reshape(inputs)
        res = tf.matmul(x, self.w * self.scale)
        return res


class BiasApplier(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        shape = [1 for _ in input_shape]
        shape[-1] = input_shape[-1]
        self.b = self.add_weight(name='bias', shape=shape, initializer='zeros', dtype='float32')

    def call(self, inputs, **kwargs):
        return inputs + self.b


class DiscriminatorBlockProgressive(layers.Layer):

    def __init__(self, res, use_sn=False, **kwargs):
        super().__init__(**kwargs)
        self.res = res
        self.use_sn = use_sn

    def build(self, input_shape):
        self.conv_1 = WeightScaleConv(filters=n_filters(self.res - 1), kernel_size=3, name='conv_1')
        if self.use_sn:
            self.conv_1 = SpectralNormalization(self.conv_1)
        self.bias_1 = BiasApplier(name='bias_1')
        self.act_1 = layers.Activation(lrelu)
        self.conv_2 = WeightScaleConv(filters=n_filters(self.res - 2), kernel_size=3, name='conv_2')
        if self.use_sn:
            self.conv_2 = SpectralNormalization(self.conv_2)
        self.bias_2 = BiasApplier(name='bias_2')
        self.act_2 = layers.Activation(lrelu)
        self.downscale = layers.AveragePooling2D(name='avg_pool')

    def call(self, x, **kwargs):
        x = self.act_1(self.bias_1(self.conv_1(x)))
        x = self.act_2(self.bias_2(self.conv_2(x)))
        x = self.downscale(x)
        return x


class MinibatchStddevLayer(layers.Layer):

    def __init__(self, group_size, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        '''
        Implemented according to: https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
        :param inputs:
        :param kwargs:
        :return:
        '''
        shape = tf.shape(inputs)  # NHWC
        x = tf.reshape(inputs, [self.group_size, -1, shape[1], shape[2], shape[3]])  # GMHWC
        x -= tf.reduce_mean(x, axis=0, keepdims=True)  # GMHWC
        x = tf.reduce_mean(tf.square(x), axis=0)  # MHWC
        x = tf.sqrt(x + 1e-8)
        x = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)  # M111
        x = tf.tile(x, [self.group_size, shape[1], shape[2], 1])  # NHW1
        x = tf.concat([inputs, x], axis=3)  # NHWC
        return x


class DiscriminatorFinalBlockProgressive(layers.Layer):
    # res = 4x4
    def __init__(self, group_size, res, use_sn=False, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size
        self.res = res

    def build(self, input_shape):
        self.mbstddev = MinibatchStddevLayer(self.group_size)
        self.conv_1 = WeightScaleConv(filters=n_filters(self.res - 1), kernel_size=3, name='conv_1')
        if self.use_sn:
            self.conv_1 = SpectralNormalization(self.conv_1)
        self.bias_1 = BiasApplier(name='bias_1')
        self.act_1 = layers.Activation(lrelu)
        self.dense_1 = WeightScaleDense(n_filters(self.res - 2), name='dense_1')
        if self.use_sn:
            self.dense_1 = SpectralNormalization(self.dense_1)
        self.bias_2 = BiasApplier(name='bias_2')
        self.act_2 = layers.Activation(lrelu)
        self.dense_2 = WeightScaleDense(1, name='dense_2', gain=1)
        if self.use_sn:
            self.dense_2 = SpectralNormalization(self.dense_2)
        self.bias_3 = BiasApplier(name='bias_3')

    def call(self, x, **kwargs):
        x = self.mbstddev(x)
        x = self.act_1(self.bias_1(self.conv_1(x)))
        x = self.act_2(self.bias_2(self.dense_1(x)))
        x = self.bias_3(self.dense_2(x))
        return x


class FromRGB(layers.Layer):

    def __init__(self, res, use_sn=False, **kwargs):
        super().__init__(**kwargs)
        self.res = res
        self.use_sn = use_sn

    def build(self, input_shape):
        self.conv = WeightScaleConv(filters=n_filters(self.res - 1), kernel_size=1, name='conv')
        if self.use_sn:
            self.conv = SpectralNormalization(self.conv)
        self.bias = BiasApplier(name='bias')
        self.act = layers.Activation(lrelu)

    def call(self, inputs, **kwargs):
        return self.act(self.bias(self.conv(inputs)))


class PixelNorm(layers.Layer):

    def __init__(self, e=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.e = e

    def call(self, x, **kwargs):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.e)

    def compute_output_shape(self, input_shape):
        return input_shape


class GeneratorBlockProgressive(layers.Layer):

    def __init__(self, res, **kwargs):
        super().__init__(**kwargs)
        self.res = res

    def build(self, input_shape):
        self.upscale = layers.UpSampling2D(name='upscale')
        self.conv_1 = WeightScaleConv(filters=n_filters(self.res - 1), kernel_size=3, name='conv_1')
        self.bias_1 = BiasApplier(name='bias_1')
        self.act_1 = layers.Activation(lrelu)
        self.pixel_norm1 = PixelNorm(name='pn1')
        self.conv_2 = WeightScaleConv(filters=n_filters(self.res - 1), kernel_size=3, name='conv_2')
        self.bias_2 = BiasApplier(name='bias_2')
        self.act_2 = layers.Activation(lrelu)
        self.pixel_norm2 = PixelNorm(name='pn2')

    def call(self, x, **kwargs):
        x = self.upscale(x)
        x = self.conv_1(x)
        x = self.pixel_norm1(self.act_1(self.bias_1(x)))
        x = self.conv_2(x)
        x = self.pixel_norm2(self.act_2(self.bias_2(x)))
        return x


class GeneratorFirstBlockProgressive(layers.Layer):
    # 4x4 res
    def __init__(self, res, **kwargs):
        super().__init__(**kwargs)
        self.res = res

    def build(self, input_shape):
        self.input_pn = PixelNorm(name='input')
        self.dense = WeightScaleDense(n_filters(self.res - 1) * 16, name='dense', gain=np.sqrt(2) / 4)
        self.reshape = layers.Reshape((4, 4, n_filters(self.res - 1)), name='reshape')
        self.bias_1 = BiasApplier(name='bias_1')
        self.act_1 = layers.Activation(lrelu)
        self.pixel_norm1 = PixelNorm(name='pn1')
        self.conv = WeightScaleConv(filters=n_filters(self.res - 1), kernel_size=3, name='conv')
        self.bias_2 = BiasApplier(name='bias_2')
        self.act_2 = layers.Activation(lrelu)
        self.pixel_norm2 = PixelNorm(name='pn2')

    def call(self, x, **kwargs):
        x = self.input_pn(x)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.pixel_norm1(self.act_1(self.bias_1(x)))
        x = self.pixel_norm2(self.act_2(self.bias_2(self.conv(x))))
        return x


class ToRGB(layers.Layer):

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.conv = WeightScaleConv(filters=self.channels, kernel_size=1, name='conv', gain=1)
        self.bias = BiasApplier(name='bias')

    def call(self, x, **kwargs):
        return self.bias(self.conv(x))

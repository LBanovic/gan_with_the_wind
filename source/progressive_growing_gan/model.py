import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

import gan
import losses
import parameter_config as pc
from .layers import GeneratorBlockProgressive, GeneratorFirstBlockProgressive, DiscriminatorBlockProgressive, \
    ToRGB, FromRGB, DiscriminatorFinalBlockProgressive


def generator(resolution, channels, noise_dims, map_depth=4):
    class Generator(Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.resolution = resolution
            self.map_depth = map_depth
            # mapping network
            # for i in range(map_depth):
            #     if i == 0:
            #         setattr(self, f'map_{i}',
            #                 layers.Dense(noise_dims, activation='relu', input_shape=(noise_dims,), name=f'map_{i}',
            #                              kernel_initializer='random_normal'))
            #     else:
            #         setattr(self, f'map_{i}',
            #                 layers.Dense(noise_dims, activation='relu', name=f'map_{i}', input_shape=(noise_dims,),
            #                              kernel_initializer='random_normal'))

            self.first_layer = GeneratorFirstBlockProgressive(2, name='input_layer')

            for res in range(3, resolution + 1):
                setattr(self, f'gb_{res}', GeneratorBlockProgressive(res, name=f'gen_block_{res}'))

            if resolution > 2:
                self.help_upsample = layers.UpSampling2D(name='helper')

            for res in range(max(2, resolution - 1), resolution + 1):
                setattr(self, f'out_{res}', ToRGB(channels, name=f'to_rgb_{res}'))

        def call(self, x, training=None, alpha=1.0):
            for i in range(map_depth):
                x = getattr(self, f'map_{i}')(x)
            x = self.first_layer(x)
            for res in range(3, self.resolution):
                x = getattr(self, f'gb_{res}')(x, training=training)
            if self.resolution == 2:
                return self.out_2(x, training=training)
            elif np.isclose(alpha, 1.0):
                x = getattr(self, f'gb_{self.resolution}')(x, training=training)
                return getattr(self, f'out_{self.resolution}')(x, training=training)
            else:
                y = getattr(self, f'gb_{self.resolution}')(x, training=training)
                y = getattr(self, f'out_{self.resolution}')(y, training=training)
                x = getattr(self, f'out_{self.resolution - 1}')(x, training=training)
                x = self.help_upsample(x)
                return (1 - alpha) * x + alpha * y

        def store(self, dir):
            gen_string = 'generator'
            for layer in self.layers:
                weights = layer.get_weights()
                for i, weight in enumerate(weights):
                    np.save(os.path.join(dir, f'{gen_string}_{layer.name}_{i}.npy'), weight)

        def restore(self, dir):
            gen_string = 'generator'
            for layer in self.layers:
                weights = []
                layer_file_prefix = f'{gen_string}_{layer.name}'
                for i, _ in enumerate([file for file in os.listdir(dir) if file.startswith(layer_file_prefix)]):
                    layer_weight_file = os.path.join(dir, f'{layer_file_prefix}_{i}.npy')
                    weight = np.load(layer_weight_file)
                    weights.append(weight)
                if len(weights) > 0:
                    try:
                        layer.set_weights(weights)
                    except ValueError:
                        print(
                            f'Warning: could not restore weights in {layer.name} since it is not used during layer call.')
                else:
                    print(f'Warning: could not restore weights in {layer.name} since no weights exist.')

    return Generator()


def discriminator(resolution):
    class Discriminator(Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.resolution = resolution

            setattr(self, f'layer_in_{resolution}', FromRGB(resolution, name=f'input_{resolution}'))
            if resolution > 2:
                setattr(self, f'layer_in_{resolution - 1}', FromRGB(resolution - 1, name=f'input_{resolution - 1}'))

            if resolution > 2:
                self.help_downscale = layers.AveragePooling2D(name='downscale')

            for res in range(resolution, 2, -1):
                setattr(self, f'db_{res}', DiscriminatorBlockProgressive(res, name=f'disc_block_{res}'))

            self.out = DiscriminatorFinalBlockProgressive(4, 2, name='disc_final')

        def call(self, x, training=None, alpha=1.0):
            y = getattr(self, f'layer_in_{self.resolution}')(x)
            if self.resolution > 2:
                y = getattr(self, f'db_{self.resolution}')(y)
                if not np.isclose(alpha, 1.0):
                    # TODO deduce what is wrong
                    x = self.help_downscale(x)
                    x = getattr(self, f'layer_in_{self.resolution - 1}')(x)
                    x = (1 - alpha) * x + alpha * y
                else:
                    x = y
                for res in range(self.resolution - 1, 2, -1):
                    x = getattr(self, f'db_{res}')(x)
            else:
                x = y
            return self.out(x)

        def store(self, dir):
            gen_string = 'discriminator'
            for layer in self.layers:
                weights = layer.get_weights()
                for i, weight in enumerate(weights):
                    np.save(os.path.join(dir, f'{gen_string}_{layer.name}_{i}.npy'), weight)

        def restore(self, dir):
            gen_string = 'discriminator'
            for layer in self.layers:
                weights = []
                layer_file_prefix = f'{gen_string}_{layer.name}'
                for i, _ in enumerate([file for file in os.listdir(dir) if file.startswith(layer_file_prefix)]):
                    layer_weight_file = os.path.join(dir, f'{layer_file_prefix}_{i}.npy')
                    weight = np.load(layer_weight_file)
                    weights.append(weight)
                if len(weights) > 0:
                    try:
                        layer.set_weights(weights)
                    except ValueError:
                        print(
                            f'Warning: could not restore weights in {layer.name} since it is not used during layer call.')
                else:
                    print(f'Warning: could not restore weights in {layer.name} since no weights exist.')

    return Discriminator()


def sndcgan(model_dir, resolution, channels, input_dims, g_lr, d_lr):
    generator_optimizer = tf.keras.optimizers.Adam(g_lr,
                                                   beta_1=pc.adam_beta1,
                                                   beta_2=pc.adam_beta2)
    discriminator_optimizer = tf.keras.optimizers.Adam(d_lr,
                                                       beta_1=pc.adam_beta1,
                                                       beta_2=pc.adam_beta2)

    class ProgGAN(gan.GAN):

        def __init__(self, model_dir, generator, discriminator, generator_optimizer, discriminator_optimizer):
            generator_loss = losses.generator_wgan_loss
            discriminator_loss = losses.discriminator_wgan_loss
            super().__init__(model_dir, generator, discriminator, generator_loss, discriminator_loss,
                             generator_optimizer, discriminator_optimizer)

        def get_losses(self, real_images, noise, **kwargs):
            fake_images = self.generator(noise, **kwargs)

            real_scores = self.discriminator(real_images, **kwargs)
            fake_scores = self.discriminator(fake_images, **kwargs)

            generator_loss = self.generator_loss(fake_scores)
            disc_loss = self.discriminator_loss(real_scores, fake_scores)
            grad_penalty = losses.grad_penalty(real_scores, fake_scores, self.discriminator, real_images, fake_images)
            return tf.reduce_mean(generator_loss), tf.reduce_mean(disc_loss + grad_penalty)

    nn = ProgGAN(model_dir,
                 generator=generator(resolution, channels, input_dims, map_depth=0),
                 discriminator=discriminator(resolution),
                 generator_optimizer=generator_optimizer,
                 discriminator_optimizer=discriminator_optimizer)
    return nn

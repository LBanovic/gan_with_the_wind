import tensorflow as tf
import os
import numpy as np
import shutil


class GAN:

    def __init__(self, model_dir, generator, discriminator, generator_loss, discriminator_loss,
                 generator_optimizer, discriminator_optimizer):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.model_dir = model_dir


    def train_on_batch(self, images, noise, **kwargs):
        # TODO split losses, first update discriminator, then generator
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_loss, disc_loss = self.get_losses(images, noise, training=True, **kwargs)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

    def get_losses(self, images, noise, **kwargs):
        generated_images = self.generator(noise, **kwargs)

        real_output = self.discriminator(images, **kwargs)
        fake_output = self.discriminator(generated_images, **kwargs)

        gen_loss = self.generator_loss(fake_output)
        disc_loss = self.discriminator_loss(real_output, fake_output)
        return tf.reduce_mean(gen_loss), tf.reduce_mean(disc_loss)

    def make_checkpoint(self, epoch, store_optimizers=False):
        dir = os.path.join(self.model_dir, f'{epoch}_epoch')
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)
        self.generator.store(dir)
        self.discriminator.store(dir)
        if store_optimizers:
            self._store_optimizers(dir)

    def _store_optimizers(self, dir):
        genopt_w = self.generator_optimizer.get_weights()
        discopt_w = self.discriminator_optimizer.get_weights()
        genopt_savepath = os.path.join(dir, 'generator_optimizer.npy')
        discopt_savepath = os.path.join(dir, 'discriminator_optimizer.npy')
        np.save(genopt_savepath, genopt_w)
        np.save(discopt_savepath, discopt_w)

    def _restore_optimizers(self, dir):
        genopt_savepath = os.path.join(dir, 'generator_optimizer.txt')
        discopt_savepath = os.path.join(dir, 'discriminator_optimizer.txt')
        genopt_w = np.load(genopt_savepath)
        discopt_w = np.load(discopt_savepath)
        self.generator_optimizer.set_weights(genopt_w)
        self.discriminator_optimizer.set_weights(discopt_w)

    def restore_from_checkpoint(self, model_dir, epoch, reset_optimizer=True):
        dir = os.path.join(model_dir, f'{epoch}_epoch')
        self.generator.restore(dir)
        self.discriminator.restore(dir)
        if not reset_optimizer:
            self._restore_optimizers(dir)

    def generate(self, noise):
        return self.generator(noise, training=False)

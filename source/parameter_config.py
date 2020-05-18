# parameters for celebahq128 dataset
# copied from https://github.com/google/compare_gan

# general params
batch_size = 64
steps_in_epoch = 10_000
epochs = 15
noise_dims = 128
adam_beta1 = 0.0
adam_beta2 = 0.999

# generator
g_lr = 0.0002

# discriminator
d_lr = 0.0001

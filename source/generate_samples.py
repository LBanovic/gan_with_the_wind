import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys

from progressive_growing_gan import model
from visualize_data import visualize
import cv2

res = 6
input_dim = 512
input_channels = 3
batch_size = 64




model_dir = sys.argv[1]
def get_savedir(res):
    savedir = os.path.join(model_dir, f'{2 ** res}x{2 ** res}')
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    return savedir


import numpy as np

seed1 = np.reshape(np.arange(100, 100 + 8192), [16, 512])
seed2 = np.reshape(np.arange(100 + 8192, 100 + 8192 + 8192), [16, 512])
seed3 = np.reshape(np.arange(100 + 8192 + 8192, 100 + 8192 + 8192 + 8192), [16, 512])

seed1 = tf.random.stateless_truncated_normal([4, 512], seed=[100, 200])
seed2 = tf.random.stateless_truncated_normal([4, 512], seed=[8292, 16484])
seed3 = tf.random.stateless_truncated_normal([4, 512], seed=[200_000, 28292])

seed = seed1 - seed2 + seed3


gan = model.sndcgan(get_savedir(res), res, input_channels, input_dim, 0, 0)
gan.generator(tf.zeros((batch_size, input_dim)), training=False, alpha=0)
gan.discriminator(tf.zeros((batch_size, 2 ** res, 2 ** res, input_channels)), training=False, alpha=0)

gan.restore_from_checkpoint(get_savedir(res), epoch='1_50000')

fps = 25
length = 60


images_1 = gan.generate(seed1)

images_2 = gan.generate(seed2)

images_3 = gan.generate(seed3)

images = gan.generate(seed)

all = np.vstack([images_1, images_2, images_3, images])

visualize(all)
plt.show()

# import numpy as np
# images = np.zeros((fps * length, 64, 64, 3))
#
# alpha = np.linspace(0, 1, fps).reshape((fps, 1))
# alpha = np.tile(alpha, [1, 512])
#
# assert alpha.shape == (fps, 512), alpha.shape
#
# seed1 = tf.random.truncated_normal([1, input_dim]).numpy()
#
# for i in range(length):
#     print(i, '\r', end='')
#     seed2 = tf.random.truncated_normal([1, input_dim]).numpy()
#     seed = seed1 + (seed2 - seed1) * alpha
#     generated = gan.generator(seed)
#     images[i * fps:(i + 1) * fps] = generated.numpy()
#     seed1 = seed2
#
# images = images * 127.5 + 127.5
# images = np.rint(images).clip(0, 255).astype(np.uint8)
#
# writer = cv2.VideoWriter('images/interpolation2/interpolation.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (64, 64))
# for i, image in enumerate(images):
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     writer.write(image)
# writer.release()
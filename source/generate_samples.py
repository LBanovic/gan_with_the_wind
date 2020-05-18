import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys

from sndcgan import model
from visualize_data import visualize

res = 2
input_dim = 512
input_channels = 3
batch_size = 64


model_dir = sys.argv[1]
def get_savedir(res):
    savedir = os.path.join(model_dir, f'{2 ** res}x{2 ** res}')
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    return savedir


num_showcase_imgs = 16
seed = tf.random.normal([num_showcase_imgs, input_dim])

gan = model.sndcgan(get_savedir(res), res, input_channels, input_dim, 0, 0)
gan.generator(tf.zeros((batch_size, input_dim)), training=False, alpha=0)
gan.discriminator(tf.zeros((batch_size, 2 ** res, 2 ** res, input_channels)), training=False, alpha=0)
import pdb; pdb.set_trace()

gan.restore_from_checkpoint(get_savedir(res), epoch=3)

images = gan.generator(seed)
visualize(images)
plt.show()

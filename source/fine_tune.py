import sys
import time
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from visualize_data import visualize
from load_data.dataset_loader import CelebALoader
from progressive_growing_gan import model

model_dir = sys.argv[1]
celeba_loc = sys.argv[2]

fine_tune_epochs = 2
stabilize = True
start_epoch = 2
stabilizing_epochs = 4

res = 6
input_dim = 512
input_channels = 3

batch_size = 32
train_set_size = 200_000

g_lr = 0.0008
d_lr = 0.0008

checkpoint_every = 50_000 // batch_size

def get_alpha(batch_index, batch_epoch):
    num_batches = train_set_size / batch_size
    maxnum_batch = num_batches * stabilizing_epochs
    current_batch = (batch_epoch + start_epoch) * num_batches + batch_index
    return current_batch / maxnum_batch

def get_savedir(res):
    savedir = os.path.join(model_dir, f'{2 ** res}x{2 ** res}')
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    return savedir

def save_images(savedir, epoch, predictions):
    if not os.path.exists(f'{savedir}/images'):
        os.mkdir(f'{savedir}/images')
    visualize(predictions)
    plt.savefig(f'{savedir}/images/image_at_epoch_{epoch}.png')
    plt.close()

def epoch_cleanup(gan, epoch, batch_num, savedir, seed):
    log = f'{epoch}_{batch_num}'
    gan.make_checkpoint(log)
    save_images(savedir, log, gan.generate(seed))

num_showcase_imgs = 16
seed = tf.random.normal([num_showcase_imgs, input_dim])

train_data, test_data = CelebALoader(celeba_loc, res, train_set_size=train_set_size).load(batch_size, input_dim)

savedir = get_savedir(res)
gan = model.sndcgan(savedir, res, input_channels, input_dim, g_lr, d_lr)
gan.generator(tf.zeros((batch_size, input_dim)), training=False, alpha=0)
gan.discriminator(tf.zeros((batch_size, 2 ** res, 2 ** res, input_channels)), training=False, alpha=0)
gan.generator.summary()
gan.discriminator.summary()

print(f'Starting fine tune. Resolution: {2**res}x{2**res}, batch size: {batch_size}')
gan.restore_from_checkpoint(savedir, start_epoch)
print(f'Restored from epoch {start_epoch}.')

for epoch in range(fine_tune_epochs):
    epoch_cleanup(gan, start_epoch + epoch, 0, savedir, seed)
    print(f'Current epoch: {epoch + start_epoch}')
    s = time.time()
    for j, (images, noise) in enumerate(train_data):
        if stabilize:
            alpha = get_alpha(j + 1, epoch)
        else:
            alpha = 1
        gan.train_on_batch(images, noise, alpha=alpha)
        if (j + 1) % checkpoint_every == 0:
            epoch_cleanup(gan, epoch + start_epoch, (j + 1) * batch_size, savedir, seed)
            t = time.time()
            print(f'{(j + 1) * batch_size} images shown in {t - s}')
            s = t

epoch_cleanup(gan, fine_tune_epochs + start_epoch, 0, savedir, seed)
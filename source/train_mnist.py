import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
import matplotlib.pyplot as plt
import sys
import os
import time
import numpy as np

from progressive_growing_gan import model
from visualize_data import visualize

from load_data.dataset_loader import MnistLoader
from metrics import FID_other

model_dir = sys.argv[1]
# run params
start_res = 2
maxres = 6

stabilize_epochs = 4
learn_epochs = 4
map_depth = 4
input_dim = 128
input_channels = 1

batch_size = 64
train_set_size = 60_000

g_lr = 0.001
d_lr = 0.001

start = time.time()

def get_alpha(batch_index, batch_epoch):
    num_batches = train_set_size / batch_size
    maxnum_batch = num_batches * stabilize_epochs
    current_batch = batch_epoch * num_batches + batch_index
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
    plt.savefig(f'{savedir}/images/image_at_epoch_{epoch:04d}.png')
    plt.close()

def epoch_cleanup(gan, epoch, res, savedir, seed, test_data=None):
    gan.make_checkpoint(epoch)
    save_images(savedir, epoch, gan.generate(seed))

num_showcase_imgs = 16
seed = tf.random.normal([num_showcase_imgs, input_dim])

overall_epoch = 0
restore_from_epoch = 7
log_every = 200

best_fid = np.inf
patience = 3
waiting = 0

# _, test_data = MnistLoader(5).load(batch_size, input_dim)

for res in range(start_res, maxres + 1):
    tf.keras.backend.clear_session()
    savedir = get_savedir(res)
    gan = model.gan_mnist(savedir, res, input_dim, map_depth, g_lr, d_lr)
    train_data, test_data = MnistLoader(res).load(batch_size, input_dim)
    print(f'Training on resolution {2 ** res}x{2 ** res}')
    # need to run through just so the model builds
    gan.generator(np.zeros((batch_size, input_dim)), training=False, alpha=0)
    gan.discriminator(np.zeros((batch_size, 2 ** res, 2 ** res, input_channels)), training=False, alpha=0)
    gan.generator.summary()
    gan.discriminator.summary()
    if res > 2:
        if res == start_res:
            overall_epoch = restore_from_epoch + 1

        restore_from_epoch = overall_epoch - 1
        overall_epoch = 0

        print("Restoring from epoch", restore_from_epoch)
        gan.restore_from_checkpoint(get_savedir(res - 1), restore_from_epoch)
        print('Layers from previous resolution restored.')
        print('Stabilization phase.')
        for i in range(stabilize_epochs):
            print(f'Epoch {overall_epoch}')
            for j, (image, noise) in enumerate(train_data):
                if j % log_every == 0:
                    t = time.time()
                    print(f"{j * batch_size} images shown during stabilization in {t - start} s.")
                    start = t
                    fid = FID_other(gan, test_data)
                    if fid < best_fid:
                        print('Saved best FID', fid)
                        epoch_cleanup(gan, overall_epoch, res, savedir, seed, test_data=None)
                        best_fid = fid
                        waiting = 0
                    elif fid > best_fid:
                        waiting += 1
                        if waiting == patience:
                            break
                alpha = get_alpha(batch_index=j + 1, batch_epoch=i)
                gan.train_on_batch(image, noise, alpha=alpha)
            if waiting == patience:
                break
            epoch_cleanup(gan, overall_epoch, res, savedir, seed, test_data=None)
            overall_epoch += 1

    print('Learning phase.')
    for i in range(learn_epochs):
        print(f'Epoch {overall_epoch}')
        for j, (image, noise) in enumerate(train_data):
            if j % log_every == 0:
                t = time.time()
                print(f"{j * batch_size} images shown during learning in {t - start} s.")
                start = t
                fid = FID_other(gan, test_data)
                if fid < best_fid:
                    print('Saved best FID:', fid)
                    epoch_cleanup(gan, overall_epoch, res, savedir, seed, test_data=None)
                    best_fid = fid
                    waiting = 0
                elif fid > best_fid:
                    waiting += 1
                    if waiting == patience:
                        break
            gan.train_on_batch(image, noise)
        if waiting == patience:
            break
        epoch_cleanup(gan, overall_epoch, res, savedir, seed, test_data=None)
        overall_epoch += 1
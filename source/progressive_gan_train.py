import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
import matplotlib.pyplot as plt
import sys
import os
import time

from progressive_growing_gan import model
from visualize_data import visualize

from load_data.dataset_loader import CelebALoader

model_dir = sys.argv[1]
celeba_loc = sys.argv[2]
# run params
start_res = 6

stabilize_epochs = 4
learn_epochs = 4
input_dim = 512
input_channels = 3
maxres = 6
restore_from_epoch = 7

batch_size = 16
train_set_size = 200_000


g_lr = 0.001
d_lr = 0.001

log_every = 1000
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


log_file = f'{model_dir}/log.txt'

def log(res, epoch, gen_loss, disc_loss):
    with open(log_file, 'a') as logger:
        line = f'Epoch {epoch}, resolution {2 ** res}x{2 ** res}: gen_loss={gen_loss}, disc_loss={disc_loss}\n'
        logger.write(line)


def epoch_cleanup(gan, epoch, res, savedir, seed, test_data=None):
    gan.make_checkpoint(epoch)
    save_images(savedir, epoch, gan.generate(seed))
    gen_loss, disc_loss = 0, 0
    if test_data:
        for i, (image, noise) in enumerate(test_data):
            gen_loss_batch, disc_loss_batch = gan.get_losses(image, noise, training=False)
            gen_loss += gen_loss_batch
            disc_loss += disc_loss_batch
        gen_loss, disc_loss = gen_loss / (i + 1) / batch_size, disc_loss / (i + 1) / batch_size
        log(res, epoch, gen_loss, disc_loss)


num_showcase_imgs = 16
seed = tf.random.normal([num_showcase_imgs, input_dim])

# if len(os.listdir(model_dir)) != 0:
#    print('Model directory must be empty. Exiting.')
#    exit()

overall_epoch = 0
for res in range(start_res, maxres + 1):
    tf.keras.backend.clear_session()
    savedir = get_savedir(res)
    gan = model.sndcgan(savedir, res, input_channels, input_dim, g_lr, d_lr)
    train_data, test_data = CelebALoader(celeba_loc, res, train_set_size=train_set_size).load(batch_size, input_dim)
    print(f'Training on resolution {2 ** res}x{2 ** res}')
    # need to run through just so the model builds
    gan.generator(tf.zeros((batch_size, input_dim)), training=False, alpha=0)
    gan.discriminator(tf.zeros((batch_size, 2 ** res, 2 ** res, input_channels)), training=False, alpha=0)
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
                alpha = get_alpha(batch_index=j + 1, batch_epoch=i)
                gan.train_on_batch(image, noise, alpha=alpha)
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
            gan.train_on_batch(image, noise)
        epoch_cleanup(gan, overall_epoch, res, savedir, seed, test_data=None)
        overall_epoch += 1

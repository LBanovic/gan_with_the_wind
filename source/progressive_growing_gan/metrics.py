from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from load_data.dataset_loader import CelebALoader
from skimage.transform import resize
import scipy
import numpy as np
import os

def scale(images, shape):
    for image in images:
        yield resize(image, shape, 0)

def FID(model, dataset, shape=(128, 128, 3)):
    inception = InceptionV3(include_top=False, pooling='avg', input_shape=shape)
    images_fake = []
    images_real = []

    for real, noise in dataset:
        fake = model.generate(noise)
        fake = fake * 127.5 + 127.5
        real = real * 127.5 + 127.5
        images_fake.extend(scale(fake, shape))
        images_real.extend(scale(real, shape))

    images_fake = preprocess_input(images_fake)
    images_real = preprocess_input(images_real)
    feat_real = inception.predict(images_real)
    feat_fake = inception.predict(images_fake)
    mean_real, mean_fake = feat_real.mean(axis=0), feat_fake.mean(axis=0)
    cov_real, cov_fake = np.cov(feat_real, rowvar=False), np.cov(feat_fake, rowvar=False)

    fid = (mean_fake - mean_real) ** 2 + np.trace(cov_real + cov_fake - 2 * scipy.linalg.sqrtm(cov_real @ cov_fake + 1e-10))
    return fid


def get_savedir(res, model_dir):
    savedir = os.path.join(model_dir, f'{2 ** res}x{2 ** res}')
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    return savedir

if __name__ == '__main__':

    import sys
    from progressive_growing_gan import model

    input_dim = 128
    input_channels = 3

    model_dir = sys.argv[1]
    epoch = sys.argv[2]
    celeba_loc = sys.argv[3]
    res = int(sys.argv[4])

    gan = model.sndcgan(get_savedir(res, model_dir), res, input_channels, input_dim, 0, 0)
    gan.generator(np.zeros((64, input_dim)), training=False, alpha=0)
    gan.discriminator(np.zeros((64, 2 ** res, 2 ** res, input_channels)), training=False, alpha=0)
    gan.restore_from_checkpoint(get_savedir(res, model_dir), epoch=epoch)

    loader = CelebALoader(celeba_loc, res)
    train_ds, test_ds = loader.load(64, input_dim)

    # TESTING
    test_ds = test_ds.take(10)

    fid = FID(gan, test_ds)
    print(f'Resolution: {2 ** res}, FID = {fid}')
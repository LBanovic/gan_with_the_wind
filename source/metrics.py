from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3
from load_data.dataset_loader import CelebALoader
from skimage.transform import resize
import scipy
import numpy as np
import os

def scale(images, shape):
    for image in images:
        yield resize(image, shape, 0)


def FID(model, dataset, shape=(128, 128, 3)):
    feat_fakes = []
    feat_reals = []

    inception = InceptionV3(include_top=False, pooling='avg', input_shape=shape)
    for real, noise in dataset:
        fake = model.generate(noise)
        fake = np.array(list(scale(fake, shape)))
        real = np.array(list(scale(real, shape)))
        feat_real = inception.predict(real)
        feat_fake = inception.predict(fake)
        feat_fakes.extend(feat_fake)
        feat_reals.extend(feat_real)
    feat_fakes = np.array(feat_fakes)
    feat_reals = np.array(feat_reals)
    mean_real, mean_fake = feat_reals.mean(axis=0), feat_fakes.mean(axis=0)
    cov_real, cov_fake = np.cov(feat_reals, rowvar=False), np.cov(feat_fakes, rowvar=False)
    cov_mean = scipy.linalg.sqrtm(cov_real @ cov_fake + 1e-10)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = np.sum((mean_fake - mean_real)**2) + np.trace(cov_real + cov_fake - 2 * cov_mean)
    return fid



def get_savedir(res, model_dir):
    savedir = os.path.join(model_dir, f'{2 ** res}x{2 ** res}')
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    return savedir

if __name__ == '__main__':

    import sys
    import json

    from progressive_growing_gan import model

    input_dim = 128
    input_channels = 3
    batch_size = 128

    celeba_loc = sys.argv[1]
    models_dir = sys.argv[2]

    results = []

    loader = CelebALoader(celeba_loc, 6)
    train_ds, test_ds = loader.load(64, input_dim)
    for model_dir in os.listdir(models_dir):
        res = int(np.log2(int(model_dir.split('x')[0])))
        print(res)
        directory = os.path.join(models_dir, model_dir)
        gan = model.sndcgan(directory, res, input_channels, input_dim, 0, 0)
        gan.generator(np.zeros((64, input_dim)), training=False, alpha=0)
        gan.discriminator(np.zeros((64, 2 ** res, 2 ** res, input_channels)), training=False, alpha=0)
        for epoch in os.listdir(directory):
            if epoch != 'images':
                imdir = f'{res}_{epoch}_images'
                if not os.path.exists(imdir):
                    os.mkdir(imdir)
                epoch = epoch[:-len('_epoch')]
                gan.restore_from_checkpoint(directory, epoch)
                # for i, (_, noise) in enumerate(test_ds):
                #     images = gan.generate(noise)
                #     results.extend(images.numpy())
                # np.save(f'{imdir}-{2**res}x{2**res}-{epoch}', np.array(results))
                fid = FID(gan, test_ds)
                results.append(f'Resolution-Epoch: {2 ** res}-{epoch}, FID = {fid}')
                print(f'Resolution-Epoch: {2 ** res}-{epoch}, FID = {fid}')
        del gan
    with open(f'{models_dir}.json', 'w') as dirjson:
        json.dump(results, dirjson)
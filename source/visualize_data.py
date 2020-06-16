import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from load_data.dataset_loader import MnistLoader


def visualize(imgs):
    # import pdb; pdb.set_trace()
    scale = 255 / 2
    bias = 255 - scale
    if imgs.shape[-1] == 1:
        imgs = imgs[:, :, :, 0]
    for i, image in enumerate(imgs):
        plt.subplot(4, 4, i+1)
        image = image * scale + bias
        image = np.rint(image).clip(0, 255).astype(np.uint8)
        plt.imshow(image)
    plt.axis('off')


if __name__ == "__main__":
    train, test = MnistLoader(2).load(batch_size=16, noise_dims=128)
    imgs, noise = next(iter(train))
    visualize(imgs)
    plt.show()
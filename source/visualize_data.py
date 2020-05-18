import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from load_data.dataset_loader import CelebALoader


def visualize(imgs):
    # import pdb; pdb.set_trace()
    scale = 255 / 2
    bias = 255 - scale
    for i, image in enumerate(imgs):
        plt.subplot(4, 4, i+1)
        image = image * scale + bias
        image = np.rint(image).clip(0, 255).astype(np.uint8)
        plt.imshow(image)
    plt.axis('off')


if __name__ == "__main__":
    dataset = CelebALoader(r'C:\Users\lbanovic\Documents\FER\Diplomski rad\datasets\celeba\tfrecords', 5).load(batch_size=16, noise_dims=128)
    imgs, noise = next(iter(dataset))
    visualize(imgs)
    plt.show()
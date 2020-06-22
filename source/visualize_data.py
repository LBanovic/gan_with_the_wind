import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

from load_data.dataset_loader import CelebALoader


def visualize(imgs):
    scale = 255 / 2
    bias = 255 - scale
    if imgs.shape[-1] == 1:
        imgs = imgs[:, :, :, 0]
    for i, image in enumerate(imgs):
        plt.subplot(4, 4, i + 1)
        image = image * scale + bias
        image = np.rint(image).clip(0, 255).astype(np.uint8)
        plt.axis('off')
        plt.imshow(image)


if __name__ == "__main__":
    for res in range(2, 7):
        print('RES', res)
        train, test = CelebALoader(r'C:\Users\lbanovic\Documents\FER\Diplomski rad\datasets\celeba\tfrecords', res).load(batch_size=4, noise_dims=128)
        imgs, noise = next(iter(test))
        imgs = imgs.numpy()
        # for j, img in enumerate(imgs):
        #     img = img * 127.5 + 127.5
        #     img = np.rint(img).clip(0, 255).astype(np.uint8)
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     img = cv2.resize(img, (64, 64))
        #     cv2.imwrite(f'images/ground_truth/res_{i}_{j}.png', img)
        for _ in range(6 - res):
            imgs = imgs.repeat(2, axis=1).repeat(2, axis=2)

        scale = 255 / 2
        bias = 255 - scale
        for i, image in enumerate(imgs):
            plt.subplot(4, 1, i + 1)
            image = image * scale + bias
            image = np.rint(image).clip(0, 255).astype(np.uint8)
            plt.axis('off')
            plt.imshow(image)

        plt.savefig(f'images/ground_truth/res_{res}.png')
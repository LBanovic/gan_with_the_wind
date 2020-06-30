import argparse
import numpy as np
import cv2
import tensorflow as tf

from progressive_growing_gan import model

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', help='where the model is stored', required=True)
parser.add_argument('--epoch', help='which epoch is chosen (name of the epoch dir in model dir)', required=True)
parser.add_argument('--output', help='location of the produced .avi file', required=True)
parser.add_argument('--fps', default=25, type=int)
parser.add_argument('--duration', help='the duration of the resulting clip in seconds', default=60, type=int)

args = parser.parse_args()

RES = 6
INPUT_DIM = 512
INPUT_CHANNELS = 3
BATCH_SIZE = 64
FPS = args.fps
LENGTH = args.duration

model_dir = args.model_dir
epoch = args.epoch
output = args.output

if __name__ == '__main__':
    gan = model.sndcgan(model_dir, RES, INPUT_CHANNELS, INPUT_DIM, 0, 0)
    gan.generator(tf.zeros((BATCH_SIZE, INPUT_DIM)), training=False, alpha=0)
    gan.discriminator(tf.zeros((BATCH_SIZE, 2 ** RES, 2 ** RES, INPUT_CHANNELS)), training=False, alpha=0)

    gan.restore_from_checkpoint(model_dir, epoch=epoch)

    images = np.zeros((FPS * LENGTH, 64, 64, 3))

    alpha = np.linspace(0, 1, FPS).reshape((FPS, 1))
    alpha = np.tile(alpha, [1, 512])
    assert alpha.shape == (FPS, 512), alpha.shape

    seed1 = tf.random.truncated_normal([1, INPUT_DIM]).numpy()
    for i in range(LENGTH):
        print(i, '\r', end='')
        seed2 = tf.random.truncated_normal([1, INPUT_DIM]).numpy()
        seed = seed1 + (seed2 - seed1) * alpha
        generated = gan.generator(seed)
        images[i * FPS:(i + 1) * FPS] = generated.numpy()
        seed1 = seed2

    images = images * 127.5 + 127.5
    images = np.rint(images).clip(0, 255).astype(np.uint8)

    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'DIVX'), FPS, (64, 64))
    for i, image in enumerate(images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        writer.write(image)
    writer.release()
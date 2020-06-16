import abc
import os
import tensorflow as tf
import numpy as np
import cv2

class Loader(abc.ABC):

    @abc.abstractmethod
    def load(self, batch_size, noise_dims):
        pass


class CelebALoader(Loader):

    def __init__(self, tfrecords_dir, resolution, train_set_size=150_000):
        """
        Initializes the CelebALoader. 

        Parameters
        ----------
        tfrecords_dir: 
            Where Celeb-A tfrecords are stored
        resolution:
            resolution of the image denoted as a power of 2. For example, resolution of 3 would indicate 
            a 8x8 resolution.
        """
        # load only the version with the highest resolution
        self.train_set_size = train_set_size
        self.tfrecords = [os.path.join(tfrecords_dir, filename) for
                          filename in os.listdir(tfrecords_dir) if filename.endswith(f'r0{resolution}.tfrecords')]

        self.resolution = resolution

    def _tfrecord_parse(self, record):
        features = tf.io.parse_single_example(record, features={
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'data': tf.io.FixedLenFeature([], tf.string)})
        data = tf.io.decode_raw(features['data'], tf.uint8)
        data = tf.transpose(tf.reshape(data, features['shape']), perm=[1, 2, 0])
        data = (tf.cast(data, tf.float32) - 127.5) / 127.5
        data.set_shape([None, None, 3])
        return data

    def load(self, batch_size, noise_dims):
        noise_ds = tf.data.Dataset.from_tensors(0).repeat()
        noise_ds = noise_ds.map(lambda _: tf.random.normal([batch_size, noise_dims]))

        dataset = tf.data.TFRecordDataset(self.tfrecords)
        train_ds = dataset.take(self.train_set_size).map(self._tfrecord_parse,
                                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.resolution <= 5:
            train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(buffer_size=1000)
        train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1000)

        test_ds = dataset.skip(self.train_set_size).map(self._tfrecord_parse,
                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1000)

        return tf.data.Dataset.zip((train_ds, noise_ds)), tf.data.Dataset.zip((test_ds, noise_ds))


class MnistLoader(Loader):

    def __init__(self, res):
        self.res = res

    def _resize(self, images):
        factor = 32 // 2 ** self.res
        smaller = images.reshape((-1, 2 ** self.res, factor, 2 ** self.res, factor, 1)).max(4).max(2)
        return smaller

    def load(self, batch_size, noise_dims):
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
        train_images = np.pad(train_images, (2, 2), mode='edge')[2:-2]  # get to 32x32
        train_images = train_images.reshape(train_images.shape[0], 32, 32, 1).astype('float32')
        assert train_images.shape[1:] == (32, 32, 1), f'{train_images.shape}'
        train_images = self._resize(train_images)
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(10000).batch(batch_size, drop_remainder=True).prefetch(100)

        noise_dataset = tf.data.Dataset.from_tensors(0).repeat()
        noise_dataset = noise_dataset.map(lambda _: tf.random.normal([batch_size, noise_dims]))

        test_images = np.pad(test_images, (2, 2), mode='edge')[2:-2]  # get to 32x32
        test_images = test_images.reshape(test_images.shape[0], 32, 32, 1).astype('float32')
        assert test_images.shape[1:] == (32, 32, 1), f'{test_images.shape}'
        test_images = self._resize(test_images)
        test_images = (test_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

        test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(10000).batch(batch_size,drop_remainder=True).prefetch(100)

        noise_dataset_test = tf.data.Dataset.from_tensors(0).repeat()
        noise_dataset_test = noise_dataset_test.map(lambda _: tf.random.normal([batch_size, noise_dims]))

        return tf.data.Dataset.zip((train_dataset, noise_dataset)), tf.data.Dataset.zip((test_dataset, noise_dataset_test))

import tensorflow as tf

class SpectralNormalization(tf.keras.layers.Wrapper):

    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.u = self.add_weight(shape=(self.w_shape[0], 1),
                                   initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                                   trainable=False,
                                   name='sn_u')
        super().build()

    @tf.function
    def call(self, inputs, training=None):
        if training:
            w = tf.reshape(self.w, (tf.shape(self.w)[0], -1))

            # only one power iteration
            u = tf.linalg.l2_normalize(self.u)
            v = tf.linalg.l2_normalize(tf.matmul(tf.transpose(w), u))
            u = tf.linalg.l2_normalize(tf.matmul(w, v))

            sigma = tf.matmul(tf.matmul(tf.transpose(u), w), v)
            w = tf.divide(w, sigma)

            self.layer.kernel.assign(tf.reshape(w, tf.shape(self.layer.kernel)))
            self.u.assign(u)

        return self.layer(inputs)

import tensorflow as tf

_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss_non_saturating(disc_output_for_gen):
    return _cross_entropy(tf.ones_like(disc_output_for_gen), disc_output_for_gen)


def discriminator_loss_non_saturating(disc_output_for_real, disc_output_for_gen):
    real_loss = _cross_entropy(tf.ones_like(disc_output_for_real), disc_output_for_real)
    gen_loss = _cross_entropy(tf.zeros_like(disc_output_for_gen), disc_output_for_gen)
    return real_loss + gen_loss


def r1penalty(r1_gamma, discriminator_real_gradients):
    return r1_gamma / 2 * tf.reduce_sum(tf.square(discriminator_real_gradients), axis=[1, 2, 3], keepdims=True)


def generator_wgan_loss(fake_scores):
    return -fake_scores

def discriminator_wgan_loss(real_scores, fake_scores):
    return fake_scores - real_scores

def grad_penalty(real_scores, fake_scores, discriminator, real_images, fake_images,
                 wgan_lambda=10.0, wgan_target=1.0, wgan_eps=0.001):
    minibatch_size = real_images.shape[0]

    # mix images
    alphas = tf.random.uniform([minibatch_size, 1, 1, 1], 0.0, 1.0)
    mixed_images = real_images + (fake_images - real_images) * alphas
    with tf.GradientTape() as disc_tape:
        disc_tape.watch(mixed_images)
        mixed_scores = discriminator(mixed_images, training=True)
        mixed_loss = tf.reduce_sum(mixed_scores)
    mixed_grads = disc_tape.gradient(mixed_loss, [mixed_images])[0]
    mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1, 2, 3]))
    gradient_penalty = tf.reshape(tf.square(mixed_norms - wgan_target), [-1, 1])
    loss = wgan_lambda / (wgan_target ** 2) * gradient_penalty
    # epsilon penalty
    e_penalty = tf.square(real_scores)
    loss += e_penalty * wgan_eps

    # maybe fix up disc loss

    return loss

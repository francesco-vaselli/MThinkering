import pandas as pd
import tensorflow as tf
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 
from scipy.stats import norm, normaltest, wasserstein_distance, beta
from tensorflow.keras import layers
from tensorflow.python.framework import dtypes
import time
plt.switch_backend('agg')


z_dim = 46
x_dim = 46
batch_size = 1024
x_samples = int(1.024e6)
steps = 1000
bures_weight = 1.

default_adam_beta1 = 0.5
default_adam_learning_rate = 1e-5
default_z_sampler = z_sampler = tf.random.normal
AUTOTUNE = tf.data.AUTOTUNE


def fully_connected_generator_model(z_dim, x_dim, n_hidden=128, n_layers=2, activation='tanh'):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(z_dim,)))
    for _ in range(n_layers):
        model.add(layers.Dense(n_hidden, activation=activation))
    model.add(layers.Dense(x_dim))
    return model


def fully_connected_discriminator_model(x_dim, n_hidden=128, n_layers=2, activation='tanh', output_activation=None,
                                        return_feature_map=False):
    inp = layers.Input(shape=(x_dim,))
    dense = layers.Dense(n_hidden, activation)(inp)
    for _ in range(n_layers - 1):
        dense = layers.Dense(n_hidden, activation=activation)(dense)
    out = layers.Dense(1, activation=output_activation)(dense)
    dis_model = tf.keras.Model(inputs=inp, outputs=out)
    if return_feature_map:
        feature_map_model = tf.keras.Model(inputs=inp, outputs=dense)
        return dis_model, feature_map_model
    else:
        return dis_model


cross_entropy_from_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def discriminator_loss(real_output, fake_output, logits=True):
    if logits:
        real_loss = cross_entropy_from_logits(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy_from_logits(tf.zeros_like(fake_output), fake_output)
    else:
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss) / 2
    return total_loss


def generator_loss(fake_output, logits=True):
    if logits:
        loss = cross_entropy_from_logits(tf.ones_like(fake_output), fake_output)
    else:
        loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss


# Matrix square root using the Newton-Schulz method
def sqrt_newton_schulz(A, iterations=15, dtype='float64'):
    dim = A.shape[0]
    normA = tf.norm(A)
    Y = tf.divide(A, normA)
    I = tf.eye(dim, dtype=dtype)
    Z = tf.eye(dim, dtype=dtype)
    for i in range(iterations):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    sqrtA = Y * tf.sqrt(normA)
    return sqrtA


# Matrix square root using the eigenvalue decompostion
def matrix_sqrt_eigen(mat):
    eig_val, eig_vec = tf.linalg.eigh(mat)
    diagonal = tf.linalg.diag(tf.pow(eig_val, 0.5))
    mat_sqrt = tf.matmul(diagonal, tf.transpose(eig_vec))
    mat_sqrt = tf.matmul(eig_vec, mat_sqrt)
    return mat_sqrt


def wasserstein_bures_kernel(fake_phi, real_phi, sqrtm_func=sqrt_newton_schulz, epsilon=10e-14, normalize=True , dtype='float64',weight=bures_weight,method='NewtonSchultz'):
    if dtype == 'float64':
        fake_phi = tf.cast(fake_phi, dtypes.float64)
        real_phi = tf.cast(real_phi, dtypes.float64)

    batch_size = fake_phi.shape[0]

    # Center and normalize
    fake_phi = fake_phi - tf.ones(shape=(batch_size, 1), dtype=dtype) @ tf.math.reduce_mean(fake_phi, axis=0, keepdims=True)
    real_phi = real_phi - tf.ones(shape=(batch_size, 1), dtype=dtype) @ tf.math.reduce_mean(real_phi, axis=0, keepdims=True)
    if normalize:
        fake_phi = tf.nn.l2_normalize(fake_phi, 1)
        real_phi = tf.nn.l2_normalize(real_phi,1)

    K11 = fake_phi @ tf.transpose(fake_phi)
    K11 = K11 + epsilon * tf.eye(batch_size, dtype=dtype)
    K22 = real_phi @ tf.transpose(real_phi)
    K22 = K22 + epsilon * tf.eye(batch_size, dtype=dtype)

    K12 = fake_phi @ tf.transpose(real_phi) + epsilon * tf.eye(batch_size, dtype=dtype)

    if method == 'NewtonSchultz':
        bures = tf.linalg.trace(K11) + tf.linalg.trace(K22) - 2 * tf.linalg.trace(sqrtm_func(K12 @ tf.transpose(K12)))
    else:
        bures = tf.linalg.trace(K11) + tf.linalg.trace(K22) - 2 * tf.linalg.trace(matrix_sqrt_eigen(K12 @ tf.transpose(K12)))

    return weight * bures


def wasserstein_bures_covariance(fake_phi, real_phi, epsilon=10e-14, sqrtm_func=sqrt_newton_schulz, normalize=True, dtype='float64',weight=bures_weight,method='NewtonSchultz',adaptive_weight = 'fixed'): 
    if dtype == 'float64':
        fake_phi = tf.cast(fake_phi, tf.float64)
        real_phi = tf.cast(real_phi, tf.float64)

    batch_size = fake_phi.shape[0]
    h_dim = fake_phi.shape[1]

    # Center and normalize
    fake_phi = fake_phi - tf.ones(shape=(batch_size, 1), dtype=dtype) @ tf.math.reduce_mean(fake_phi, axis=0, keepdims=True)
    real_phi = real_phi - tf.ones(shape=(batch_size, 1), dtype=dtype) @ tf.math.reduce_mean(real_phi, axis=0, keepdims=True)
    if normalize:
        fake_phi = tf.nn.l2_normalize(fake_phi, 1)
        real_phi = tf.nn.l2_normalize(real_phi,1)
    
    # bures
    C1 = tf.transpose(fake_phi) @ fake_phi
    C1 = C1 + epsilon * tf.eye(h_dim, dtype=dtype)
    C2 = tf.transpose(real_phi) @ real_phi
    C2 = C2 + epsilon * tf.eye(h_dim, dtype=dtype)
    
    if method == 'NewtonSchultz':
        bures = tf.linalg.trace(C1) + tf.linalg.trace(C2) - 2 * tf.linalg.trace(sqrtm_func(C1 @ C2))
    else:
        bures = tf.linalg.trace(C1) + tf.linalg.trace(C2) - 2 * tf.linalg.trace(matrix_sqrt_eigen(C1 @ C2))

    return weight * bures


@tf.function
def train_step(x_batches):
    noise = z_sampler([batch_size, z_dim], dtype=tf.dtypes.float64)
    real_batch = x_batches
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_batch = generator(tf.concat([real_batch[:, 0:6], noise], 1), training=True)

        real_y = discriminator(real_batch[:, 6:52], training=True)
        fake_y = discriminator(generated_batch, training=True)

        phi_real = discriminator_feature_map(real_batch[:, 6:52], training=True)
        phi_fake = discriminator_feature_map(generated_batch, training=True)

        diversity_loss = diversity_loss_func(phi_fake, phi_real)
        gen_cross_entropy = generator_loss(fake_y)
        gen_cross_entropy = tf.cast(gen_cross_entropy, tf.float64)
        gen_loss = 0.5 * (gen_cross_entropy + diversity_loss)
        disc_loss = discriminator_loss(real_y, fake_y)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_cross_entropy, disc_loss, diversity_loss


def train():
  for epoch in range(steps):
    start = time.time()

    for batch in tfd:
      train_step(batch)
      print('done one batch')

    if epoch%50 == 0:
      generator.save_weights(f'./checkpoints/my_gcheckpoint_{epoch}')
      discriminator.save_weights(f'./checkpoints/my_dcheckpoint_{epoch}')
      print("saved checkpoints")


    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


if __name__=='__main__':

    generator = fully_connected_generator_model(6+z_dim, x_dim, n_hidden=512, n_layers=10, activation='relu')
    discriminator, discriminator_feature_map = fully_connected_discriminator_model(x_dim, n_hidden=512, n_layers=10, activation='relu',
											 return_feature_map=True)

    # choice of bures loss
    diversity_loss_func = wasserstein_bures_kernel

    generator_optimizer = tf.keras.optimizers.Adam(beta_1=default_adam_beta1,
                                                            learning_rate=default_adam_learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(beta_1=default_adam_beta1,
                                                                learning_rate=default_adam_learning_rate)

    df = pd.read_hdf("../jetData/MjetData.h5", stop=1.024e6+18)
    dff = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    tfd = tf.data.Dataset.from_tensor_slices(dff).cache().shuffle(int(1.024e6)).batch(batch_size).prefetch(AUTOTUNE)

    train()

    discriminator.save('./saves/disc')
    generator.save('./saves/gen')

    df_test = pd.read_hdf("/content/drive/MyDrive/thesis_data/MjetData.h5", start=1.024e6+19, stop=1.05e6)
    dff_test = df_test[~df_test.isin([np.nan, np.inf, -np.inf]).any(1)]
    dff_test_gen = dff_test.iloc[:, 0:6]
    dff_test_reco = dff_test.iloc[:, 6:52]

    gen_input = tf.convert_to_tensor(dff_test_gen)
    noise = z_sampler([len(gen_input), z_dim], dtype=tf.dtypes.float64)
    final_gen_input = tf.concat([gen_input, noise], 1)
    generated_sample = generator(final_gen_input, training=False).numpy()

    for i in range(0, generated_sample.shape[1]):
        ws = wasserstein_distance(dff_test_reco.iloc[:, i], generated_sample[:, i].flatten())
        plt.figure()
        plt.hist(dff_test_reco.iloc[:, i], bins=100, alpha=0.6)
        plt.hist(generated_sample[:, i].flatten(), bins=100, label=f'ws = {ws}')
        plt.title(f"Comparison of {list(dff_test_reco)[i]}")
        plt.legend()
        plt.savefig(f"./figures/{list(dff_test_reco)[i]}.png")

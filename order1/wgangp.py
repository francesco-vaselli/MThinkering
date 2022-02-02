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
# weight for gp and number of advantage steps for discriminator
lam = 1e-3
n_critic = 5

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


def gradient_penalty(x, x_gen):
        epsilon = tf.random.uniform([x.shape[0]] + [1] * (len(x.shape) - 1), 0.0,
                                    1.0)  # gives [batch_size, 1...,1 ] with a 1 for each dimension other than the batch size
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = discriminator(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=list(range(1, len(x.shape)))))  # L2
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)  # expectation
        return d_regularizer


def update_discriminator(noise, real_batch):
        generated_batch = generator(noise, training=True)
        with tf.GradientTape() as disc_tape:
            real_y = discriminator(real_batch, training=True)
            fake_y = discriminator(generated_batch, training=True)
            d_regularizer = gradient_penalty(real_batch, generated_batch)
            disc_loss = tf.reduce_mean(real_y) - tf.reduce_mean(fake_y) + lam * d_regularizer
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))
        return disc_loss


def update_generator(noise):
        with tf.GradientTape() as gen_tape:
            generated_batch = generator(noise, training=True)
            fake_y = discriminator(generated_batch, training=True)
            gen_loss = tf.reduce_mean(fake_y)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        return gen_loss


@tf.function
def train_step(x_batches):
    disc_loss = 0.0
    for i in range(n_critic):
        noise = z_sampler([batch_size, z_dim], dtype=tf.dtypes.float32)
        noise = tf.concat([x_batches[i][:, 0:6], noise], 1)
        real_batch = x_batches[i][:, 6:52]
        disc_loss = update_discriminator(noise, real_batch)

    noise = z_sampler([batch_size, z_dim])
    noise = tf.concat([x_batches[i][:, 0:6], noise], 1)
    gen_loss = update_generator(noise)
    return gen_loss, disc_loss


def train():
  for epoch in range(steps):
    start = time.time()

    batch_list = []
    for batch in tfd:
      batch_list.append(batch)
      if len(batch_list)%n_critic == 0:
        train_step(batch_list)
        print(f'Epoch {epoch}: done one set of {n_critic} batches')
        batch_list.clear()
        
    if epoch%20 == 0:
      generator.save_weights(f'./checkpoints/my_gcheckpoint_{epoch}')
      discriminator.save_weights(f'./checkpoints/my_dcheckpoint_{epoch}')
      print("saved checkpoints")

    print (f'Time for epoch {epoch} is {time.time()-start} sec')


if __name__=='__main__':

    # define models and optimizers
    generator = fully_connected_generator_model(6+z_dim, x_dim, n_hidden=512, n_layers=10, activation='relu')
    discriminator = fully_connected_discriminator_model(x_dim, n_hidden=512, n_layers=10, activation='relu')

    
    generator_optimizer = tf.keras.optimizers.Adam(beta_1=default_adam_beta1,
                                                            learning_rate=default_adam_learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(beta_1=default_adam_beta1,
                                                                learning_rate=default_adam_learning_rate)

    # import data and prepare dataset
    df = pd.read_hdf("../jetData/MjetData.h5", stop=1.024e6+18)
    dff = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].astype('float32')
    tfd = tf.data.Dataset.from_tensor_slices(dff).cache().shuffle(int(1.024e6)).batch(batch_size).prefetch(AUTOTUNE)

    # train and save
    train()

    discriminator.save('./saves/disc')
    generator.save('./saves/gen')

    # simple testing on new samples
    df_test = pd.read_hdf("/content/drive/MyDrive/thesis_data/MjetData.h5", start=1.024e6+19, stop=1.05e6)
    dff_test = df_test[~df_test.isin([np.nan, np.inf, -np.inf]).any(1)].astype('float32')
    dff_test_gen = dff_test.iloc[:, 0:6]
    dff_test_reco = dff_test.iloc[:, 6:52]

    gen_input = tf.convert_to_tensor(dff_test_gen)
    noise = z_sampler([len(gen_input), z_dim], dtype=tf.dtypes.float32)
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
from tensorflow.keras import layers
import glob
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import multiprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Generator(keras.Model):

  def __init__(self):
    super(Generator, self).__init__()

    # z: [b, 100] => [b, 3*3*512] => [b, 3, 3, 512] => [b, 64, 64, 3]
    self.fc = layers.Dense(6 * 6)

    self.dense1 = layers.Dense(12, name="dense1")
    self.bn1 = layers.LayerNormalization()

    self.dense2 = layers.Dense(12, name="dense2")
    self.bn2 = layers.LayerNormalization()

    self.dense3 = layers.Dense(6, name="output")

  def call(self, inputs, training=None):
    x = self.fc(inputs)
    x = tf.nn.leaky_relu(x)
    x = tf.nn.leaky_relu(self.bn1(self.dense1(x), training=training))
    x = tf.nn.leaky_relu(self.bn2(self.dense2(x), training=training))
    x = self.dense3(x)
    x = tf.tanh(x)
    return x


class Discriminator(keras.Model):

  def __init__(self):
    super(Discriminator, self).__init__()

    # [b, 64, 64, 3] => [b, 1]
    self.dense1 = layers.Dense(6, name="dense1")

    self.dense2 = layers.Dense(3, name="dense2")
    self.bn2 = layers.LayerNormalization()

    self.dense3 = layers.Dense(3, name="dense3")
    self.bn3 = layers.LayerNormalization()

    self.fc = layers.Dense(1, name="output")

  def call(self, inputs, training=None):

    x = tf.nn.leaky_relu(self.dense1(inputs))
    x = tf.nn.leaky_relu(self.bn2(self.dense2(x), training=training))
    x = tf.nn.leaky_relu(self.bn3(self.dense3(x), training=training))
    # [b, -1] => [b, 1]
    logits = self.fc(x)

    return logits


def celoss_ones(logits):
  # [b, 1]
  # [b] = [1, 1, 1, 1,]
  # loss = tf.keras.losses.categorical_crossentropy(y_pred=logits,
  #                                                y_true=tf.ones_like(logits))
  return - tf.reduce_mean(logits)


def celoss_zeros(logits):
  # [b, 1]
  # [b] = [1, 1, 1, 1,]
  # loss = tf.keras.losses.categorical_crossentropy(y_pred=logits,
  #                                                y_true=tf.zeros_like(logits))
  return tf.reduce_mean(logits)


def gradient_penalty(discriminator, batch_x, fake_image):

  batchsz = batch_x.shape[0]

  # [b, h, w, c]
  t = tf.random.uniform([batchsz, 6])
  # [b, 1, 1, 1] => [b, h, w, c]
  t = tf.broadcast_to(t, batch_x.shape)

  interplate = t * batch_x + (1 - t) * fake_image

  with tf.GradientTape() as tape:
    tape.watch([interplate])
    d_interplote_logits = discriminator(interplate, training=True)
  grads = tape.gradient(d_interplote_logits, interplate)

  # grads:[b, h, w, c] => [b, -1]
  grads = tf.reshape(grads, [grads.shape[0], -1])
  gp = tf.norm(grads, axis=1)  # [b]
  gp = tf.reduce_mean((gp - 1)**2)

  return gp


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
  # 1. treat real image as real
  # 2. treat generated image as fake
  fake_image = generator(batch_z, is_training)
  d_fake_logits = discriminator(fake_image, is_training)
  d_real_logits = discriminator(batch_x, is_training)

  d_loss_real = celoss_ones(d_real_logits)
  d_loss_fake = celoss_zeros(d_fake_logits)
  gp = gradient_penalty(discriminator, batch_x, fake_image)

  loss = d_loss_real + d_loss_fake + 10. * gp

  return loss, gp


def g_loss_fn(generator, discriminator, batch_z, is_training):

  fake_image = generator(batch_z, is_training)
  d_fake_logits = discriminator(fake_image, is_training)
  loss = celoss_ones(d_fake_logits)

  return loss


def get_data_in_category(datas, category, y_column_name):
  groupby_data = datas.groupby(y_column_name)
  category_data = groupby_data.get_group(category)
  return category_data


def main():
  # hyper parameters
  z_dim = 36
  epochs = 300
  batch_size = 35
  learning_rate = 0.0005
  is_training = True
  random_state = 1

  ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
  file_path = os.path.join(
    ROOT_DIR, 'dataset', "seeds_revised.csv")
  dataset = pd.read_csv(file_path)
  categories = dataset[dataset.columns[-1]].unique()
  category_target_name = dataset.columns[-1]
  for category in categories:
    category_train_data = get_data_in_category(
      dataset, category, category_target_name)
    # train_test_split
    x = category_train_data.drop(category_train_data.columns[-1], axis=1)
    y = category_train_data[category_train_data.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.5, shuffle=False,
                                                        random_state=random_state + category)
    scaler = StandardScaler().fit(X_train.values)
    X_scaled = scaler.transform(X_train.values)

    # KFold
    # kf = KFold(n_splits=2)
    # for X_train, X_test in kf.split(category_train_data.values):
    #   print(X_train, "\n", X_test)

  # tf.random.set_seed(233)
  # np.random.seed(233)
    assert tf.__version__.startswith('2.')

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 6))
    z_sample = tf.random.normal([100, z_dim])

    g_optimizer = tf.keras.optimizers.Adam(
      learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(
      learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):

      for _ in range(5):
        batch_z = tf.random.normal([batch_size, z_dim])
        batch_x = X_scaled

        # train D
        with tf.GradientTape() as tape:
          d_loss, gp = d_loss_fn(generator, discriminator,
                                 batch_z, batch_x, is_training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(
          zip(grads, discriminator.trainable_variables))

      batch_z = tf.random.normal([batch_size, z_dim])

      with tf.GradientTape() as tape:
        g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
      grads = tape.gradient(g_loss, generator.trainable_variables)
      g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

      print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss),
            'gp:', float(gp))

      if epoch + 1 == epochs:
        z = tf.random.normal([250, z_dim])
        fake_data = generator(z, training=False)
        fake_data = scaler.inverse_transform(fake_data)
        x = pd.DataFrame(fake_data, columns=[
                         'A', 'P', 'length of kernel', 'width of kernel', 'asymmetry coefficient', 'length of kernel groove'])
        y = pd.DataFrame(np.full(250, category, dtype=np.int),
                         columns=['Column8'])
        fake_data = x.merge(y, how='inner', left_index=True, right_index=True)
        fake_data.to_csv(os.path.join(
          ROOT_DIR, 'dataset', f'category{category}_dataset.csv'), index=False)


main()

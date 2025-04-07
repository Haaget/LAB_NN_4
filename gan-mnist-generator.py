import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Завантаження MNIST
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_train = np.expand_dims(X_train, axis=-1)  # (60000, 28, 28, 1)

BUFFER_SIZE = 10000
BATCH_SIZE = 128
LATENT_DIM = 100
EPOCHS = 500
SAVE_INTERVAL = 100

# Побудова генератора
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(LATENT_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))  # (7,7,256)

    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='sigmoid'))

    return model

# Побудова дискримінатора
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# Створення моделей
generator = build_generator()
discriminator = build_discriminator()

# Оптимізатори
cross_entropy = tf.keras.losses.BinaryCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Пайплайн даних
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Тренувальний цикл
def train_step(images, epoch):
    noise = tf.random.normal([tf.shape(images)[0], LATENT_DIM])  # Используем размер текущего батча

    # Тренування дискримінатора
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        real_label = tf.ones_like(real_output)  # Динамические метки
        fake_label = tf.zeros_like(fake_output)  # Динамические метки

        disc_loss = cross_entropy(real_label, real_output) + cross_entropy(fake_label, fake_output)

    grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_variables))

    # Тренування генератора
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = cross_entropy(real_label, fake_output)  # Динамическая метка для генератора

    grads_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(grads_gen, generator.trainable_variables))

    real_accuracy = tf.reduce_mean(tf.cast(real_output > 0.5, tf.float32))
    fake_accuracy = tf.reduce_mean(tf.cast(fake_output < 0.5, tf.float32))
    disc_accuracy = (real_accuracy + fake_accuracy) / 2

    # Рахуємо точність генератора (як кількість "реальних" згенерованих зображень)
    gen_accuracy = tf.reduce_mean(tf.cast(fake_output > 0.5, tf.float32))

    print(f"Epoch {epoch}, Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}, "
          f"Discriminator Accuracy: {disc_accuracy:.4f}, Generator Accuracy: {gen_accuracy:.4f}")
    epoch += 1

    return gen_loss, disc_loss, disc_accuracy, gen_accuracy, epoch

# Візуалізація
def generate_and_save_images(model, epoch):
    test_input = tf.random.normal([16, LATENT_DIM])
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'gan_mnist_epoch_{epoch}.png')
    plt.close()

# Навчання
for epoch in range(1, EPOCHS+1):
    for images in dataset:
        gen_loss, disc_loss, disc_accuracy, gen_accuracy, epoch = train_step(images, epoch)

    if epoch % SAVE_INTERVAL == 0 or epoch == 1:
        generate_and_save_images(generator, epoch)

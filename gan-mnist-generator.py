import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, BatchNormalization, Conv2D, Conv2DTranspose, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128 * 7 * 7, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(32, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(16, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(1, kernel_size=3, activation='tanh', padding='same'))
    return model

def build_discriminator(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    fake_image = generator(gan_input)
    gan_output = discriminator(fake_image)
    model = Model(gan_input, gan_output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

def train_gan(generator, discriminator, gan, epochs=500, batch_size=128):
    (X_train, _), _ = mnist.load_data()
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=-1)

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_images, real)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake)
        d_loss = [(d_loss_real[0] + d_loss_fake[0]) / 2, (d_loss_real[1] + d_loss_fake[1]) / 2]

        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real)
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f} | G Loss: {g_loss:.4f}")

def generate_and_save_images(generator, num_examples=10):
    noise = np.random.normal(0, 1, (num_examples, 100))
    gen_images = generator.predict(noise)
    gen_images = (gen_images + 1) / 2.0

    plt.figure(figsize=(10, 2))
    for i in range(num_examples):
        plt.subplot(1, num_examples, i+1)
        plt.imshow(gen_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
        
latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator((28, 28, 1))
gan = build_gan(generator, discriminator)
train_gan(generator, discriminator, gan)

generate_and_save_images(generator)
generator.save("generator_model.h5")
plt.show()

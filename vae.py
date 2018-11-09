"""Explore sequences of user actions using variational autoencoder."""

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt


def toy_example_mnist(type="regularized"):
    """
    Run a simple toy examlpe from https://blog.keras.io/building-autoencoders-in-keras.html
    :return:
    """
    from keras.datasets import mnist
    # this is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    # this is our input placeholder
    input_img = Input(shape=(784,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(784, activation='sigmoid')(encoded)
    # create the training and testing data
    (x_train, _), (x_test, _) = mnist.load_data()
    # We will normalize all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784.
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)
    if type == "standard": # from section "Let's build the simplest possible autoencoder"
        # this model maps an input to its reconstruction
        autoencoder = Model(input_img, decoded)
        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))
        # configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        # train autoencoder for 50 epochs
        autoencoder.fit(x_train, x_train,
                        epochs=50,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test))
    elif type == "regularized": # from section "Adding a sparsity constraint on the encoded representations"
        # train a regularized model for 100 epochs; longer because it is less likely to overfit
        encoding_dim = 32
        input_img = Input(shape=(784,))
        # add a Dense layer with a L1 activity regularizer
        encoded = Dense(encoding_dim, activation='relu',
                        activity_regularizer=regularizers.l1(10e-5))(input_img)
        decoded = Dense(784, activation='sigmoid')(encoded)
        autoencoder = Model(input_img, decoded)
        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(x_train, x_train,
                        epochs=100, #todo increase back to 100 after testing
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test))

    # try to visualize the reconstructed inputs and the encoded representations
    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    # use Matplotlib
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.title("{} VAE".format(type))
    plt.show()
    return

def toy_example_lstm():
    return



def main():
    toy_example_mnist()
    return


if __name__ == "__main__":
    main()

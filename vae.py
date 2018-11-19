"""Explore sequences of user actions using variational autoencoder."""

from keras.layers import Dense
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.foldit import get_pid_uid_sequences
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from preprocessing import preprocess_lstm_data
from keras.callbacks import TensorBoard
# from ggplot import *

MAX_LEN = 5000


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
    if type == "standard":  # from section "Let's build the simplest possible autoencoder"
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
    elif type == "regularized":  # from section "Adding a sparsity constraint on the encoded representations"
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
                        epochs=100,  # todo increase back to 100 after testing
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


def lstm_vae(X, Y, max_len=MAX_LEN, latent_dim=10, batch_size=50, n_epochs=20,
             reg=regularizers.l1_l2(l1=0.01, l2=0.01)):
    input_dim = X.shape[2]  # number of observations per timestep
    # create x_train and x_test from array
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=43354)
    # create lstm
    inputs = Input(shape=(max_len, input_dim))
    encoded = LSTM(latent_dim, kernel_regularizer=reg, dropout=0.2, recurrent_dropout=0.2)(inputs)
    decoded = RepeatVector(max_len)(encoded)
    decoded = LSTM(input_dim, return_sequences=True, kernel_regularizer=reg, dropout=0.2, recurrent_dropout=0.2)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    sequence_autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy') #todo: loss function might change
    sequence_autoencoder.fit(X_train, X_train,
                             epochs=n_epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(X_test, X_test),
                             callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    # get latent encoding for test observations
    x_test_encoded = encoder.predict(X_test, batch_size=batch_size)
    # use tsne to project results into lower dimension
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(x_test_encoded)
    # plot the results
    plt.figure(figsize=(6, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=Y[:, 0])  # Y[:,0] is pid; Y[:,1] is pid
    plt.colorbar()
    plt.show()
    return


def main():
    # toy_example_mnist()
    data = get_pid_uid_sequences("./data/foldit_user_events_2003433_2003465.csv", max_len=MAX_LEN)
    X, Y = preprocess_lstm_data(data, MAX_LEN)
    lstm_vae(X, Y)
    return


if __name__ == "__main__":
    main()

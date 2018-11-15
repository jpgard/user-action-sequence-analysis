import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def preprocess_lstm_data(input_dict, max_len):
    """
    Reshape input_dict to work as input to a keras LSTM.
    :param input_dict: heirarchical dict of [pid][uid] keys with sequences for that uid/pid as values.
    :return: np.array for input to keras Input layer
    """
    # get pid,uid,sequences; pull from dict so ordering is nonrandom
    pid_uid_seqs = [(pid, uid, input_dict[pid][uid]) for pid in input_dict.keys() for uid in input_dict[pid].keys()]
    n = len(pid_uid_seqs)
    X = pad_sequences([row[2] for row in pid_uid_seqs], maxlen=max_len, padding="post", truncating="post", value=-1)
    import ipdb;ipdb.set_trace()
    X = to_categorical(X)
    # X = np.reshape(X, (n, max_len, 1)) # values for tuple are (num_obs, timeseries_length, observations_per_timestep/num_words_in_vocab); see https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    # create Y, an array of (pid, uid) for each row in X
    Y = np.array([(row[0], row[1]) for row in pid_uid_seqs]) #todo: these should be one-hot encoded vectors of length num_events, not sequences of continuous values
    assert Y.shape[0] == X.shape[0]
    return X, Y

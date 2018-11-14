import numpy as np
from keras.preprocessing.sequence import pad_sequences

def preprocess_lstm_data(input_dict, max_len):
    """
    Reshape input_dict to work as input to a keras LSTM.
    :param input_dict: heirarchical dict of [pid][uid] keys with sequences for that uid/pid as values.
    :return: np.array for input to keras Input layer
    """
    # get pid,uid,sequences; pull from dict so ordering is nonrandom
    pid_uid_seqs = [(pid, uid, input_dict[pid][uid]) for pid in input_dict.keys() for uid in input_dict[pid].keys()]
    n = len(pid_uid_seqs)
    X = pad_sequences([row[2] for row in pid_uid_seqs], maxlen=max_len, padding="post", truncating="post", value=-99)
    X = np.reshape(X, (n, max_len, 1)) # values for tuple are (num_obs, timeseries_length, observations_per_timestep); see https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    return X

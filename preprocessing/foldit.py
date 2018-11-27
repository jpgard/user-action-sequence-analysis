"""Functions to preprocess foldit data into user-puzzle sequences"""

import pandas as pd
from collections import defaultdict


def get_pid_uid_sequences(infile, max_len=5000):
    """
    Fetch a hierarchical dictionary with keys [pid][uid] and values which are sequences of actions
    :param infile:
    :param max_len: maximum length of sequence to store
    :return: hierarchical dictionary with keys [pid][uid] and values which are np.ndarray containing sequences of actions
    """
    print("[INFO] fetching data from {}...".format(infile))
    pid_col = "puzzle_id"
    uid_col = "user_id"
    ts_col = "timestamp"
    tool_col = "tool"
    tool_code_col= "tool_code" #column to generate; numeric identifiers for tools
    empty_event_value=-999
    df = pd.read_csv(infile)
    df[tool_col] = pd.Categorical(df[tool_col])
    df[tool_code_col] = df[tool_col].cat.codes
    pid_uid_seqs = defaultdict(dict)
    uid_pid_values = df[[uid_col, pid_col]].drop_duplicates()
    print("[INFO] \n\tunique user count: {}\n\tunique pid count: {}".format(len(uid_pid_values[uid_col].unique()),
                                                                            len(uid_pid_values[pid_col].unique())))
    user_sequence_lens = df.groupby([uid_col, pid_col]).size().quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print("[INFO] printing quantiles for sequence lengths:")
    print(user_sequence_lens)
    for ix,uid,pid in uid_pid_values.itertuples():
        pid_uid_seqs[pid][uid] = df.loc[(df[uid_col] == uid) & (df[pid_col] == pid), tool_code_col].values[0:max_len]
    print("[INFO] complete")
    return pid_uid_seqs

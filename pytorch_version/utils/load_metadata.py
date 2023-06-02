import pandas as pd


def set_tr_val_samples_labels(meta_filename, val_size):
    column_names = ["filename", "class_label"]
    data_frame = pd.read_csv(meta_filename, header=None, names=column_names, sep="\s+")
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    split_idx = int(len(data_frame) * (1 - val_size))
    return (
        data_frame["filename"][:split_idx].tolist(),
        data_frame["class_label"][:split_idx].tolist(),
        data_frame["filename"][split_idx:].tolist(),
        data_frame["class_label"][split_idx:].tolist(),
    )


def set_test_samples_labels(meta_filename):
    sample_paths, class_labels = [], []
    column_names = ["filename", "class_label"]
    data_frame = pd.read_csv(meta_filename, header=None, names=column_names, sep="\s+")
    return data_frame["filename"].tolist(), data_frame["class_label"].tolist()

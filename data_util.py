import numpy as np
import pandas as pd
from locus_sampling.cross_validation import BalancedGroupKFold
# from locus_sampling.cross_validation import FixedReplicatedKFold
from tensorflow import convert_to_tensor, float32

# TRACK_PATH = "Track_Data/Track_osu18_SNP_2kbp_window_chr_all.npy"
# BED_PATH = "SNP_Data/osu18_SNP_2kbp_window_chr_all.bed"
# GROUP_PATH = "SNP_Data/osu18_groups.tsv"

CV_RANDOM_SEED = 88

def load_track_matrix(track_path):
    # Instant loading! Cost ~5G RAM
    X = np.load(track_path)  # shape is (38784, 2001, 9)
    
    new_shape = (38784, 9, 2001)
    X = np.reshape(X, new_shape)

    print("[data_util] Tracks loaded; matrix shape is {}".format(X.shape))

    return X

def load_SNP_info(bed_path, group_path):
    bed = pd.read_csv(bed_path, sep="\t", names=['chr', 'start', 'end', 'name', 'label'], header=None)
    print("[data_util] BED file loaded; shape is {}".format(bed.shape))

    g = pd.read_csv("SNP_Data/osu18_groups.tsv", sep="\t")
    print("[data_util] Group file loaded; shape is {}".format(g.shape))

    snp_info = bed.merge(g.loc[:, "name":], on="name")
    print("[data_util] SNP info merged; shape is {}".format(snp_info.shape))

    return snp_info

def load_data(track_path, bed_path, group_path):
    X = load_track_matrix(track_path)

    snp_info = load_SNP_info(bed_path, group_path)
    Y = snp_info["label"].values
    groups = snp_info["group_id"].values

    return X, Y, groups

def calc_class_weight(y):
    pos = sum(y)
    total = len(y)
    neg = total - pos
    
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1.0 / neg)*(total)/2.0 
    weight_for_1 = (1.0 / pos)*(total)/2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    return class_weight

def calc_sample_weight(y):
    class_weight = calc_class_weight(y)
    sample_weight = np.array([class_weight[label] for label in y])

    return sample_weight

def split_train_test(X, Y, groups, n_splits, which=0, x_as_tensor=True):
    cv = BalancedGroupKFold(n_splits=n_splits, random_state=CV_RANDOM_SEED)
    splits = cv.split(X, Y, groups)

    for _ in range(0, which):
        next(splits)  # skip the preceding (which-1) splits

    # get exactly split specified by `which`
    train_index, test_index = next(splits)

    if type(X) == pd.DataFrame:
        x_train, x_test = X.loc[train_index], X.loc[test_index]
        if x_as_tensor: 
            print("[data_util] X is a Pandas DataFrame; ignore parameter x_as_tensor=True")
    else:  # type(X) == np.ndarray
        x_train, x_test = X[train_index], X[test_index]
        if x_as_tensor:
            x_train = convert_to_tensor(x_train[:, :, :, np.newaxis], dtype=float32) 
            x_test = convert_to_tensor(x_test[:, :, :, np.newaxis], dtype=float32) 

    y_train, y_test = Y[train_index], Y[test_index]

    return x_train, y_train, x_test, y_test

def split_train_val_test(X, Y, groups, n_splits, which_val=0, which_test=0, x_as_tensor=True):
    # Step 1: Split the whole dataset into two parts: Temp vs Test
    cv = BalancedGroupKFold(n_splits=n_splits, random_state=CV_RANDOM_SEED)
    splits = cv.split(X, Y, groups)

    for _ in range(0, which_test):
        next(splits)  # skip the preceding (which-1) splits

    # get exactly split specified by `which`
    temp_index, test_index = next(splits)
    if type(X) == pd.DataFrame:
        x_temp, x_test = X.loc[temp_index], X.loc[test_index]
    else:  # type(X) == np.ndarray
        x_temp, x_test = X[temp_index], X[test_index]
    y_temp, y_test = Y[temp_index], Y[test_index]
    groups_temp = groups[temp_index]

    # Step 2: Split the Temp dataset into two parts: Train/Val
    cv = BalancedGroupKFold(n_splits=n_splits-1, random_state=CV_RANDOM_SEED)
    splits = cv.split(x_temp, y_temp, groups_temp)

    for _ in range(0, which_val):
        next(splits)  # skip the preceding (which-1) splits

    # get exactly split specified by `which`
    train_index, val_index = next(splits)
    if type(X) == pd.DataFrame:
        x_train, x_val = x_temp.loc[train_index], x_temp.loc[val_index]
    else:  # type(X) == np.ndarray
        x_train, x_val = x_temp[train_index], x_temp[val_index]
    y_train, y_val = y_temp[train_index], y_temp[val_index]
    
    if type(X) != pd.DataFrame and x_as_tensor:
        x_train = convert_to_tensor(x_train[:, :, :, np.newaxis], dtype=float32) 
        x_test = convert_to_tensor(x_test[:, :, :, np.newaxis], dtype=float32) 
        x_val = convert_to_tensor(x_val[:, :, :, np.newaxis], dtype=float32)

    return x_train, y_train, x_val, y_val, x_test, y_test

def cv_split_train_test(X, Y, groups, n_splits):
    cv = BalancedGroupKFold(n_splits=n_splits, random_state=CV_RANDOM_SEED)

    for train_index, test_index in cv.split(X, Y, groups):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        x_train = convert_to_tensor(x_train[:, :, :, np.newaxis], dtype=float32) 
        x_test = convert_to_tensor(x_test[:, :, :, np.newaxis], dtype=float32)

        yield x_train, y_train, x_test, y_test
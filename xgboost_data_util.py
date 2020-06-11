import pandas as pd
import numpy as np
from data_util import load_SNP_info

# n2v_param_dist = dict(d=2, l=6, r=12, k=4, p=4, q=8, w=True)
# sgn_weight_dist = dict(w_4DGp=3.0, w_4DGt=0.3, w_GTEx=0.3, w_TFBS=0.1, w_NG=0.1, w_coexp=0.3, w_hn=0.3, w_bg=3.0)
# N2V_FEAT_PATH = "XGBoost_Data/Sgn_3.0_0.3_0.3_0.1_0.1_0.3_0.3_3.0_sum_2_6_12_4_4_8_True.tsv"
# BASE_FEAT_PATH = "XGBoost_Data/osu18_cerenkov_feat_mat_plus_group_size.tsv"

def load_xgboost_data(base_feat_path, n2v_feat_path, bed_path, group_path):
    snp_info = load_SNP_info(bed_path, group_path)

    n2v_feat = pd.read_csv(n2v_feat_path, sep="\t")
    # keep entries appearing in the track bed
    n2v_feat = n2v_feat.set_index("name").loc[snp_info["name"]].reset_index()
    n2v_feat = n2v_feat.drop(columns=["label", "INT_ID"])

    base_feat = pd.read_csv(base_feat_path, sep="\t")
    # keep entries appearing in the track bed
    base_feat = base_feat.set_index("name").loc[snp_info["name"]].reset_index()

    X = n2v_feat.merge(base_feat, on="name")
    X = X.drop(columns=["name", "label"])

    Y = snp_info["label"].values
    groups = snp_info["group_id"].values

    print("[xgboost_data_util] base+n2v feature loaded; shape is {}".format(X.shape))

    return X, Y, groups

def get_binary_columns(df):
    """
    util function for tf.feature_column when building BoostedTreesClassifier
    See https://medium.com/tensorflow/how-to-train-boosted-trees-models-in-tensorflow-ca8466a53127
    """
    bin_cols = [col for col in df 
                if np.isin(df[col].dropna().unique(), [0, 1]).all()]
    return bin_cols 

def compute_rare_event_rate(y):
    """
    In CERENKOV3 we init XGBClassifier(base_score=rare_event_rate)
    """
    return sum(y) / len(y)
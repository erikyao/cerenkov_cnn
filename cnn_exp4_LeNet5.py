import os
from data_util import load_data, split_train_test, split_train_val_test
from report_util import get_model_name, save_all_metrics_of_multi_models, save_single_metric_of_multi_models, save_history_copies
from cnn_util import CNN_Builder, compile_model, fit_model

# Data Paths
TRACK_PATH = "Track_Data/Track_osu18_SNP_2kbp_window_chr_all.npy"
BED_PATH = "SNP_Data/osu18_SNP_2kbp_window_chr_all.bed"
GROUP_PATH = "SNP_Data/osu18_groups.tsv"

# Model Fitting Parameters
BATCH_SIZE = 256
EPOCHS = 20
USE_CLASS_WEIGHT = True
USE_SAMPLE_WEIGHT = True
USE_REDUCE_RL = True
USE_EARLY_STOPPING = False
VERBOSE = 2

# CV Parameter
N_SPLITS = 5

"""
Fourth trial of LeNet5 inspired models
Same model configs with cnn_exp3_LeNet5 except doubling kernel numbers
"""
# See https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7
# LeNet-5 inspired models: ICMpCMpCMpCMpFDDDD
# Use wide conv and wide max pooling layers
model_configs = [("InputLayer:9x2001x1",
                    "Conv2D:16-1x{}-1x1-selu-valid".format(w ** 2),
                    "MaxPooling2D:1x4-1x4-valid",
                    "Conv2D:16-1x{}-1x1-selu-valid".format(w ** 2),
                    "MaxPooling2D:1x4-1x4-valid",
                    "Conv2D:32-1x{}-1x1-selu-valid".format(w ** 2),
                    "MaxPooling2D:1x4-1x4-valid",
                    "Conv2D:32-{}x{}-1x1-selu-valid".format(w, w),
                    "MaxPooling2D:2x2-2x2-valid",
                    "Flatten", 
                    "Dense:120-selu", 
                    "Dense:84-selu",
                    "Dense:10-selu",
                    "Dense:1-sigmoid") for w in [3, 5, 7]]

def main():
    # Step 1: load data
    X, Y, groups = load_data(track_path=TRACK_PATH, bed_path=BED_PATH, group_path=GROUP_PATH)
    x_train, y_train, x_test, y_test = split_train_test(X, Y, groups, N_SPLITS)
    
    # Step 2: run models
    models = [CNN_Builder.build_Sequential(model_config) for model_config in model_configs]
    for model in models:
        compile_model(model, metric_names=("auprc", "auroc"), optimizer="adam")
    histories = [fit_model(model, x_train, y_train, x_test, y_test, 
                           batch_size=BATCH_SIZE, epochs=EPOCHS, use_class_weight=USE_CLASS_WEIGHT, use_sample_weight=USE_SAMPLE_WEIGHT, 
                           use_reduce_rl=USE_REDUCE_RL, use_early_stopping=USE_EARLY_STOPPING, verbose=VERBOSE) for model in models]

    # Step 3: save artifacts
    script_fn = os.path.basename(__file__)
    folder = script_fn.split(".py")[0]
    model_names = [get_model_name(model_configs[i], i+1) for i in range(len(model_configs))]

    save_single_metric_of_multi_models(folder, model_names, histories, "val_auprc")
    save_all_metrics_of_multi_models(folder, model_names, histories)
    save_history_copies(folder, model_names, histories)

    print("{} finished!".format(script_fn))

if __name__ == "__main__":
    main()
import pandas as pd
import os
import pickle

LAYER_SHORT_NAME_MAP = {
    "InputLayer": "I",
    "Conv2D": "C", 
    "MaxPooling2D": "Mp",
    "AveragePooling2D": "Ap",
    "Flatten": "F",
    "Dense": "D",
    "Dropout": "Do",
    "BatchNormalization": "Bn"
}

def get_model_name(model_config, serial_num):
    layer_names = [layer_config.split(":")[0] for layer_config in model_config]
    layer_short_names = [LAYER_SHORT_NAME_MAP[layer_name] for layer_name in layer_names]
    model_name = "".join(layer_short_names) + "-" + str(serial_num)
    return model_name

def get_model_config_str(model_config):
    return ",".join(model_config)

def get_all_metrics(history, fill_early_stop=-1):
    num_epochs = history.params["epochs"]

    # in case of early stopping, fill metrics in un-executed epochs with `fill_early_stop` 
    for metric in history.history:
        history.history[metric] = history.history[metric] + [fill_early_stop] * (num_epochs - len(history.history[metric]))

    df = pd.DataFrame(history.history)
    df = df.assign(epoch=range(1, num_epochs + 1))

    return df

def save_all_metrics_of_multi_models(folder, model_names, histories, fill_early_stop=-1):
    if len(model_names) != len(histories):
        raise ValleError("number of model names and number of history objects do not match. Got {} model names and {} history objects".format(len(model_names), len(histories)))

    if not os.path.exists(folder):
        os.makedirs(folder)
        
    for i in range(len(model_names)):
        metrics_df = get_all_metrics(histories[i])
        metrics_fn = '{}_metrics.tsv'.format(model_names[i])
        
        metrics_df.to_csv(os.path.join(folder, metrics_fn), sep="\t", index=False)

def get_single_metric_of_multi_models(model_names, histories, metric_name, fill_early_stop=-1):
    if len(model_names) != len(histories):
        raise ValleError("number of model names and number of history objects do not match. Got {} model names and {} history objects".format(len(model_names), len(histories)))

    num_epochs = max(history.params["epochs"] for history in histories)

    for history in histories:
        history.history[metric_name] = history.history[metric_name] + [fill_early_stop] * (num_epochs - len(history.history[metric_name]))

    metric_dict = {}
    for i in range(len(model_names)):
        metric_dict[model_names[i]] = histories[i].history[metric_name]

    metric_df = pd.DataFrame(metric_dict)
    metric_df = metric_df.assign(metric=metric_name)
    metric_df = metric_df.assign(epoch=range(1, num_epochs + 1))

    return metric_df

def save_single_metric_of_multi_models(folder, model_names, histories, metric_name, fill_early_stop=-1):
    if not os.path.exists(folder):
        os.makedirs(folder)

    metric_fn = '{}.tsv'.format(metric_name)
    metric_df = get_single_metric_of_multi_models(model_names, histories, metric_name, fill_early_stop)
    metric_df.to_csv(os.path.join(folder, metric_fn), sep="\t", index=False)

class HistoryCopy:
    """
    An empty class for copies of histories, to support duck typing in `_make_history_copy`
    """
    pass

def _make_history_copy(history):
    history_copy = HistoryCopy()

    # history.model is a weak reference to the trained model object, which cannot be pickled
    # on the other hand, we are only interested in "history" and "params" fields
    history_copy.history = history.history
    history_copy.params = history.params
    
    return history_copy

def save_history_copies(folder, model_names, histories):
    history_copies = [_make_history_copy(history) for history in histories]

    for i in range(len(history_copies)):
        history_copies[i].model_name = model_names[i]

    with open(os.path.join(folder, "history_copies"), 'wb') as handle:
        pickle.dump(history_copies, handle)

def load_history_copies(folder):
    with open(os.path.join(folder, "history_copies"), "rb") as handle:
        history_copies = pickle.load(handle)

    return history_copies
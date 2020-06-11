import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy, AUC, FalsePositives, FalseNegatives
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, InputLayer, LeakyReLU
# from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import Model

from data_util import calc_class_weight, calc_sample_weight

TF_RANDOM_SEED = 486
tf.random.set_seed(TF_RANDOM_SEED)
print("[cnn_util] Tensorflow Random Seed set to {}".format(TF_RANDOM_SEED))

class CNN_Builder:
    DROP_OUT_RANDOM_SEED = 42

    @classmethod
    def build_Conv2D(cls, config):
        """
        e.g. config = "3-3x3-1x1-relu-valid" means 
            Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="valid")
        """
        param_list = config.split("-")
        
        if len(param_list) != 5:
            raise ValueError("Exact 5 parameters to config. Got config={}".format(config))
        
        # unpack the param list into specific variables
        filters, kernel_size, strides, activation, padding = param_list

        filters = int(filters)

        kernel_size = tuple(int(dim) for dim in kernel_size.split("x"))
        if len(kernel_size) == 1:
            kernel_size = kernel_size[0]

        strides = tuple(int(stride) for stride in strides.split("x"))
        if len(strides) == 1:
            strides = strides[0]

        if activation == "LeakyReLU":
            activation = lambda x: LeakyReLU()(x)

        return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding)

    @classmethod
    def build_Dense(cls, config):
        """
        e.g. config = "16-sigmoid" means Dense(units=16, activation='sigmoid')
        """
        param_list = config.split("-")
        
        if len(param_list) != 2:
            raise ValueError("Exact 2 parameters to config. Got config={}".format(config))
        
        # unpack the param list into specific variables
        units, activation = param_list

        units = int(units)

        if activation == "LeakyReLU":
            activation = lambda x: LeakyReLU()(x)

        if activation == "None":
            activation = None

        return Dense(units=units, activation=activation)

    @classmethod
    def build_Flatten(cls):
        return Flatten()

    @classmethod
    def build_BatchNormalization(cls):
        return BatchNormalization()

    @classmethod
    def _build_Pooling2D(cls, config, class_name):
        param_list = config.split("-")
        
        if len(param_list) != 3:
            raise ValueError("Exact 3 parameters to config. Got config={}".format(config))
        
        # unpack the param list into specific variables
        pool_size, strides, padding = param_list

        pool_size = tuple(int(dim) for dim in pool_size.split("x"))
        if len(pool_size) == 1:
            pool_size = pool_size[0]

        strides = tuple(int(stride) for stride in strides.split("x"))
        if len(strides) == 1:
            strides = strides[0]

        if class_name == "MaxPooling2D":
            return MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
        elif class_name == "AveragePooling2D":
            return AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)
        else:
            raise ValueError("cannot recognize Pooling2D class name. Got {}".format(class_name))

    @classmethod
    def build_MaxPooling2D(cls, config):
        """
        e.g. config = "3x3-2x2-valid" means MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")

        Note that `pool_size` usually is a tuple like (3, 3), denoting a 3x3 window. 
        If only one integer is specified, the same window length will be used for both dimensions.
        """
        
        return cls._build_Pooling2D(config, "MaxPooling2D")

    @classmethod
    def build_AveragePooling2D(cls, config):
        """
        e.g. config = "3x3-2x2-valid" means AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")

        Note that `pool_size` usually is a tuple like (3, 3), denoting a 3x3 window. 
        If only one integer is specified, the same window length will be used for both dimensions.
        """
        
        return cls._build_Pooling2D(config, "AveragePooling2D")

    @classmethod
    def build_InputLayer(cls, shape):
        """
        e.g. shape = "9x2001x1" means InputLayer(input_shape=(9, 2001, 1))
        """

        input_shape = tuple(int(dim) for dim in shape.split("x"))
        
        return InputLayer(input_shape=input_shape)

    @classmethod
    def build_Dropout(cls, rate):
        rate = int(rate)
        return Dropout(rate=rate, seed=cls.DROP_OUT_RANDOM_SEED)

    @classmethod
    def build_Sequential(cls, model_config):
        """
        e.g. model_config = ("InputLayer:9x2001x1", "Conv2D:3-3-1-relu-valid", "Dense:1-sigmoid") will lead to a Sequential model of 
            
            Sequential([
                InputLayer(input_shape=(9, 2001, 1)),
                Conv2D(filters=3, kernel_size=3, strides=1, activation="relu"), 
                Dense(units=1, activation='sigmoid')
            ])
        """
        model = Sequential([])

        for layer_config in model_config:
            layer_name, *config = layer_config.split(":")
            
            if config:
                layer = getattr(cls, "build_" + layer_name)(config[0])
            else:
                layer = getattr(cls, "build_" + layer_name)()

            model.add(layer)

        return model

def compile_model(model, metric_names, optimizer="adam", build=True):
    loss = BinaryCrossentropy(name="loss")

    metrics = []
    if "acc" in metric_names:
        metrics.append(BinaryAccuracy(name="acc"))
    if "auprc" in metric_names:
        metrics.append(AUC(curve="PR", name="auprc"))
    if "auroc" in metric_names:
        metrics.append(AUC(curve="ROC", name="auroc"))
    if "fp" in metric_names:
        metrics.append(FalsePositives(name="fp"))
    if "fn" in metric_names:
        metrics.append(FalseNegatives(name="fn"))

    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    if build:
        # See https://stackoverflow.com/a/59356545 for the usage of build() function
        model.build()

def fit_model(model, x_train, y_train, x_test, y_test, batch_size, epochs, use_class_weight=False, use_sample_weight=False, use_reduce_rl=False, use_early_stopping=False, verbose=1):
    train_class_weight = calc_class_weight(y_train) if use_class_weight else None
    test_sample_weight = calc_sample_weight(y_test) if use_sample_weight else None

    if train_class_weight is not None:
        print('[cnn_util] using class weight {} for training'.format(train_class_weight))
    if test_sample_weight is not None:
        print('[cnn_util] using sample weight for testing')

    if x_test is None and y_test is None:
        validation_data = None
        callbacks = None
    else:
        validation_data = (x_test, y_test, test_sample_weight)

        callbacks = []
        if use_reduce_rl:
            callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3))
        if use_early_stopping:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True))
        if not callbacks:
            callbacks = None

    history = model.fit(x=x_train, y=y_train, class_weight=train_class_weight,
                        validation_data=validation_data,
                        batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose)

    return history

def get_activations(model, x, batch_size=1000):
    # get the activations from the last but one layer
    activation_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    with tf.device('/gpu:1'):
        activations = activation_model.predict(x, batch_size=batch_size)

    return activations
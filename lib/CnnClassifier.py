import keras
import tensorflow
import tensorboard

class CnnClassifier:

    def __init__(self):
        None


    def build_network(self, input_shape, num_outputs, pool_size, kernel_shape, num_filters=1, use_bias=1):
        # Input shape. 3D tensor with shape: (batch, steps, channels)
        # Output shape is softmax num_outputs (number of class labels)
        model = keras.Sequential()
        model.add(keras.layers.Conv1D(num_filters, kernel_size=kernel_shape,
                                      input_shape=input_shape,
                                      padding='valid',
                                      activation='relu',
                                      use_bias=use_bias,
                                      kernel_initializer=keras.initializers.he_normal(seed=None)))
        model.add(keras.layers.MaxPool1D(pool_size=pool_size, strides=None, padding='valid', data_format='channels_last'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(num_outputs,
                                     activation='softmax',
                                     kernel_initializer=keras.initializers.he_normal(seed=None)))
        return model

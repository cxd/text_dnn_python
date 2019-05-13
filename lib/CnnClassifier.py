import keras
import tensorflow
import tensorboard

class CnnClassifier:

    def __init__(self):
        None


    def build_network(self, vocab_size, max_sequence_length, num_outputs, pool_size, kernel_shape, embed_dim=50, num_channels=1, num_filters=1, use_bias=1, embedding_matrix=None):
        # Input shape. 3D tensor with shape: (batch, steps, channels)
        # Output shape is softmax num_outputs (number of class labels)
        model = keras.Sequential()
        # an embedding layer of vocab_size x embed_dim x max_sequence_length
        model.add(keras.layers.Embedding(vocab_size, embed_dim, input_length=max_sequence_length))

        print((max_sequence_length, embed_dim, num_channels))

        model.add(keras.layers.Conv1D(num_filters,
                                      input_shape=(max_sequence_length, embed_dim, num_channels),
                                      kernel_size=kernel_shape,
                                      padding='valid',
                                      activation='relu',
                                      use_bias=use_bias,
                                      data_format="channels_last",
                                      kernel_initializer=keras.initializers.he_normal(seed=None)))
        model.add(keras.layers.MaxPool1D(pool_size=pool_size, strides=None, padding='valid', data_format='channels_last'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(num_outputs,
                                     activation='softmax',
                                     kernel_initializer=keras.initializers.he_normal(seed=None)))

        # If the embedding matrix is supplied we assign it as the weights of the embedding layer.
        # we set the layer not be to able to be trained.
        if embedding_matrix is not None:
            model.layers[0].set_weights([embedding_matrix])
            model.layers[0].trainable = False

        return model

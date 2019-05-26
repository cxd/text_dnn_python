import keras


class WideCnnClassifier:

    def __init__(self):
        None


    def build_network(self,  vocab_size, max_sequence_length, num_outputs, kernel_shape, pool_size=2, embed_dim=50, num_channels=1, ngram_widths=[2,3,5], use_bias=1, embedding_matrix=None, train_embedding=False, cnn_padding='valid', dropout=0.3):
        # This network uses three filter sizes to process the input data in parallel.
        # Translating to 3 different lengths of ngram tuples per each parallel cnn.
        # this feeds into a softmax dense layer for output mapping.
        # Input shape. 3 replicas of 3D tensor with shape: (batch, steps, channels)
        # Output shape is softmax num_outputs (number of class labels)

        num_filters = ngram_widths

        # We will need to use the keras functional api
        # to train separate banks of parrallel convolutional layers we need to replicate the input and embeddings
        input1 = keras.layers.Input(shape=(max_sequence_length,))
        input2 = keras.layers.Input(shape=(max_sequence_length,))
        input3 = keras.layers.Input(shape=(max_sequence_length,))

        # an embedding layer of vocab_size x embed_dim x max_sequence_length
        embed1 = keras.layers.Embedding(vocab_size, embed_dim, input_length=max_sequence_length)
        embed2 = keras.layers.Embedding(vocab_size, embed_dim, input_length=max_sequence_length)
        embed3 = keras.layers.Embedding(vocab_size, embed_dim, input_length=max_sequence_length)

        embed1_fn = embed1(input1)
        embed2_fn = embed2(input2)
        embed3_fn = embed3(input3)
        # If the embedding matrix is supplied we assign it as the weights of the embedding layer.
        # we set the layer not be to able to be trained.
        if embedding_matrix is not None:
            embed1.set_weights([embedding_matrix])
            embed1.trainable = train_embedding
            embed2.set_weights([embedding_matrix])
            embed2.trainable = train_embedding
            embed3.set_weights([embedding_matrix])
            embed3.trainable = train_embedding


        # We will create 3 parallel cnn layers that are fully connected to the input.


        cnn1 =  keras.layers.Conv1D(num_filters[0],
                                    input_shape=(max_sequence_length, embed_dim, num_channels),
                                    kernel_size=kernel_shape,
                                    padding=cnn_padding,
                                    activation='relu',
                                    use_bias=use_bias,
                                    data_format="channels_last",
                                    kernel_initializer=keras.initializers.he_normal(seed=None))(embed1_fn)

        cnn2 =  keras.layers.Conv1D(num_filters[1],
                                    input_shape=(max_sequence_length, embed_dim, num_channels),
                                    kernel_size=kernel_shape,
                                    padding=cnn_padding,
                                    activation='relu',
                                    use_bias=use_bias,
                                    data_format="channels_last",
                                    kernel_initializer=keras.initializers.he_normal(seed=None))(embed2_fn)

        cnn3 =  keras.layers.Conv1D(num_filters[2],
                                    input_shape=(max_sequence_length, embed_dim, num_channels),
                                    kernel_size=kernel_shape,
                                    padding=cnn_padding,
                                    activation='relu',
                                    use_bias=use_bias,
                                    data_format="channels_last",
                                    kernel_initializer=keras.initializers.he_normal(seed=None))(embed3_fn)
        # dropout configuration
        drop1 = keras.layers.Dropout(dropout)(cnn1)
        drop2 = keras.layers.Dropout(dropout)(cnn2)
        drop3 = keras.layers.Dropout(dropout)(cnn3)

        # Max pooling configuration
        pool1 = keras.layers.MaxPool1D(pool_size=pool_size, strides=None, padding=cnn_padding, data_format='channels_last')(drop1)
        pool2 = keras.layers.MaxPool1D(pool_size=pool_size, strides=None, padding=cnn_padding, data_format='channels_last')(drop2)
        pool3 = keras.layers.MaxPool1D(pool_size=pool_size, strides=None, padding=cnn_padding, data_format='channels_last')(drop3)

        # In this architecture the layers are flattened directly and then fed into the softmax layer.
        flat1 = keras.layers.Flatten()(pool1)
        flat2 = keras.layers.Flatten()(pool2)
        flat3 = keras.layers.Flatten()(pool3)

        # Merge the three layers togethor.
        merged = keras.layers.merge.concatenate([flat1, flat2, flat3])

        # Add the softmax layer
        softmaxOut = keras.layers.Dense(num_outputs,
                                        activation='softmax',
                                        kernel_initializer=keras.initializers.he_normal(seed=None))(merged)



        model = keras.models.Model(inputs=[input1, input2, input3], outputs=softmaxOut)

        return model
import pandas as pd
import numpy as np
import numpy.random as nrnd
import os as os
import functools as functools

class GloveReader:
    # GloveReader manages the glove embeddings.

    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.model = None
        self.model_size = 0

    model_sizes = {
        'model50': os.path.join("glove", "glove.6B.50d.txt"),
        'model100': os.path.join("glove", "glove.6B.100d.txt"),
        'model200': os.path.join("glove", "glove.6B.200d.txt"),
        'model300': os.path.join("glove", "glove.6B.300d.txt")
    }

    model_dim = {
        'model50': 50,
        'model100': 100,
        'model200': 200,
        'model300': 300
    }


    def read_glove_model(self, model_file='model100'):
        # read the glove model that is specified by the corresponding key in self.model_sizes
        # the following keys are supported.
        # - model50
        # - model100
        # - model200
        # - model300
        # the number in each name above represents the size of the vector.
        self.model_size = GloveReader.model_dim[model_file]
        data_path = os.path.join(self.base_dir, GloveReader.model_sizes[model_file])
        model_data = pd.read_csv(data_path, sep='[\s\t]', header=None, index_col=0, engine='python',
                                 quoting=3, doublequote=False, error_bad_lines=False,
                                 warn_bad_lines=True)
        self.model = model_data.transpose()
        self.mu = self.model.mean(axis=1)
        self.sigma = np.cov(self.model)
        self.model_map = {}
        return self.model

    def get_vector_for_unknown_word(self, unknown):
        # Handle unknown words by generating a random multivariate vector
        # add the unknown word into the model for reuse.
        sample = nrnd.multivariate_normal(self.mu, self.sigma)
        self.model[unknown] = pd.Series(sample)
        self.model_map[unknown] = pd.Series(sample)
        return self.model[unknown]

    def get_vector_for_word(self, word, substitute_unknown=False, substitute_word='<NA>'):
        # Get a vector for a given word. If unknown generate a new vector 
        # based on the multivariate distribution.
        # note that we use an internal dictionary model_map to cache
        # a subset of lookups to avoid repeated lookups into the large glove dataset.
        # the idea is that we reduce the size of the selected words that we are working with.
        if self.model_map.__contains__(word):
            return self.model_map[word]
        elif self.model.columns.contains(word):
            series = self.model[word]
            self.model_map[word] = series
            return series
        else:
            if substitute_unknown is True:
                if self.model_map.__contains__(substitute_word):
                    return self.model_map[substitute_word]
                elif self.model.columns.contains(substitute_word):
                    series = self.model[substitute_word]
                    self.model_map[substitute_word] = series
                    return series
                else:
                    return self.get_vector_for_unknown_word(substitute_word)
            else:
                return self.get_vector_for_unknown_word(word)
 
    def get_word_matrix(self, vocabulary, pad_size=0, pad_word='<NA>', unknown_word='<UNKNOWN>'):
        # For each word in the vocabulary provided build a matrix
        # that has the corresponding glove embedding vector.
        # Unknown words are substituted for the padding word.
        # They are handled as 'NA' words since they are not in the vocabulary.
        rows = pad_size + len(vocabulary)
        matrix = np.zeros((rows, self.model.shape[0]))

        n = 0

        def assign(n, matrix, vector):
            matrix[n] = vector
            np.nan_to_num(matrix, copy=False)
            return matrix

        updatemat = lambda pair, vector: (pair[0] + 1, assign(pair[0], pair[1], vector))

        matrix_pairs = (n, matrix)
        if pad_size > 0:
            column = self.get_vector_for_word(pad_word)
            vectors = [column for i in range(0, pad_size)]
            matrix_pairs = functools.reduce(updatemat, vectors, (n, matrix))

        vectors = map(lambda word: self.get_vector_for_word(word, substitute_unknown=True, substitute_word=unknown_word), vocabulary)

        matrix_pairs = functools.reduce(lambda pair, vector: (pair[0] + 1, assign(pair[0], pair[1], vector)), vectors, matrix_pairs)

        return matrix_pairs[1]

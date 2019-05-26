import os as os
import pandas as pd
import numpy as np
import keras_preprocessing as kp
import re
import tensorflow as tf
from lib import Stopwords

class TextReader:

    def __init__(self, base_dir, stopwords_dir, stopword_reader=Stopwords.Stopwords()):
        self.base_dir = base_dir
        self.stopwords_dir = stopwords_dir
        self.stopword_reader = stopword_reader
        self.index_cache = {}



    def read_labeled_documents(self, file, class_indices=[0], text_delim='[,\s]', punc_filter='[\\!\\\'\\"#\\$%\\&\\(\\)\\*\\+,-\\./:;<=>\\?@\\[\\]^_`{|}~\t\n]', lower_case=True, file_encoding='utf-8', header=False, remove_stopwords=True):
        # file - Read a set of labelled documents from a flat file.
        # text_delim - The text delimiter supplied is a regular expression.
        # punc_filters - The punctuation filter is used to remove unwanted punctuation as a regex.
        # lower_case - Convert all text to lower case.
        # encoding - defaults to utf-8 encoding.
        # returns - a tuple (vocab, all_word_seq, all_class_labels)
        # the vocab is the list of unique words in the corpus.
        # the all_word_seq is a sequence of sequences one for each line in the file.
        # the all_class_labels is a collection of class labels one for each line in the file.
        stopwords = self.stopword_reader.load_stopwords(self.stopwords_dir)
        path = os.path.join(self.base_dir, file)
        filter_pattern = re.compile(punc_filter)
        split_pattern = re.compile(text_delim)
        vocab = ['<NA>', '<UNKNOWN>', '<start>', '<end>']
        all_labels = []
        all_words = []
        skipped = False
        with open(path, 'r', encoding=file_encoding) as fin:
            for line in fin:
                if header is True and skipped is False:
                    skipped = True
                    continue
                words = re.split(split_pattern, line)
                idx = [i for i in range(0, len(words))]
                pairs = [pair for pair in zip(idx, words)]
                class_label = [pair[1] for pair in pairs if class_indices.__contains__(pair[0])]
                words = [pair for pair in pairs if not class_indices.__contains__(pair[0])]
                words = [re.sub(filter_pattern, '', word[1]).lower() for word in words]
                words.insert(0, '<start>')
                words.insert(len(words), '<end>')
                if remove_stopwords is True:
                    words = [word for word in filter(lambda word: not stopwords.__contains__(word), words)]
                all_labels.append(class_label)
                all_words.append(words)
                for word in words:
                    if not vocab.__contains__(word) and not stopwords.__contains__(word):
                        vocab.append(word)
        vocab.sort()
        return vocab, all_words, all_labels

    def longest_sequence(self, all_words):
        # Find the longest sequence of all words.
        lengths = [words.__len__() for words in all_words]
        return max(lengths)

    def one_hot_encode_classes(self, all_classes, cls_index=0):
        # One hot encode the classes.
        # this results in a dataframe with column names corresponding to classes.
        # and vectors encoded in one hot encoding values.
        series = pd.Series([cls[cls_index] for cls in all_classes])
        return pd.get_dummies(series)


    def get_indices(self, vocab, words):
        def get_index(vocab, word):
            if self.index_cache.__contains__(word) is True:
                return self.index_cache[word]
            else:
                if vocab.__contains__(word):
                    self.index_cache[word] = vocab.index(word)
                    return self.index_cache[word]
                else:
                    self.index_cache['<UNKNOWN>'] = vocab.index('<UNKNOWN>')
                    return self.index_cache['<UNKNOWN>']

        return [get_index(vocab, word) for word in words]

    def get_word_from_index(self, vocab, index):
        vocab[index]

    def make_index_sequences(self, vocab, all_words):
        # Generate a dataframe consisting of padded sequences of word indexes.
        # Sequences are padded with <NA> tokens.
        # Each column of the resulting dataset contains the index position of the word from the vocabulary.
        # The resulting dataset is a left padded sequence in each row.
        # Sequences are padded with '<NA>' up to the maximum length.
        # The remainder of the sequence is the word index from each record in all words.
        max_len = self.longest_sequence(all_words)
        all_data = None
        padding = vocab.index('<NA>')
        for word_seq in all_words:
            indices = self.get_indices(vocab, word_seq)
            row = pd.Series([padding for i in range(0, max_len)])
            for i in range(0, indices.__len__()):
                target_idx = max_len - i - 1
                source_idx = indices.__len__() - i - 1
                row[target_idx] = indices[source_idx]
            df = pd.DataFrame(row).transpose()
            if all_data is None:
                all_data = df
            else:
                all_data = pd.concat([all_data, df])
        return all_data

    def pad_sentence(self, max_len, word_seq):
        # Pad a sentence from the left with '<NA>' token.
        # requires max_len to be specified.
        # The original max_len can be taken from the training set width.
        row = ['<NA>' for i in range(0, max_len)]
        for i in range(0, len(word_seq)):
            target_idx = max_len - i - 1
            source_idx = len(word_seq) - i - 1
            row[target_idx] = word_seq[source_idx]
        return row


    def vocab_to_embedding_matrix(self, embedding_reader, vocab):
        # Use the preloaded embedding to generate an embedding matrix
        # this is defined on the vocab which we use to obtain indexes
        # in our padded sequences.
        embed_frame = embedding_reader.get_word_matrix(vocab, pad_size=0)
        return embed_frame
import sys
import os
import numpy as np

sys.path.insert(0, '../lib')

from lib import TextReader
from lib import GloveReader

basedir = os.path.join('..', 'data')

reader = TextReader.TextReader(os.path.join(basedir, 'news20'), basedir)
vocab, all_words, all_classes = reader.read_labeled_documents('mini20-train.txt')

targets = reader.one_hot_encode_classes(all_classes)

sequences = reader.make_index_sequences(vocab, all_words)

embed_reader = GloveReader.GloveReader(base_dir=basedir)
glove1 = embed_reader.read_glove_model('model50')

# save the vocab embeddings because they are expensive to recreate.
import pickle
output = os.path.join(basedir, 'news20')
output = os.path.join(output, 'vocab_embeddings.pickle')

vocab_embedding = None
if os.path.exists(output):
    with open(output, 'rb') as fin:
        vocab_embedding = pickle.load(fin)
else:
    vocab_embedding = reader.vocab_to_embedding_matrix(embed_reader, vocab)
    with open(output, 'wb') as fout:
        pickle.dump(vocab_embedding, fout)


# the vocab embedding can be used with our cnn embedding model.
from lib import CnnClassifier

classifier = CnnClassifier.CnnClassifier()

max_sequence_length = sequences.shape[1]
embed_dim = vocab_embedding.shape[1]
num_outputs = targets.shape[1]
pool_size = targets.shape[1]
kernel_shape = 4

model = classifier.build_network(len(vocab), max_sequence_length, num_outputs, pool_size, kernel_shape, embed_dim, embedding_matrix=vocab_embedding)
model.summary()

model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

rows = sequences.shape[0]
# will shuffle the data
indices = np.arange(rows)
np.random.shuffle(indices)

shuffled_inputs = sequences.values[indices]
shuffled_targets = targets.values[indices]
train_percent = 0.8
trainX, validateX = np.split(shuffled_inputs, [int(train_percent*shuffled_inputs.shape[0])])
trainY, validateY = np.split(shuffled_targets, [int(train_percent*shuffled_targets.shape[0])])

history = model.fit(trainX,
                    trainY,
                    epochs=10,
                    validation_data=(validateX, validateY))

# To evaluate the model we want the test reader to load the test set since it was in a separate file.
# but we want to use the original vocabulary to define the sequences.
test_reader = TextReader.TextReader(os.path.join(os.path.join('..', 'data'), 'news20'),
                                    os.path.join('..', 'data'))
testvocab, test_words, test_classes = test_reader.read_labeled_documents('mini20-test.txt')

# get the test targets
test_targets = test_reader.one_hot_encode_classes(test_classes)
# get the test sequences but use the indexes in the vocabulary we trained on.
# words in the test set not in the original vocab are substituted with '<UNKNOWN>'
test_sequences = test_reader.make_index_sequences(vocab, test_words)

# we need to set the max width of sequences to equal the maximum width of the
# training data.
test_width = test_sequences.shape[1]
if test_width > max_sequence_length:
    delta = test_width - max_sequence_length
    test_sequence = test_sequences.iloc[:, delta:]
test_sequence.shape

loss, accuracy = model.evaluate(test_sequence, test_targets)


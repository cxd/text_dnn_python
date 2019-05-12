import sys
import os
sys.path.insert(0, '../lib')

from lib import TextReader
from lib import GloveReader

reader = TextReader.TextReader(os.path.join(os.path.join('..', 'data'), 'news20'), os.path.join('..', 'data'))
vocab, all_words, all_classes = reader.read_labeled_documents('mini20-train.txt')

targets = reader.one_hot_encode_classes(all_classes)

# not required
#sequences = reader.make_index_sequences(vocab, all_words)

embed_reader = GloveReader.GloveReader(base_dir=os.path.join('..', 'data'))
glove1 = embed_reader.read_glove_model('model50')

all_tensors, tf_dataset = reader.make_embedding_matrices(embed_reader, all_words)

import os as os
import io as io
import re

class Stopwords:

    def __init__(self, base_dir='data'):
        self.base_dir = base_dir

    def load_stopwords(self, base_dir=None, stopwords_file='stopwords.csv'):
        # Load stopwords from file.
        if base_dir is not None:
            self.base_dir = base_dir
        filename = os.path.join(self.base_dir, stopwords_file)

        self.stopwords = []
        pattern = re.compile('[\r\n]')
        with open(filename, 'r', encoding='utf-8') as fin:
            self.stopwords = [re.sub(pattern, '', word.lower()) for word in fin]
        return self.stopwords
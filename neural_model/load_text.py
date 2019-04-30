from numpy import array

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences


class Tokenize:
    def __init__(self, descriptions):
        self.descriptions = descriptions

    # convert a dictionary of clean descriptions to a list of descriptions
    def to_lines(self):
        all_desc = list()
        for key in self.descriptions.keys():
            [all_desc.append(d) for d in self.descriptions[key]]
        return all_desc

    def max_length(self):
        lines = self.to_lines()
        return max(len(d.split()) for d in lines)

    # fit a tokenizer given caption descriptions

    def create_tokenizer(self):
        lines = self.to_lines()
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    def create_sequences(self, max_length, desc_list, photo):
        tokenizer = self.create_tokenizer()
        vocab_size = len(tokenizer.word_index) + 1
        first, second, y = list(), list(), list()
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                first.append(photo)
                second.append(in_seq)
                y.append(out_seq)
        return array(first), array(second), array(y)

    def data_generator(self, photos, tokenizer, max_length):
        # loop for ever over images
        while 1:
            for key, desc_list in self.descriptions.items():
                # retrieve the photo feature
                photo = photos[key][0]
                in_img, in_seq, out_word = self.create_sequences(max_length, desc_list, photo)
                yield [[in_img, in_seq], out_word]

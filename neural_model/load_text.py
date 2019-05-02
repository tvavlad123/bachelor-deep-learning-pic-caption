from keras import Model
from keras.applications import VGG16
from keras.engine.saving import load_model
from keras.applications.densenet import preprocess_input
from keras_preprocessing.image import load_img, img_to_array
from numpy import array, argmax
from nltk.translate.bleu_score import corpus_bleu
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from pickle import load


class Tokenize:
    def __init__(self, descriptions):
        self.descriptions = descriptions

    def extract_features(self, filename):
        # load the model
        model = VGG16()
        # re-structure the model
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        # load the photo
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        return feature

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

    def data_generator(self, photos, max_length):
        # loop for ever over images
        while 1:
            for key, desc_list in self.descriptions.items():
                # retrieve the photo feature
                photo = photos[key][0]
                in_img, in_seq, out_word = self.create_sequences(max_length, desc_list, photo)
                yield [[in_img, in_seq], out_word]

    # map an integer to a word
    def word_for_id(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    # generate a description for an image
    def generate_desc(self, model, tokenizer, photo, max_length):
        # seed the generation process
        in_text = 'startseq'
        # iterate over the whole length of the sequence
        for i in range(max_length):
            # integer encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
            # predict next word
            yhat = model.predict([photo, sequence], verbose=0)
            # convert probability to integer
            yhat = argmax(yhat)
            # map integer to word
            word = self.word_for_id(yhat, tokenizer)
            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += ' ' + word
            # stop if we predict the end of the sequence
            if word == 'endseq':
                break
        return in_text

    # evaluate the skill of the model
    def evaluate_model(self, model, descriptions, photos, tokenizer, max_length):
        actual, predicted = list(), list()
        # step over the whole set
        for key, desc_list in descriptions.items():
            # generate description
            yhat = self.generate_desc(model, tokenizer, photos[key], max_length)
            # store actual and predicted
            references = [d.split() for d in desc_list]
            actual.append(references)
            predicted.append(yhat.split())
        # calculate BLEU score
        print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
        print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
        print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# filename = '../Flicker8k_text/Flickr_8k.trainImages.txt'
# data_loader = DataLoader()
# train = data_loader.load_set(filename)
# print('Dataset: %d' % len(train))
# # descriptions
# train_descriptions = data_loader.load_clean_descriptions('descriptions.txt', train)
# print('Descriptions: train=%d' % len(train_descriptions))
# tknz = Tokenize(train_descriptions)
# # prepare tokenizer
# tokenizer = tknz.create_tokenizer()
# # save the tokenizer
# dump(tokenizer, open('tokenizer.pkl', 'wb'))



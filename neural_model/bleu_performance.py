from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from numpy import argmax
import os
from neural_model.load_data import DataLoader
from neural_model.load_text import Tokenize


class BleuPerformance:
    def __init__(self, model, descriptions, photos, tokenizer, max_length):
        self.model = model
        self.descriptions = descriptions
        self.photos = photos
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate_desc(self, photo):
        in_text = 'startseq'
        for i in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            yhat = self.model.predict([photo, sequence], verbose=0)
            yhat = argmax(yhat)
            word = None
            for wrd, index in self.tokenizer.word_index.items():
                if index == yhat:
                    word = wrd
                    break
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        return in_text

    def evaluate_model(self):
        actual, predicted = list(), list()
        for key, desc_list in self.descriptions.items():
            yhat = self.generate_desc(self.photos[key])
            references = [d.split() for d in desc_list]
            actual.append(references)
            predicted.append(yhat.split())
        print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
        print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
        print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


def scoring():
    print("Test model")
    model = input("Enter model: ")
    filename = '../Flicker8k_Text/Flickr_8k.testImages.txt'
    data_loader = DataLoader()
    test = data_loader.load_set(filename)
    print('Dataset: %d' % len(test))
    # descriptions
    test_descriptions = data_loader.load_clean_descriptions('../preparation/descriptions.txt', test)
    print('Descriptions: train=%d' % len(test_descriptions))
    tknz = Tokenize(test_descriptions)
    # prepare tokenizer
    token = tknz.create_tokenizer()
    mod = load_model(os.getcwd() + "\\models\\" + model)
    test_features = data_loader.load_photo_features('../preparation/features.pkl', test)
    bleu = BleuPerformance(mod, test_descriptions, test_features, token, 34)
    print(bleu.evaluate_model())

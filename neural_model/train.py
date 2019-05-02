from keras.engine.saving import load_model

from neural_model.load_data import DataLoader
from neural_model.load_text import Tokenize
from neural_model.model_merge_architecture import NeuralModel
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


class TrainModel:
    def __init__(self, filename, descriptions, features, epochs):
        self.filename = filename
        self.descriptions = descriptions
        self.features = features
        self.epochs = epochs

    def train(self):
        load_data = DataLoader()
        train = load_data.load_set(self.filename)
        print('Dataset: %d' % len(train))
        # descriptions
        train_descriptions = load_data.load_clean_descriptions(self.descriptions, train)
        print('Descriptions: train=%d' % len(train_descriptions))
        # photo features
        train_features = load_data.load_photo_features(self.features, train)
        print('Photos: train=%d' % len(train_features))

        tokenize = Tokenize(train_descriptions)
        tokenizer = tokenize.create_tokenizer()
        vocab_size = len(tokenizer.word_index) + 1
        print('Vocabulary Size: %d' % vocab_size)
        # determine the maximum sequence length
        max_length = tokenize.max_length()
        print('Description Length: %d' % max_length)

        # define the model
        model = load_model('model_new_0.h5')
        # train the model, run epochs manually and save after each epoch
        steps = len(train_descriptions)
        for i in range(1, self.epochs):
            # create the data generator
            generator = tokenize.data_generator(train_features, max_length)
            # fit for one epoch
            model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
            # save model
            model.save('model_new_' + str(i) + '.h5')


filename = '../Flicker8k_text/Flickr_8k.trainImages.txt'
train_descriptions = '../preparation/descriptions.txt'
features = '../preparation/features.pkl'
train = TrainModel(filename, train_descriptions, features, 5)
train.train()

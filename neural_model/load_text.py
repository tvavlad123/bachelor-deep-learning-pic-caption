from keras import Model
from keras.applications import VGG16
from keras.applications.densenet import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.image import load_img, img_to_array
from nltk.translate.bleu_score import corpus_bleu


class Tokenize:
    def __init__(self, descriptions):
        self.descriptions = descriptions

    @staticmethod
    def extract_features(filename):
        model = VGG16()
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        return feature

    def to_lines(self):
        all_desc = list()
        for key in self.descriptions.keys():
            [all_desc.append(d) for d in self.descriptions[key]]
        return all_desc

    def max_length(self):
        lines = self.to_lines()
        return max(len(d.split()) for d in lines)

    def create_tokenizer(self):
        lines = self.to_lines()
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    # map an integer to a word
    @staticmethod
    def word_for_id(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

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

# filename = '../Flicker8k_text/Flickr_8k.testImages.txt'
# data_loader = DataLoader()
# train = data_loader.load_set(filename)
# print('Dataset: %d' % len(train))
# # descriptions
# train_descriptions = data_loader.load_clean_descriptions('descriptions.txt', train)
# print('Descriptions: train=%d' % len(train_descriptions))
# tknz = Tokenize(train_descriptions)
# # prepare tokenizer
# tokenizer = tknz.create_tokenizer()
# model = load_model('inject_model3.h5')
# train_features = data_loader.load_photo_features('features.pkl', train)
# tknz.evaluate_model(model, train_descriptions, train_features, tokenizer, 34)
# save the tokenizer
# dump(tokenizer, open('tokenizer.pkl', 'wb'))

import logging
import os
from pickle import load

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True


class CaptionGenerator:
    def __init__(self, model, tokenizer):
        self.max_length = 34
        self.model = model
        self.tokenizer = tokenizer

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

    def word_for_id(self, integer):
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def generate_desc(self, photo):
        photo = self.extract_features(photo)

        in_text = 'startseq'
        for i in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            next = self.model.predict([photo, sequence], verbose=0)
            next = argmax(next)
            word = self.word_for_id(next)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        return in_text


def generate(path):
    tok = load(open('tokenizer.pkl', 'rb'))
    mod = load_model('models/inject_all_8k8.h5')
    caption_generator = CaptionGenerator(mod, tok)
    print(caption_generator.generate_desc(os.getcwd() + "\\images\\" + path))


if __name__ == "__main__":
    generate()

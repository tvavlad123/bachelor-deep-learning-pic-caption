from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model


class DataPreparation:
    def __init__(self, directory):
        self.directory = directory

    def extract_features(self):
        # load the model
        model = VGG16()
        # re-structure the model
        # pop last layer of NN
        model.layers.pop()
        # model = shape of input + output last layer before classification
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        # summarize
        print(model.summary())
        # extract features from each photo
        features = dict()
        path = f'../{self.directory}'
        for name in listdir(path):
            # load an image from file
            filename = f'../{self.directory}/{name}'
            image = load_img(filename, target_size=(224, 224))
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            # only one sample + size1, size2 and channels
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)
            # get features
            feature = model.predict(image, verbose=0)
            # get image id
            image_id = name.split('.')[0]
            # store feature
            features[image_id] = feature
            print('>%s' % name)
        return features

    def preparation(self):
        feat = self.extract_features()
        print('Extracted Features: %d' % len(feat))
        # save to file
        dump(feat, open('features.pkl', 'wb'))


data_preparation = DataPreparation('Flicker8k_Dataset')
data_preparation.preparation()

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from keras.engine.saving import load_model
import neural_model.load_data
import neural_model.load_text
import neural_model.inject_merge_architecture


class Train:
    def __init__(self, filename, descriptions, features, path_model, func):
        self.filename = filename
        self.descriptions = descriptions
        self.features = features
        self.path_model = path_model
        self.func = func

    def start_training(self):
        load_data = neural_model.load_data.DataLoader()
        train = load_data.load_set(self.filename)
        print('Dataset: %d' % len(train))
        train_descriptions = load_data.load_clean_descriptions(self.descriptions, train)
        print('Descriptions: train=%d' % len(train_descriptions))
        train_features = load_data.load_photo_features(self.features, train)
        print('Photos: train=%d' % len(train_features))
        tokenize = neural_model.load_text.Tokenize(train_descriptions)
        tokenizer = tokenize.create_tokenizer()
        vocab_size = len(tokenizer.word_index) + 1
        print('Vocabulary Size: %d' % vocab_size)
        max_length = tokenize.max_length()
        print('Description Length: %d' % max_length)

        train_network = neural_model.inject_merge_architecture.NeuralModel(train_descriptions, vocab_size, max_length)
        train_network.train(self.path_model, train_descriptions, 20, train_features, tokenizer, self.func)

    def test_model(self, filename_test, model, model_file):
        load_data = neural_model.load_data.DataLoader()
        train = load_data.load_set(self.filename)
        print('Dataset: %d' % len(train))
        train_descriptions = load_data.load_clean_descriptions(self.descriptions, train)
        tokenize = neural_model.load_text.Tokenize(train_descriptions)
        tokenizer = tokenize.create_tokenizer()
        vocab_size = len(tokenizer.word_index) + 1
        print('Vocabulary Size: %d' % vocab_size)
        max_length = tokenize.max_length()
        print('Description Length: %d' % max_length)
        test = load_data.load_set(filename_test)
        print('Dataset: %d' % len(test))
        test_descriptions = load_data.load_clean_descriptions(self.descriptions, test)
        print('Descriptions: test=%d' % len(test_descriptions))
        test_features = load_data.load_photo_features(self.features, test)
        print('Photos: test=%d' % len(test_features))
        train_network = neural_model.inject_merge_architecture.NeuralModel(test_descriptions, vocab_size, max_length)
        if model in ["merge", "inject"]:
            model = load_model(model_file)
        X1test, X2test, ytest = train_network.create_sequences(tokenizer, 34, test_descriptions, test_features)
        print(model.evaluate([X1test, X2test], ytest, ))


def trainer():
    model_file = input("Model file: ")
    model = input("Model type: ")
    trainer = Train('../Flicker8k_Text/Flickr_8k.allImages.txt',
                    '../preparation/descriptions.txt',
                    '../preparation/features.pkl', os.getcwd() + "\\models\\" + model_file, model)
    trainer.start_training()


if __name__ == "__main__":
    trainer()
    # trainer.test_model('../Flicker8k_Text/Flickr_8k.testImages.txt', 'inject', 'models/inject_all_8k8.h5')

from keras import Input, Model
from keras.engine.saving import load_model
from keras.layers import Dropout, Dense, Embedding, LSTM, add
from keras.utils import plot_model, to_categorical
from keras_preprocessing.sequence import pad_sequences
from numpy import array


class NeuralModel:
    def __init__(self, descriptions, vocab_size, max_length):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.descriptions = descriptions

    def define_merge_model(self):
        inputs1 = Input(shape=(4096,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.summary()
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def define_inject_model(self):
        inputs1 = Input(shape=(4096,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = add([fe2, se2])
        decoder1 = LSTM(256)(se3)
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.summary()
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def create_sequences(self, tokenizer, max_length, descriptions, photos):
        X1, X2, y = list(), list(), list()
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    X1.append(photos[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
        return array(X1), array(X2), array(y)

    def create_sequences_progressive_loading(self, tokenizer, photo, desc_list):
        first, second, y = list(), list(), list()
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                first.append(photo)
                second.append(in_seq)
                y.append(out_seq)
        return array(first), array(second), array(y)

    def data_generator(self, descriptions, photos, tokenizer):
        while 1:
            for key, desc_list in descriptions.items():
                photo = photos[key][0]
                in_img, in_seq, out_word = self.create_sequences_progressive_loading(tokenizer, photo, desc_list)
                yield [[in_img, in_seq], out_word]

    def train(self, model_file, descriptions, epochs, features, tokenizer, func):
        if not model_file:
            if func == "merge":
                model = self.define_merge_model()
            if func == "inject":
                model = self.define_inject_model()
        else:
            if func in ["merge", "inject"]:
                model = load_model(model_file)
        steps = len(self.descriptions)
        for i in range(7, epochs):
            generator = self.data_generator(descriptions, features, tokenizer)
            model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
            model.save('merge_all_8k' + str(i) + '.h5')

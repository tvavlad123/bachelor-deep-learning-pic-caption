from pickle import load

import preparation.text_preparation


class DataLoader:

    @staticmethod
    def load_set(filename):
        data_preparation = preparation.text_preparation.TextPreparation()
        doc = data_preparation.load_doc(filename)
        dataset = list()
        for line in doc.split('\n'):
            if len(line) < 1:
                continue
            identifier = line.split('.')[0]
            dataset.append(identifier)
        return set(dataset)

    @staticmethod
    def load_clean_descriptions(filename, dataset):
        data_preparation = preparation.text_preparation.TextPreparation()
        doc = data_preparation.load_doc(filename)
        descriptions = dict()
        for line in doc.split('\n'):
            tokens = line.split()
            image_id, image_desc = tokens[0], tokens[1:]
            if image_id in dataset:
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                descriptions[image_id].append(desc)
        return descriptions

    @staticmethod
    def load_photo_features(filename, dataset):
        all_features = load(open(filename, 'rb'))
        features = {k: all_features[k] for k in dataset}
        return features

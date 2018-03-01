#!/usr/bin/env python3
import sys
import os.path
import argparse
import logging
import time
import progressbar
import uuid
import pickle
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model, model_from_json
from keras.engine import Input
from keras.layers import Activation, Convolution1D, Embedding, Merge, Lambda, merge, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from preprocess import pp_with_duplicate_quote, pp_without_duplicate_quote, pp, rus_mystem

class SentenceSimilarity:
    data = {}
    metrics = {}
    _data_sets = ['training', 'validation', 'testing']
    _data_subkeys = ['texts_a',
                     'texts_b',
                     'texts_label',
                     'data',
                     'labels']
    _metrics_keys = ['loss',
                     'accuracy',
                     'samples']    
    for k in _data_sets:
        data[k] = dict.fromkeys(_data_subkeys,[])
        metrics[k] = dict.fromkeys(_metrics_keys,0.0)        
    highest_length = 0
    set_of_labels = set()
    word_index = None # TODO: get rid of. Use tokenizer.word_index
    embedding_matrix = None
    uuid = None
    filename_tokenizer = None
    tokenizer = None
    #
    # defaults (from Bogdanova et al.):
    # TODO: print them to log
    embedding_dim = 200
    conv_filter_dim = 300
    conv_filters = [3]
    lr = 0.005
    optimizer = SGD(lr)
    loss = 'mse'
    validation_split = 0.0
    nb_epoch = 20
    char_level = False
    pooling = 'max'
    
    def __init__(self, filename_train, filename_validation, filename_test, filename_embeddings, model_uuid, epoch_id):
        """Initialize the classifier."""
        if model_uuid:
            self.uuid = model_uuid
            # TODO: check if directory exists
        else:
            self.uuid = uuid.uuid4().hex
            os.makedirs(os.path.join("cached_models",self.uuid))
        self.filename_train = filename_train
        self.filename_validation = filename_validation
        self.filename_test = filename_test
        self.filename_embeddings = filename_embeddings
        if model_uuid:
            self.filename_model = os.path.join("cached_models",self.uuid, "model.json")
            self.filename_tokenizer = os.path.join("cached_models", self.uuid, "tokenizer.pickle")
            if epoch_id:
                self.filename_weights = os.path.join("cached_models",self.uuid, "ep{}.h5".format(epoch_id))
            else:
                self.filename_weights = os.path.join("cached_models",self.uuid, "final.h5")
        
    def _preprocess_text(self, text):
        """Clean the input text
        note: just a stub. Keras Tokenizer lowercases text by default."""
        pattern_obj = re.compile(r'[^\w\d\s]')
        text = pattern_obj.sub("", text)
        return text

    def _parse_tsv(self, filename, data_set, separator='\t'):
        """Read the data in the given TSV file into the internal data structure."""
        assert data_set in self._data_sets
        logging.info('Reading {} sentence pairs from {}:'.format(data_set, filename))
        time_start = time.perf_counter()
        data_ref = self.data[data_set]
        data_ref['texts_a'] = []
        data_ref['texts_b'] = []
        data_ref['texts_label'] = []
        class_count = {}        
        with open(filename) as handle:
            bar = progressbar.ProgressBar()
            for line in bar(handle):
                if len(line.strip().split(separator)) != 3:
                    print(line)
                label, text_a, text_b = line.strip().split(separator)
                data_ref['texts_a'].append(self._preprocess_text(text_a))
                data_ref['texts_b'].append(self._preprocess_text(text_b))
                data_ref['texts_label'].append(label)
                if label not in class_count:
                    class_count[label] = 0
                class_count[label] += 1                 
        self.set_of_labels.update(data_ref['texts_label'])
        logging.info('...read {0} pairs in {1:.2f} seconds.'.format(len(data_ref['texts_a']), time.perf_counter() - time_start))
        total = sum(class_count.values())
        class_count_ratio = ['{0} = {1} ({2:.1%})'.format(cls, class_count[cls], class_count[cls]/total) for cls in sorted(class_count.keys())]
        logging.info('...class distribution: ' + ' | '.join(class_count_ratio))

    def _init_tokenizer(self):
        with open(self.filename_tokenizer, 'rb') as f:
            self.tokenizer = pickle.load(f)
            logging.info('...tokenizer was loaded from {};'.format(f))
        
    def _vectorize_data(self):
        """Vectorize the textual data"""
        # Fit on training data
        logging.info('Vectorizing data:')
        time_start = time.perf_counter()
        # TODO: refactor -- filename_tokenizer only initialized when model_uuid is provided
        if not self.filename_tokenizer:
            self.tokenizer = Tokenizer(char_level=self.char_level,
                                       lower=False,
                                       filters="")
            self.tokenizer.fit_on_texts(self.data['training']['texts_a'])
            self.tokenizer.fit_on_texts(self.data['training']['texts_b'])
            logging.info('...fitted tokenizer in {0:.2f} seconds;'.format(time.perf_counter() - time_start))
        else:
            self._init_tokenizer()
        self.word_index = self.tokenizer.word_index
        logging.info('...found {0} unique tokens;'.format(len(self.word_index)))
        # Vectorize data sets
        time_start = time.perf_counter()
        # Q: Is sorting necessary? 
        all_labels = sorted(list(self.set_of_labels))
        for data_set in self._data_sets:
            seq_a = self.tokenizer.texts_to_sequences(self.data[data_set]['texts_a'])
            seq_b = self.tokenizer.texts_to_sequences(self.data[data_set]['texts_b'])
            self.data[data_set]['data'] = [seq_a, seq_b]
            self.data[data_set]['labels'] = np.asarray([all_labels.index(l)
                                                        for l in self.data[data_set]['texts_label']])
            self.metrics[data_set]['samples'] = len(self.data[data_set]['labels'])
            
    def _initialize_embeddings(self):
        """Initialize the word embeddings matrix.
        For the sake of efficiency, the embedding matrix only contains the vectors of terms that were seen in the
        training data.
        """
        cache_filename = '{0}.min.cache.npy'.format(self.filename_embeddings)
        embeddings = {}
        if False:#os.path.isfile(cache_filename):
            logging.info('Load embeddings from cached file {0}:'.format(cache_filename))
            self.embedding_matrix = np.load(cache_filename)
            self.embedding_dim = self.embedding_matrix.shape[1]
            logging.info('...loaded cached embedding matrix with shape {0};'.format(self.embedding_matrix.shape))
        else:
            logging.info('Load embeddings from {0}:'.format(self.filename_embeddings))
            time_start = time.perf_counter()
            with open(self.filename_embeddings) as file_handle:
                self.embedding_dim = int(file_handle.readline().strip().split()[1])  # Read dimension from the first line.
                for line in file_handle:
                    fields = line.split()
                    word_form, word_vector = fields[0], np.asarray(fields[1:], dtype='float32')
                    embeddings[word_form] = word_vector
            logging.info('...read {0} word embeddings in {1:.2f} seconds;'.format(len(embeddings),
                                                                                  time.perf_counter() - time_start))
            # Create the initial embedding matrix, including only those terms that were observed in the training data.
            # TODO: Why add one to the vocabulary size?
            embedding_matrix_shape = (len(self.word_index) + 1, self.embedding_dim)
            self.embedding_matrix = np.zeros(embedding_matrix_shape)
            for word_form, index in self.word_index.items():
                if word_form in embeddings:
                    self.embedding_matrix[index] = embeddings[word_form]
                else:
                    pass #self.embedding_matrix[index] = np.random.randn(self.embedding_dim)
            logging.info('...created embedding matrix with shape {0};'.format(self.embedding_matrix.shape))
            np.save(cache_filename, self.embedding_matrix)
            logging.info('...cached matrix in file {0}.'.format(cache_filename))

    def _create_model(self):
        """Create CNN model.
        """
        logging.info('Creating CNN model:')
        #
        # Input layers
        input_a = Input(shape=(None,), dtype='int32')
        input_b = Input(shape=(None,), dtype='int32')
        #
        # Embedding layers with shared weights
        if type(self.embedding_matrix).__module__ == np.__name__:
            weights = [self.embedding_matrix] 
        else:
            logging.info('Embedding weights were randomly initialized.')
            weights = None 
        embedding_layer = Embedding(len(self.word_index)+1,
                                    self.embedding_dim,
                                    weights=weights)
        a = embedding_layer(input_a)
        b = embedding_layer(input_b)
        #
        # Convolutional layers with shared weights. Can be swithed off.
        if self.conv_filters:
            if len(self.conv_filters) > 1:
                cnns = [Convolution1D(filter_length=filt,
                                      nb_filter=self.conv_filter_dim,
                                      activation='tanh',
                                      border_mode='same')
                        for filt in self.conv_filters]
                a = merge([cnn(a) for cnn in cnns], mode='concat')
                b = merge([cnn(b) for cnn in cnns], mode='concat')
            else:
                conv_layer = Convolution1D(filter_length=self.conv_filters[0],
                                           nb_filter=self.conv_filter_dim,
                                           activation='tanh',
                                           border_mode='same')
                a = conv_layer(a)
                b = conv_layer(b)
        #
        # Pooling layers with activation
        if self.pooling == 'avg':
            pooling_layer = GlobalAveragePooling1D()
        else:
            pooling_layer = GlobalMaxPooling1D()
        a = pooling_layer(a)
        b = pooling_layer(b)
        activation_layer = Activation('tanh')
        a = activation_layer(a)
        b = activation_layer(b)
        #
        # Similarity calculation layer
        similarity = merge([a,b], mode='cos')
        self.model = Model(input=[input_a,input_b],
                           output=similarity)
        logging.info('...model created.')

    def _compile_model(self):
        """Compile model.
        """
        logging.info('Compiling model:')
        # TODO: support other optimizers
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['accuracy'])
        logging.info('...model {} compiled with optimizer: {}, lr (sgd-only): {}, loss: {}. '.format(self.uuid,
                                                                                                     self.optimizer,
                                                                                                     self.lr,
                                                                                                     self.loss))
        
    def _split_and_shuffle_data(self):
        """Shuffles training data. 
        Splits training data if it's necessary.        
        TODO: support testing as well.
        """
        if not self.data['validation']['data'][0] and self.validation_split > 0:
            nb_validation_samples = int(self.validation_split * len(self.data['training']['labels']))
            x_train = np.asarray([self.data['training']['data'][0][:-nb_validation_samples],
                                  self.data['training']['data'][1][:-nb_validation_samples]])
            y_train = np.asarray(self.data['training']['labels'][:-nb_validation_samples])
            x_val = np.asarray([self.data['training']['data'][0][-nb_validation_samples:],
                                self.data['training']['data'][1][-nb_validation_samples:]])
            y_val = np.asarray(self.data['training']['labels'][-nb_validation_samples:])
        else:
            x_train = np.asarray([self.data['training']['data'][0],
                                  self.data['training']['data'][1]])
            y_train = np.asarray(self.data['training']['labels'])
            x_val = np.asarray([self.data['validation']['data'][0],
                                self.data['validation']['data'][1]])
            y_val = np.asarray(self.data['validation']['labels'])
        self.metrics['training']['samples'] = y_train.shape[0]
        self.metrics['validation']['samples'] = y_val.shape[0]
        if True: # only training data is shuffled
            indices = np.arange(x_train[0].shape[0])
            np.random.shuffle(indices)
            x_train = [x_train[0][indices], x_train[1][indices]]
            y_train = y_train[indices]
        return (x_train, y_train, x_val, y_val)

    def _process_samples(self, mode, x, y):
        """Train, validate, or test on each sample.
        """
        sum_loss = 0.0
        sum_accuracy = 0.0
        assert mode in self._data_sets
        if mode == 'training':            
            bar = progressbar.ProgressBar()
            for i in bar(range(len(y))):
                x_a = np.array([x[0][i]])
                x_b = np.array([x[1][i]])
                y_i = np.array([y[i]])
                i_loss, i_acc = self.model.train_on_batch([x_a, x_b], y_i)
                sum_loss += i_loss
                sum_accuracy += i_acc
        else:
            for i in range(len(y)):
                x_a = np.array([x[0][i]])
                x_b = np.array([x[1][i]])
                y_i = np.array([y[i]])
                i_loss, i_acc = self.model.test_on_batch([x_a, x_b], y_i)
                sum_loss += i_loss
                sum_accuracy += i_acc  
        self.metrics[mode]['loss'] = sum_loss / self.metrics[mode]['samples']
        self.metrics[mode]['accuracy'] = sum_accuracy / self.metrics[mode]['samples']
        logging.info('...{0} loss: {1:.5f}, accuracy: {2:.5f}'.format(mode,
                                                              self.metrics[mode]['loss'],
                                                              self.metrics[mode]['accuracy']))
            
    def _fit_model(self):
        """Fit model on training data with batch_size=1        
        TODO: add baseline information into logging
        """
        x_train, y_train, x_val, y_val = self._split_and_shuffle_data()
        logging.info('Train on {} samples, validate on {} samples'.format(len(y_train),
                                                                          len(y_val)))
        for epoch in range(1,self.nb_epoch+1):
            logging.info('Epoch {}/{}'.format(epoch,self.nb_epoch))
            x_train, y_train, x_val, y_val = self._split_and_shuffle_data()
            self._process_samples('training',x_train,y_train)
            if len(y_val) > 0:
                self._process_samples('validation',x_val,y_val)
            self.model.save_weights(os.path.join("cached_models",self.uuid,"ep{}.h5".format(epoch)))
        # Save model, final weights,tokenizer
        with open(os.path.join("cached_models",self.uuid,"model.json"), 'a') as f:
            f.write(self.model.to_json())
        self.model.save_weights(os.path.join("cached_models",self.uuid,"final.h5"))
        pickle.dump(self.tokenizer, open(os.path.join("cached_models",self.uuid,"tokenizer.pickle"), 'wb'))
        logging.info('...training complete.')

    def _evaluate_model(self):
        """Tests model on test set.
        """
        x_test = np.asarray([self.data['testing']['data'][0],
                             self.data['testing']['data'][1]])
        y_test = np.asarray(self.data['testing']['labels'])
        logging.info('Test on {} samples.'.format(y_test.shape[0]))
        self._process_samples('testing',x_test,y_test)        

    def load(self):
        """Loads model from file"""
        with open(self.filename_model) as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(self.filename_weights,
                                by_name=False)
        # TODO: save parameters and compile with them
        self._compile_model()

    def load_and_test(self, separator='\t'):
        """Loads model from file and runs it over a test set"""
        self.load()
        self._parse_tsv(self.filename_test, 'testing', separator=separator)
        self._vectorize_data()
        self._evaluate_model()

    def predict(self, text_a, text_b):
        if not self.tokenizer:
            self._init_tokenizer()
        sa = np.array(self.tokenizer.texts_to_sequences([text_a]))
        sb = np.array(self.tokenizer.texts_to_sequences([text_b]))
        if sa.size == 0 or sb.size == 0:
            return None        
        proba = self.model.predict_on_batch([sa,sb])
        # taken from keras model.predict_classes method  
        if proba.shape[-1] > 1:
            prediction = proba.argmax(axis=-1)
        else:
            prediction = (proba > 0.5).astype('int32')
        return int(prediction[0][0])
        
    def run(self, separator='\t'):
        """Runs all steps: 
        reads and vectorizes data; 
        reads embeddings; 
        creates, fits and evaluates model.
        """
        self._parse_tsv(self.filename_train, 'training', separator=separator)
        if self.filename_validation:
            self._parse_tsv(self.filename_validation, 'validation', separator=separator)
        if self.filename_test:
            self._parse_tsv(self.filename_test, 'testing', separator=separator)
        self._vectorize_data()
        if self.filename_embeddings:
            self._initialize_embeddings()
        self._create_model()
        self._compile_model()
        print('Model summary:')
        self.model.summary()
        self._fit_model()
        if self.filename_test:
            self._evaluate_model()

    def set_hyperparameters(self,mode='replicate'):
        if mode in ['pp_impact', 'we_impact']:
            self._preprocess_text = lambda x: pp_without_duplicate_quote(x)
        elif mode == 'pt_1':
            self.nb_epoch = 5
            self._preprocess_text = lambda x: pp(x)
        elif mode == 'pt_2':
            self._preprocess_text = lambda x: pp(x)
            self.nb_epoch = 5
            self.embedding_dim = 400
            self.conv_filter_dim = 1000
        elif mode == 'ru_ns':
            self._preprocess_text = lambda x: rus_mystem(x)
            self.nb_epoch = 5
            self.conv_filter_dim = 300
            self.optimizer = 'rmsprop'
        elif mode == 'ru_word':
            self.nb_epoch = 5
            self.conv_filters = [3,5,8,12]
            self.optimizer = 'rmsprop'
            self.embedding_dim = 300
        elif mode == 'ru_char':
            self.conv_filters = [2,3,5,7,9,11]
            self.conv_filter_dim = 100
            self.embedding_dim = 100
            self.optimizer = 'rmsprop'
            self.nb_epoch = 20
            self.char_level = True
        else:
            self._preprocess_text = lambda x: pp_with_duplicate_quote(x)
            

    
def _handle_command_line():
    description = 'Trains and evaluates a CNN for detecting similarity between pairs of sentences.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('mode', help='mode for setting hyperparameters')
    multiple_files_group = parser.add_argument_group('when running over separate data sets')
    multiple_files_group.add_argument('--train', metavar='FILE',
                                      help='file with data for training')
    multiple_files_group.add_argument('--test', metavar='FILE',
                                      help='file with data for testing')
    multiple_files_group.add_argument('--validation', metavar='FILE',
                                      help='file with data for validation')
    pretrained_model_group = parser.add_argument_group('when using pre-trained model')
    pretrained_model_group.add_argument('--uuid', metavar='FILE',
                                        help='uuid of model in cached_models folder')
    pretrained_model_group.add_argument('--epoch', metavar='FILE',
                                        help='number of epoch to get weights from')
    single_file_group = parser.add_argument_group('when running over a single data set (NOT IMPLEMENTED YET)')
    parser.add_argument('--w2v', metavar='FILE',
                        help='file with word embeddings')
    parser.add_argument('--version', action='version', version='%(prog)s 0.2')
    args = vars(parser.parse_args())
    # TODO: check validity of arguments
    return args

def main():
    args = _handle_command_line()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s:%(message)s')
    classifier = SentenceSimilarity(args['train'],
                                    args['validation'],
                                    args['test'],
                                    args['w2v'],
                                    args['uuid'],
                                    args['epoch'])
    classifier.set_hyperparameters(mode=args['mode'])
    # DEFAULTS can be overriden here
    # classifier.nb_epoch = 10
    # classifier.optimizer = 'sgd'
    # classifier.lr = 0.005
    # classifier.validation_split = 0.1
    # classifier.embedding_dim = 100
    # classifier.conv_filter_dim = 300
    # classifier.conv_filters = [3]
    # classifier.char_level = False
    classifier.pooling = 'max'
    if args['uuid']:
        classifier.load_and_test(
            #separator='|'
        )
    else:
        classifier.run(
            #separator='|'
        )

# Launcher stub
if __name__ == '__main__':
    main()

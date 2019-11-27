import configparser
import pickle as pk
import tensorflow as tf
from tqdm import tqdm
from tqdm import trange

from .hamming import hamming_score
from sklearn.metrics import f1_score
from .bilstm_model import Bilstm_model
import numpy as np
import pickle as pk



class Bilstm(object):
    
    def __init__(self, X_train, y_train, X_test, y_test, configuration = None):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_test
        self.y_val   = y_test
        
        self.old_configuration = {
                         'vocab_size'                 : 7000,
                         'no_of_labels'               : 9,
                         'rnn_units'                  : 256, 
                         'word_embedding_dim'         : 300, 
                         'learning_rate'              : 0.001, 
                         'pretrained_embedding_matrix': None,
                         'dropout'                    : 0.2,
                         'epoch'                      : 1,
                         'batch_size'                 : 128,
                         'result_path'                : '/Users/monk/Desktop',
                         'last_output'                : False,
                         'train_embedding'            : True
                        }
        
        if configuration:
            self.old_configuration.update(configuration)
            
        if self.old_configuration['pretrained_embedding_matrix'] is None:
            self.embedding_mat = None
        else:
            with open(self.old_configuration['pretrained_embedding_matrix'], 'rb') as f:
                self.embedding_mat = np.array(pk.load(f))
                
            
        
    
    def default_configuration(self):
        
        default_conf = {
                        'vocab_size'      : 'vocab_size of corpus', 
                        'rnn_unit'        : 'bi-directional_rnn units', 
                        'embedding_dim'   : 'word_embedding_dim', 
                        'learning rate'   : 'learning rate of model', 
                        'embedding_matrix': 'path of embedded matrix  (ex glove, elmo )',
                        'train_embedding' : 'train glove embedding or not', 
                        'last_output'     : 'use lstm last state or use final output of all states'
                       }
        return default_conf
    
    
    # train data loader
    def get_train_data(self, batch_size, slice_no):


        batch_data_j = np.array(self.X_train[slice_no * batch_size:(slice_no + 1) * batch_size])
        batch_labels = np.array(self.y_train[slice_no * batch_size:(slice_no + 1) * batch_size])

        max_sequence = max(list(map(len, batch_data_j)))

        # getting Max length of sequence
        padded_sequence = [i + [0] * (max_sequence - len(i)) if len(i) < max_sequence else i for i in batch_data_j]

        return {'sentenc': padded_sequence, 'labels': batch_labels }
    
    
    # test data loader
    def get_test_data(self, batch_size,slice_no):


        batch_data_j = np.array(self.X_val[slice_no * batch_size:(slice_no + 1) * batch_size])
        batch_labels = np.array(self.y_val[slice_no * batch_size:(slice_no + 1) * batch_size])

        max_sequence = max(list(map(len, batch_data_j)))

        padded_sequence = [i + [0] * (max_sequence - len(i)) if len(i) < max_sequence else i for i in batch_data_j]

        return {'sentenc': padded_sequence, 'labels': batch_labels}
    
    
    def evaluate_(self, model, epoch_, batch_size = 120):

        sess = tf.get_default_session()
        iteration = len(self.X_val) // batch_size

        sub_accuracy    = []
        hamming_score_a = []
        hamming_loss_   = []

        micr_ac = []
        weight_ac = []

        for i in range(iteration):

            data_g = self.get_test_data(batch_size,i)

            sentences_data = data_g['sentenc']
            labels_data    = data_g['labels']

            network_out, targe = sess.run([model.predictions,model.targets], feed_dict={model.placeholders['sentence']: sentences_data,
                                                                                        model.placeholders['labels']: labels_data, 
                                                                                        model.placeholders['dropout']: 0.0})

            h_s     = hamming_score(targe, network_out)

            ham_sco = h_s['hamming_score']
            sub_acc = h_s['subset_accuracy']
            ham_los = h_s['hamming_loss']

            sub_accuracy.append(sub_acc)
            hamming_score_a.append(ham_sco)
            hamming_loss_.append(ham_los)



            micr_ac.append(f1_score(targe, network_out, average='micro'))
            weight_ac.append(f1_score(targe, network_out, average='weighted'))

        return {  'subset_accuracy' : np.mean(np.array(sub_accuracy)) , 
                  'hamming_score'   : np.mean(np.array(hamming_score_a)) , 
                  'hamming_loss'    : np.mean(np.array(hamming_loss_)), 
                   'micro_ac'       : np.mean(np.array(micr_ac)), 
                   'weight_ac'      : np.mean(np.array(weight_ac)) , 'epoch': epoch_ }
    
    
    
    
    def train_model(self, model):

        batch_size = int(self.old_configuration['batch_size'])
        epoch      = int(self.old_configuration['epoch'])

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            iteration = len(self.X_train) // batch_size


            for i in range(epoch):
                t = trange(iteration, desc='Bar desc', leave=True)

                for j in t:



                    data_g = self.get_train_data(batch_size,j)
                    sentences_data = data_g['sentenc']
                    labels_data    = data_g['labels']



                    network_out, train, targe, losss  = sess.run([model.predictions, model.optimizer, model.targets,model.loss],
                                              feed_dict={model.placeholders['sentence']: sentences_data,
                                                         model.placeholders['labels']: labels_data,
                                                         model.placeholders['dropout']: self.old_configuration['dropout']})

                    t.set_description("epoch {},  iteration {},  F1_score {},  loss {}".format(i,
                                                                                           j,
                                                                                           f1_score(targe, 
                                                                                                    network_out, 
                                                                                                    average='micro'), 
                                                                                           losss))
                    t.refresh() # to show immediately the update


                val_data = self.evaluate_(model, i, batch_size = 100)
                print("validation_acc",val_data)
                with open(str(self.old_configuration['result_path']) + '/result.txt', 'a') as f:
                    f.write(str({'test_accuracy':  val_data}) + '\n')

    def train(self):
        model = Bilstm_model(vocab_size            =   int(self.old_configuration['vocab_size']),
                       no_of_labels                =   int(self.old_configuration['no_of_labels']),
                       rnn_units                   =   int(self.old_configuration['rnn_units']), 
                       word_embedding_dim          =   int(self.old_configuration['word_embedding_dim']),
                       pretrained_embedding_matrix =   self.embedding_mat,
                       learning_rate               =   float(self.old_configuration['learning_rate']),
                       train_embedding             =   self.old_configuration['train_embedding'],
                       last_output                =    self.old_configuration['last_output'])

        self.train_model(model)
    
    
    
    
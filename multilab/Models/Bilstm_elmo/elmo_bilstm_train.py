import pickle as pk
import tensorflow as tf
from tqdm import tqdm
from tqdm import trange


from .hamming import hamming_score
from sklearn.metrics import f1_score
from .elmo_bilstm import BiElmo_model
import numpy as np
import pickle as pk



class BiLstm_Elmo(object):


    
    def __init__(self, X_train, y_train, X_test, y_test, configuration = None):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_test
        self.y_val   = y_test
        
        self.old_configuration = {
                         'no_of_labels'               : 9,
                         'learning_rate'              : 0.001,
                         'rnn_units'                  : 256,
                         'last_output'                : False,
                         'epoch'                      : 1,
                         'batch_size'                 : 128,
                         'dropout'                    : 0.2,
                         'model_type'                 : 'base',
                         'train_elmo'                 : True,
                         'result_path'                : '/Users/monk/Desktop'
                        }
                
            
        if configuration:
            self.old_configuration.update(configuration)
    
    def default_configuration(self):
        
        default_conf = {
                        'no_of_labels'    : 'total number of labels',
                        'learning rate'   : 'learning rate of model',
                        'epoch'           : 'epoch for training',
                        'result_path'     : 'path for result.txt file',
                        'dropout'         : 'dropout'
                       }
        return default_conf
    
    
    # train data loader
    def get_train_data(self, batch_size, slice_no):


        batch_data_j = np.array(self.X_train[slice_no * batch_size:(slice_no + 1) * batch_size])
        batch_labels = np.array(self.y_train[slice_no * batch_size:(slice_no + 1) * batch_size])


        return {'sentenc': batch_data_j, 'labels': batch_labels }
    
    
    # test data loader
    def get_test_data(self, batch_size,slice_no):


        batch_data_j = np.array(self.X_val[slice_no * batch_size:(slice_no + 1) * batch_size])
        batch_labels = np.array(self.y_val[slice_no * batch_size:(slice_no + 1) * batch_size])


        return {'sentenc': batch_data_j, 'labels': batch_labels}
    
    
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
                                                                                        model.placeholders['drop']: 0.0})

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
                                                         model.placeholders['drop']: self.old_configuration['dropout']})

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
        model = BiElmo_model(
            
                       no_of_labels                =   int(self.old_configuration['no_of_labels']),
                       learning_rate                =   float(self.old_configuration['learning_rate']),
                       rnn_units                    =   self.old_configuration['rnn_units'],
                       train_elmo                   =   self.old_configuration['train_elmo'],
                       last_output                  =   self.old_configuration['last_output'])
                       

        self.train_model(model)

import tensorflow as tf
import numpy as np
import os
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0



class Elmo_word_model(object):
    
    def __init__(self, 
                no_of_labels,
                learning_rate, 
                rnn_units, 
                train_elmo = True, 
                output_type = 'state_output', 
                max_sentence_words = 150):

        tf.reset_default_graph()

        # feature extraction network  ------------------------------------------>>
        
        # pass raw string 
        # one hot labels
        sentences             = tf.placeholder(tf.string, (None,), name='sentences')
        self.targets          = tf.placeholder(tf.int32, [None, None], name='labels' )
        sequence_length       = tf.placeholder(tf.int32, (None,), name='sequence_len')

        keep_prob             = tf.placeholder(tf.float32, name='dropout')



        self.placeholders     = {
                                'sentence': sentences, 
                                'labels': self.targets, 
                                'drop': keep_prob, 
                                'sequence_length': sequence_length
                                }

        module                = hub.Module('https://tfhub.dev/google/elmo/2', trainable = train_elmo )
        module_features       = module(dict(tokens=sentences, sequence_len = sequence_length),
                                 signature='tokens', as_dict=True)
        embeddings            = module_features["elmo"]



        # sequence learning network -------------------------------------------------------->
         #bilstm model
        with tf.variable_scope('forward'):
            fr_cell = tf.contrib.rnn.LSTMCell(num_units = rnn_units)
            dropout_fr = tf.contrib.rnn.DropoutWrapper(fr_cell, output_keep_prob = 1. - keep_prob)
            
        with tf.variable_scope('backward'):
            bw_cell = tf.contrib.rnn.LSTMCell(num_units = rnn_units)
            dropout_bw = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob = 1. - keep_prob)
            
        with tf.variable_scope('encoder') as scope:
            model,last_state = tf.nn.bidirectional_dynamic_rnn(dropout_fr,
                                                               dropout_bw,
                                                               inputs = embeddings,
                                                               dtype=tf.float32)

        
        if output_type == 'flat':

            logits = tf.reshape(model[0], (-1, rnn_units * max_sentence_words))
            # dense layer with xavier weights
            fc_layer = tf.get_variable(name='fully_connected',
                                    shape=[rnn_units * max_sentence_words, no_of_labels],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            # bias 
            bias    = tf.get_variable(name='bias',
                                    shape=[no_of_labels],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            #final output 
            self.x_ = tf.add(tf.matmul(logits,fc_layer),bias)

        else:

            logits = tf.concat([last_state[0].c,last_state[1].c],axis=-1)
             # dense layer with xavier weights
            fc_layer = tf.get_variable(name='fully_connected',
                                    shape=[2*rnn_units, no_of_labels],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            # bias 
            bias    = tf.get_variable(name='bias',
                                    shape=[no_of_labels],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            #final output 
            self.x_ = tf.add(tf.matmul(logits,fc_layer),bias)


         #optimization and loss calculation ---------------------------------->>
        
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.x_, labels = tf.cast(self.targets,tf.float32))
        self.loss = tf.reduce_mean(tf.reduce_sum(self.cross_entropy, axis=1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss)
        self.predictions = tf.cast(tf.sigmoid(self.x_) > 0.5, tf.int32)
















        # [batch,1,1024]
        # three dim for lstm input
 

        # sentence level representation

        # sequence learning network ------------------------------------------------------------->>
        
        #bilstm model
        with tf.variable_scope('forward'):
            fr_cell = tf.contrib.rnn.LSTMCell(num_units = rnn_units)
            dropout_fr = tf.contrib.rnn.DropoutWrapper(fr_cell, output_keep_prob = 1. - keep_prob)
            
        with tf.variable_scope('backward'):
            bw_cell = tf.contrib.rnn.LSTMCell(num_units = rnn_units)
            dropout_bw = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob = 1. - keep_prob)
            
        with tf.variable_scope('encoder') as scope:
            model,last_state = tf.nn.bidirectional_dynamic_rnn(dropout_fr,
                                                               dropout_bw,
                                                               inputs = embeddings,
                                                               dtype=tf.float32)
        
        # use lstm final output as logits
        if last_output:
            logits = tf.reshape(model[0], (-1, rnn_units))
            

        # use lstm states as output  
        else:
            
            logits = tf.concat([last_state[0].c,last_state[1].c],axis=-1)


        # dense layer --------------------------------------------------------------------->>
        
        # dense layer with xavier weights
        fc_layer = tf.get_variable(name='fully_connected',
                                   shape=[2*rnn_units, no_of_labels],
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        
        # bias 
        bias    = tf.get_variable(name='bias',
                                   shape=[no_of_labels],
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        
        #final output 
        self.x_ = tf.add(tf.matmul(logits,fc_layer),bias)
        
        
       
        
        #optimization and loss calculation ---------------------------------->>
        
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.x_, labels = tf.cast(self.targets,tf.float32))
        self.loss = tf.reduce_mean(tf.reduce_sum(self.cross_entropy, axis=1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss)
        self.predictions = tf.cast(tf.sigmoid(self.x_) > 0.5, tf.int32)








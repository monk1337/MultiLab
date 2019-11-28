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



class Elmo_model(object):
    
    def __init__(self, no_of_labels, learning_rate, train_elmo = True , model_ = 'base'):

        tf.reset_default_graph()

        # feature extraction network  ------------------------------------------>>
        
        # pass raw string 
        # one hot labels
        sentences             = tf.placeholder(tf.string, (None,), name='sentences')
        self.targets          = tf.placeholder(tf.int32, [None, None], name='labels' )
        keep_prob             = tf.placeholder(tf.float32, name='dropout')



        self.placeholders     = {'sentence': sentences, 'labels': self.targets, 'drop': keep_prob}

        module                = hub.Module('https://tfhub.dev/google/elmo/2', trainable = train_elmo)
        embeddings            = module(dict(text=sentences))

        # output [None, 1024]
        
        if model_ == 'base':
        
            # dense layer with xavier weights ------------------------------------------>>

            fc_layer = tf.get_variable(name='fully_connected',
                                    shape=[1024, no_of_labels],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            # bias 
            bias    = tf.get_variable(name='bias',
                                    shape=[no_of_labels],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            #final output 
            self.logits = tf.add(tf.matmul(embeddings,fc_layer),bias)

        elif model_ == 'two_layer':

            fc_layer_first = tf.get_variable(name='fully_connected_first',
                                    shape=[1024, 256],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

            fc_layer_second = tf.get_variable(name='fully_connected_second',
                                    shape=[256, no_of_labels],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

            # bias 
            bias_first    = tf.get_variable(name='bias_first',
                                    shape=[256],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

            # bias 
            bias_second    = tf.get_variable(name='bias_second',
                                    shape=[no_of_labels],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

            self.logits = tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(embeddings,
                                                fc_layer_first),
                                                bias_first)),
                                                fc_layer_second),
                                                bias_second)


        elif model_ == 'dropout':

            fc_layer_first = tf.get_variable(name='fully_connected_first',
                                    shape=[1024, 512],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            # bias 
            bias_first    = tf.get_variable(name='bias_first',
                                    shape=[512],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

            c_out  = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(embeddings,fc_layer_first),bias_first)), keep_prob)

            fc_layer_second = tf.get_variable(name='fully_connected_second',
                                    shape=[512, 256],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            # bias 
            bias_second    = tf.get_variable(name='bias_second',
                                    shape=[256],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

            c_o  = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(c_out,fc_layer_second),bias_second)), keep_prob)



            fc_layer_third = tf.get_variable(name='fully_connected_third',
                                    shape=[256,128],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            # bias 
            bias_third    = tf.get_variable(name='bias_third',
                                    shape=[128],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

            c_  = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(c_o,fc_layer_third),bias_third)), keep_prob)


            fc_layer_final = tf.get_variable(name='fully_connected_final',
                                    shape=[128,no_of_labels],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            # bias 
            bias_final    = tf.get_variable(name='bias_final',
                                    shape=[no_of_labels],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

            self.logits = tf.nn.relu(tf.add(tf.matmul(c_,
                                                fc_layer_final),
                                                bias_final))

        elif model_ == 'three_layer':

            fc_layer_first = tf.get_variable(name='fully_connected_first',
                                    shape=[1024, 512],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            # bias 
            bias_first    = tf.get_variable(name='bias_first',
                                    shape=[512],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

            c_out  = tf.nn.relu(tf.add(tf.matmul(embeddings,fc_layer_first),bias_first))

            fc_layer_second = tf.get_variable(name='fully_connected_second',
                                    shape=[512, 256],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            # bias 
            bias_second    = tf.get_variable(name='bias_second',
                                    shape=[256],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

            c_o  = tf.nn.relu(tf.add(tf.matmul(c_out,fc_layer_second),bias_second))



            fc_layer_third = tf.get_variable(name='fully_connected_third',
                                    shape=[256,128],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            # bias 
            bias_third    = tf.get_variable(name='bias_third',
                                    shape=[128],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

            c_  = tf.nn.relu(tf.add(tf.matmul(c_o,fc_layer_third),bias_third))


            fc_layer_final = tf.get_variable(name='fully_connected_final',
                                    shape=[128,no_of_labels],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            
            # bias 
            bias_final    = tf.get_variable(name='bias_final',
                                    shape=[no_of_labels],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

            self.logits = tf.nn.relu(tf.add(tf.matmul(c_,
                                                fc_layer_final),
                                                bias_final))





        #optimization and loss calculation ---------------------------------->>
        
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logits, labels = tf.cast(self.targets,tf.float32))
        self.loss = tf.reduce_mean(tf.reduce_sum(self.cross_entropy, axis=1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss)
        self.predictions = tf.cast(tf.sigmoid(self.logits) > 0.5, tf.int32)

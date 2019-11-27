import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Bilstm_model(object):
    
    def __init__(
                 self, 
                 vocab_size,
                 rnn_units,
                 word_embedding_dim,
                 no_of_labels,
                 learning_rate               = 0.001, 
                 pretrained_embedding_matrix = None,
                 train_embedding             = True,
                 last_output                 = False):
        

        tf.reset_default_graph()
        
        
        # placeholder for input------------------------------------------------------->>
        
        sentences             = tf.placeholder(tf.int32, [None,None], name='sentences')
        self.targets          = tf.placeholder(tf.int32, [None, None], name='labels'  )
        keep_prob             = tf.placeholder(tf.float32, name='dropout')
        
        self.placeholders     = {
                                  'sentence': sentences, 
                                 'labels': self.targets, 
                                 'dropout': keep_prob
                                 }
        
        
        
        #embeddings  ------------------------------------------------------------------->>
        
        # initialize trained embedding 
        
        if pretrained_embedding_matrix is None:

            # use random embedding
            word_embedding            = tf.get_variable(name='word_embedding_',
                                         shape=[vocab_size, word_embedding_dim],
                                         dtype=tf.float32,
                                         initializer = tf.contrib.layers.xavier_initializer())
            
        else:
            #use pretrained_ embedding
            word_embedding             = tf.get_variable(name="word_embedding_", 
                                         shape=[pretrained_embedding_matrix.shape[0],pretrained_embedding_matrix.shape[1]],
                                         initializer = tf.constant_initializer(np.array(pretrained_embedding_matrix)), 
                                         trainable = train_embedding ,dtype=tf.float32)
            
        
        
        
        # lookup and sequence count ----------------------------------------------------------->>
        # embedding lookup
        embedding_lookup = tf.nn.embedding_lookup(word_embedding, sentences)
        
        # ignore padding during sequence unfolding in lstm
        sequence_leng = tf.count_nonzero(sentences,axis=-1)
        
        
        
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
                                                               inputs=embedding_lookup,
                                                               sequence_length=sequence_leng,
                                                               dtype=tf.float32)
        
        # use lstm final output as logits
        if last_output:
            logits = tf.transpose(tf.concat(model, 2), [1, 0, 2])

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
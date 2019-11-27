import tensorflow as tf
import numpy as np
import tensorflow_hub as hub




class Elmo_model(object):
    
    def __init__(self, no_of_labels):

        tf.reset_default_graph()

        # feature extraction network  ------------------------------------------>>
        
        # pass raw string 
        # one hot labels
        sentences             = tf.placeholder(tf.string, (None,), name='sentences')
        self.targets          = tf.placeholder(tf.int32, [None, None], name='labels' )


        self.placeholders     = {'sentence': sentences, 'labels': self.targets}

        module                = hub.Module('https://tfhub.dev/google/elmo/2', trainable = True)
        embeddings            = module(dict(text=sentences))

        # output [None, 1024]
        
        
        
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

        
        
        #optimization and loss calculation ---------------------------------->>
        
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logits, labels = tf.cast(self.targets,tf.float32))
        self.loss = tf.reduce_mean(tf.reduce_sum(self.cross_entropy, axis=1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self.predictions = tf.cast(tf.sigmoid(self.logits) > 0.5, tf.int32)

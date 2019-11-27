import tensorflow as tf
import numpy as np
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
    
    def __init__(self, no_of_labels, learning_rate):

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
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss)
        self.predictions = tf.cast(tf.sigmoid(self.logits) > 0.5, tf.int32)

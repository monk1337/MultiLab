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
    
    def __init__(self, no_of_labels, learning_rate, model_ = 'base'):

        tf.reset_default_graph()

        # feature extraction network  ------------------------------------------>>
        
        # pass raw string 
        # one hot labels
        sentences             = tf.placeholder(tf.string, (None,), name='sentences')
        self.targets          = tf.placeholder(tf.int32, [None, None], name='labels' )
        keep_prob             = tf.placeholder(tf.float32, name='dropout')



        self.placeholders     = {'sentence': sentences, 'labels': self.targets, 'drop': keep_prob}

        module                = hub.Module('https://tfhub.dev/google/elmo/2', trainable = True)
        embeddings            = module(dict(text=sentences))

# let's take one dataset for example
from multilab.datasets import reuter
sentences, labels = reuter()

from multilab.preprocess import Text_preprocessing
text_preprocessing = Text_preprocessing()

dataframe = text_preprocessing.labels_to_dataframe(sentences,labels)
preprocessded_dataset = text_preprocessing.initial_preprocess(dataframe, chunk_value = 25)
dataset, frequency_list = text_preprocessing.keep_labels(preprocessded_dataset,keep_ratio=0.10)
slice_dataset = text_preprocessing.dataset_slice(dataset,ratio=0.25)

import numpy as np
all_sente = list(slice_dataset['text'])
all_label = np.array(slice_dataset.drop('text', 1))

X_train, X_test, y_train, y_test = text_preprocessing.split_dataset(all_sente, all_label)

from multilab.models import Elmo_Word_Model

config = {
                         'no_of_labels'               :  y_train.shape[1],
                         'learning_rate'              : 0.001,
                         'rnn_units'                  : 100,
                         'epoch'                      : 20,
                         'batch_size'                 : 128,
                         'dropout'                    : 0.0,
                         'output_type'                : 'state_output', # or flat
                         'train_elmo'                 : True,
                         'result_path'                : '.',
                        }


el_m = Elmo_Word_Model(X_train, y_train, X_test,  y_test, config)
el_m.train()

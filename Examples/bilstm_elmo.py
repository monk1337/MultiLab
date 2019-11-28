# let's take one dataset for example
from multilab.datasets import reuter
from multilab.models import BiLstm_Elmo
sentences, labels = reuter()

from multilab.preprocess import Text_preprocessing
text_preprocessing = Text_preprocessing()

dataframe = text_preprocessing.labels_to_dataframe(sentences,labels)
preprocessded_dataset = text_preprocessing.initial_preprocess(dataframe, chunk_value = 150)
dataset, frequency_list = text_preprocessing.keep_labels(preprocessded_dataset,keep_ratio=0.10)
slice_dataset = text_preprocessing.dataset_slice(dataset,ratio=0.5)

import numpy as np
all_sente = list(slice_dataset['text'])
all_label = np.array(slice_dataset.drop('text', 1))

X_train, X_test, y_train, y_test = text_preprocessing.split_dataset(all_sente, all_label)

cofig  =       {
                         'no_of_labels'               : len(y_train[0]),
                         'learning_rate'              : 0.001,
                         'rnn_units'                  : 150,
                         'last_output'                : False,
                         'epoch'                      : 10,
                         'batch_size'                 : 128,
                         'dropout'                    : 0.2,
                         'model_type'                 : 'base',
                         'train_elmo'                 : True,
                         'result_path'                : '.'
                        }

blelmo = BiLstm_Elmo(X_train, y_train, X_test,  y_test, cofig)
print(blelmo.train())
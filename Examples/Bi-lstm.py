# let's take one dataset for example
from multilab.datasets import reuter
sentences, labels = reuter()


# preprocessing

from multilab.preprocess import Text_preprocessing
tp = Text_preprocessing()

dataframe = tp.labels_to_dataframe(sentences,labels)


# chunking and vocab
preprocessded_dataset   = tp.initial_preprocess(dataframe, chunk_value = 150)
dataset, frequency_list = tp.keep_labels(preprocessded_dataset,keep_ratio = 0.25)


# slice dataset
slice_dataset                                    = tp.dataset_slice(dataset,ratio=0.50)
sorted_long, freq_num, word_to_int, int_to_word  = tp.vocab_freq(slice_dataset)
top_words, freq_lis, w2i, i2w                    = tp.vocab_freq(slice_dataset)
all_sentence_s, all_label_s, vocab_dict          = tp.encoder(slice_dataset, w2i)
X_train, X_test, y_train, y_test                 = tp.split_dataset(all_sentence_s, all_label_s)



# model
from multilab.models import Bilstm

config = {
                         'vocab_size'                 : len(vocab_dict),
                         'no_of_labels'               : len(frequency_list),
                         'rnn_units'                  : 256, 
                         'word_embedding_dim'         : 300, 
                         'learning_rate'              : 0.001, 
                         'pretrained_embedding_matrix': None,
                         'dropout'                    : 0.2,
                         'epoch'                      : 200,
                         'batch_size'                 : 128,
                         'result_path'                : '.',
                         'last_output'                : False,
                         'train_embedding'            : True
                        }


bl = Bilstm(X_train, y_train, X_test,  y_test, config)

print(bl.train())

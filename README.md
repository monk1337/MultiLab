<h1 align="center"> MultiLab</h1>
<p align="center">A framework for Multi-label classification, Classical machine learning models to state of art deep learning models for multi label classification, preprocessing Multi-label datasets and load benchmark multi-label datasets, metrices for accuracy calculation </p>


You can preprocess multilabel dataset simply as follows:
```python

from multilab.preprocess import Text_preprocessing

tp = Text_preprocessing()
preprocessded_dataset = tp.initial_preprocess(dataframe, chunk_value = 5)

```

Loading Models


```python
from multilab.models import BinaryRe

Bm = BinaryRe(X_train, y_train, X_test,y_test)
print(Bm.train())
```

output :

```python

{'accuracy': 0.4074074074074074, 'f1_score': 0.4395604395604396}
```

Loading Bilstm model

```python
from multilab.models import Bilstm


config = {
            'vocab_size'                 : 7000,
            'no_of_labels'               : 9,
            'rnn_units'                  : 256, 
            'word_embedding_dim'         : 300, 
            'learning_rate'              : 0.001, 
            'pretrained_embedding_matrix': None,
            'dropout'                    : 0.2,
            'epoch'                      : 5,
            'batch_size'                 : 128,
            'result_path'                : '/Users/aaditya/Desktop',
            'last_output'                : False,
            'train_embedding'            : True
        }


bl = Bilstm(X_train, y_train, X_test,  y_test, config)
bl.train()
```

output

```python
validation_acc {'subset_accuracy': 0.45166666666666666, 'hamming_score': 0.4601111111111112, 'hamming_loss': 0.1285185185185185, 'micro_ac': 0.4490395710185522, 'weight_ac': 0.2830056188426279, 'epoch': 0}
```




Loading Elmo model

```python
from multilab.models import Elmo

config = {
                         'no_of_labels'               : 9,
                         'learning_rate'              : 0.001,
                         'epoch'                      : 5,
                         'batch_size'                 : 128,
                         'model_type'                 : 'base',
                         'result_path'                : '.'
                        }


elmo_model = Elmo(X_train, y_train, X_test,  y_test, config)
elmo_model.train()

```

output

```python
validation_acc {'subset_accuracy': 0.5966666666666666, 'hamming_score': 0.6, 'hamming_loss': 0.05907407407407408, 'micro_ac': 0.6943641132818982, 'weight_ac': 0.5731481624223015, 'epoch': 0}
```



##### adding more models, work in progress..

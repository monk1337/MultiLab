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

# adding more models work in progress..

import nltk

def reuter():

    nltk.download("reuters")
    from nltk.corpus import reuters
    
    documents = reuters.fileids()
    train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                                documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                               documents))
    X_train = [(reuters.raw(doc_id)) for doc_id in train_docs_id]
    X_test = [(reuters.raw(doc_id)) for doc_id in test_docs_id]

    y_train = [reuters.categories(doc_id)
                                 for doc_id in train_docs_id]
    y_test = [reuters.categories(doc_id)
                            for doc_id in test_docs_id]

    all_sentences     =    X_train +  X_test
    all_label         =    y_train +  y_test

    return all_sentences, all_label
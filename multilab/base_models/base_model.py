import numpy as np
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
from skmultilearn.problem_transform import LabelPowerset
from sklearn.linear_model import LogisticRegression

from skmultilearn.problem_transform import ClassifierChain

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB


class Base_models(object):
    
    def __init__(self, x_data, y_data, x_test, y_test):
        
        self.x_data = x_data
        self.y_data = y_data
        self.x_test = x_test
        self.y_test = y_test
        
    def BinaryRe(self):
        classifier = BinaryRelevance(GaussianNB())
        classifier.fit(self.x_data,  self.y_data)
        predictions = classifier.predict(self.x_test)

        return {
                 'accuracy': accuracy_score(self.y_test,predictions), 
                 'f1_score': f1_score(self.y_test, predictions, average='micro') 
               }
    
    
    def powerset(self):
        
        classifier = LabelPowerset(LogisticRegression())
        classifier.fit(self.x_data,  self.y_data)


        predictions = classifier.predict(self.x_test)


        return {
                 'accuracy': accuracy_score(self.y_test,predictions), 
                'f1_score': f1_score(self.y_test, predictions, average='micro') 
               }
    
    def mlknn(self):
        
        classifier_new = MLkNN(k=10)

        x_train = lil_matrix(self.x_data).toarray()
        y_train = lil_matrix(self.y_data).toarray()
        x_test = lil_matrix(self.x_test).toarray()

        classifier_new.fit(x_train, y_train)

        # predict
        predictions = classifier_new.predict(x_test)
        
        return {
                 'accuracy': accuracy_score(self.y_test,predictions), 
                'f1_score': f1_score(self.y_test, predictions, average='micro') 
               }
    
    
    
    def classfier_chain(self):
        
        classifier = ClassifierChain(LogisticRegression())
        classifier.fit(self.x_data,  self.y_data)
        
        predictions = classifier.predict(self.x_test)
        
        
        return {
                 'accuracy': accuracy_score(self.y_test,predictions), 
                'f1_score': f1_score(self.y_test, predictions, average='micro') 
               }
        
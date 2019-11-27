
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import ClassifierChain
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class Classfierchains(object):
    
    def __init__(self, x_data, y_data, x_test, y_test):
        
        self.x_data = x_data
        self.y_data = y_data
        self.x_test = x_test
        self.y_test = y_test

    def train(self):
        classifier = ClassifierChain(LogisticRegression())
        classifier.fit(self.x_data,  self.y_data)
        
        predictions = classifier.predict(self.x_test)
        
        
        return {
                 'accuracy': accuracy_score(self.y_test,predictions), 
                'f1_score': f1_score(self.y_test, predictions, average='micro') 
               }


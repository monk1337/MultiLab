from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class MlKnn(object):
    
    def __init__(self, x_data, y_data, x_test, y_test):
        
        self.x_data = x_data
        self.y_data = y_data
        self.x_test = x_test
        self.y_test = y_test


    def train(self):
        
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
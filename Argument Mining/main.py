import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
# from sklearn import svm
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
# from sklearn.preprocessing import Binarizer

def main():
    ## YOUR CODE HERE
    import pandas as pd
    
    
     # loading the dataset
    df= [x.split('\t') for x in open('./data/train-bio.csv').read().split('\n')]
    test_df = [x.split('\t') for x in open('./data/test-bio.csv').read().split('\n')]
    train = pd.DataFrame (df, columns = ['Token', 'Label'])
    test = pd.DataFrame (test_df, columns = ['Token', 'Label'])
    
    # Cleaning the dataset
    train = train.dropna()
    test = test.dropna()
    
    
    # encoding the target labels
    le = LabelEncoder()
    train['Label'] = le.fit_transform(train['Label'])
    test['Label'] = le.fit_transform(test['Label'])

    #Creating the lists for x and y for the model
    x_train = list(train['Token'])
    y_train = list(train['Label'])
    x_test = list(test['Token'])
    y_test = list(test['Label'])
       
        
    TfIdfvectorizer = TfidfVectorizer(min_df=2, max_df=0.75, ngram_range=(1,2))
    X_train_tr = TfIdfvectorizer.fit_transform(x_train)
    X_test_tr = TfIdfvectorizer.transform(x_test)
    
    
    
    
    # svc=SVC(probability=True, kernel='linear')

    clf= AdaBoostClassifier(n_estimators=10000,learning_rate=0.75)
 
    clf.fit(X_train_tr, y_train)

    # create predictions
    y_pred = clf.predict(X_test_tr)
    Correct_pred=le.inverse_transform(y_pred)
        
    #f-1 score
    score = f1_score(y_test, y_pred, average='macro')
    score_average = f1_score(y_test, y_pred, average='weighted')
    print('F-1 score : {}'.format(np.round(score,4)))
    print('F-1 score Average: {}'.format(np.round(score_average,4)))
    
    prediction=pd.DataFrame()
    prediction['Token']=x_test
    prediction['Predictions']=Correct_pred
    prediction.to_csv('predictions.csv' ,sep="\t" ,header=None,index=False)
    #print(len(pd))
    pass
    

if __name__ == '__main__':
    main()

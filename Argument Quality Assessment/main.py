import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer 

from sklearn import preprocessing
# from nltk.tokenize import word_tokenize
# from nltk import pos_tag
# from nltk.tokenize import sent_tokenize
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

def text_preprocessing(df_col):
    #object of WordNetLemmatizer
        lm = WordNetLemmatizer()
        corpus = []
        for item in df_col:
            new_item = re.sub('[^a-zA-Z]',' ',str(item))
            new_item = new_item.lower()
            new_item = new_item.split()
            new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
            corpus.append(' '.join(str(x) for x in new_item))
        return corpus



def main():

    

    df = pd.read_csv('./data/train-test-split.csv', delimiter=';')
    df['ID'] = df['ID'].str.replace('essay', '')
    df_data = pd.read_json('./data/essay-corpus.json')
    df_data = df_data.filter(['id','text','confirmation_bias'], axis=1)

    df_data.columns = ['ID', 'Text', 'confirmation_bias']
    df = df.astype({"ID": int}, errors='raise')
    df_merged = df_data.merge(df, on='ID')
    df_train = df_merged[df_merged.SET.isin(['TRAIN'])]
    df_test = df_merged[df_merged.SET.isin(['TEST'])]


    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    lb = preprocessing.LabelEncoder()
    df_train['confirmation_bias'] = lb.fit_transform(df_train['confirmation_bias'])
    df_test['confirmation_bias'] = lb.fit_transform(df_test['confirmation_bias'])

    x_train = list(df_train['Text'])
    y_train = list(df_train['confirmation_bias'])
    x_test = list(df_test['Text'])
    y_test = list(df_test['confirmation_bias'])

    cv = CountVectorizer()
    tfid = TfidfVectorizer(min_df=2, max_df=0.75, ngram_range=(1,1))
    combined_features = FeatureUnion([("countVec", cv), ("tfid", tfid)])

    X_train_cv = combined_features.fit_transform(x_train)
    X_test_cv = combined_features.transform(x_test)
    # X_train_cv = cv.fit_transform(x_train)
    # X_test_cv = cv.transform(x_test)

    

    # Tried using different classifiers, but SVM gave the best result


    # NaiveBayes classifier
    # model = GaussianNB()
    # model.fit(X_train_cv,y_train)    
    # y_pred = model.predict(X_test_cv)
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    # AdaBoostClassifier classifier
    #clf= AdaBoostClassifier(n_estimators=5000,learning_rate=1)
    # kf=KFold(n_splits=10)
    # cv_score=cross_val_score(clf,X_train_cv,y_train,cv=kf)
    # print('\n'+'cross validation score :',cv_score)
    
    # RandomForestClassifier

    # clean_text = text_preprocessing(df_train['text'])
    # X = combined_features.fit_transform(clean_text)
    # y = df_train.confirmation_bias
    # gridSearch_bestParameters = {'bootstrap': False,'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 500}

    # rndFC = RandomForestClassifier(max_features=gridSearch_bestParameters['max_features'], max_depth=gridSearch_bestParameters['max_depth'], n_estimators=gridSearch_bestParameters['n_estimators'], min_samples_split=gridSearch_bestParameters['min_samples_split'], min_samples_leaf=gridSearch_bestParameters['min_samples_leaf'], bootstrap=gridSearch_bestParameters['bootstrap'])
    
    # rndFC.fit(X,y)
    # X_test,y_test = df_test.text,df_test.confirmation_bias
    # test_corpus = text_preprocessing(X_test)
    # testdata = combined_features.transform(test_corpus)
    # predictions = rndFC.predict(testdata)


    # We decided to SVC as we got the best result with it
    
    # Uncomment this block if the results from grid search need to be used directly
    param_grid = {'C':[0.1, 1, 10, 100, 1000],'gamma':[1, 0.1, 0.01, 0.001, 0.0001], 'kernel':['linear','rbf']}
    clf=SVC(C=10, gamma=0.0001, verbose=True, kernel="rbf")
    grid = GridSearchCV(clf,param_grid,refit = True, verbose=3, cv = 5, return_train_score=True, n_jobs=-1)
    grid.fit(X_train_cv, y_train)
    print(grid.best_params_)
    print(grid.best_estimator_)
    y_pred = grid.predict(X_test_cv)
    Correct_pred=lb.inverse_transform(y_pred)

    
    # Finding the best parameters using gridsearch and using them for the model. We got C = 10, gamma = 0.0001, kernel = "rbf"(default), the parameters can be printed the code below
    

    # Comment out this code if results from the grid search need to be used directly
    # clf=SVC(C=10, gamma=0.0001, verbose=True, kernel="rbf")
    # clf.fit(X_train_cv, y_train)
    # y_pred = clf.predict(X_test_cv)
    # Correct_pred=lb.inverse_transform(y_pred)

        
    score = f1_score(y_test, y_pred, average='macro')
    score_average = f1_score(y_test, y_pred, average='weighted')
    print('\n'+'F-1 score : {}'.format(np.round(score,4)))
    print('F-1 score : {}'.format(np.round(score_average,4)))

    prediction=pd.DataFrame()
    prediction['id']=df_test.ID
    prediction['confirmation_bias']=Correct_pred
    prediction.to_json('predictions.json', orient='records')

    

    # Train the model using the training sets
    


if __name__ == '__main__':
    main()

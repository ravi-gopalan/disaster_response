# import libraries
import sys
import re
import nltk
nltk.download(['stopwords','punkt','wordnet'])
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import joblib
import pickle

def load_data(database_filepath):
        # load data from database
    qstring = 'sqlite:///'+database_filepath
    engine = create_engine(qstring)
    
    connection = engine.connect()
    df = pd.read_sql_table('messages', connection)
    X = df['message'].values
    Y = df.drop(columns=['id','message','genre'],axis=1).values
    category_names = df.columns[3:]

    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        lemmed = lemmatizer.lemmatize(tok, pos='v')
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = re.sub(r"[^a-zA-Z0-9]"," ",lemmed.lower()).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def display_results(y_test, y_pred, category_names):
    list_of_reports = []
    
    for col in range(len(category_names)):
        report_metric = {}
        y1 = y_test.T[col]
        y2 = y_pred.T[col]

        report_metric['precision'] = precision_score(y1,y2,'micro' )
        report_metric['recall'] = recall_score(y1,y2,'micro')
        report_metric['f1'] = f1_score(y1,y2,'micro')
        report_metric['accuracy'] = accuracy_score(y1,y2)
        report_metric['kappa'] = cohen_kappa_score(y1,y2)
        list_of_reports.append(report_metric)
    performance_metrics = pd.DataFrame(list_of_reports)
    return performance_metrics

def build_model(X_train, Y_train):
    # build pipeline
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),\
                         ('tfidf',TfidfTransformer()),\
                         ('best', TruncatedSVD()),\
                         ('clf',MultiOutputClassifier(\
                                                      RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42, n_jobs=-1)))\
                          ])
    
    pipeline.fit(X_train, Y_train)
    joblib.dump(pipeline.get_params,'models/pipeline.pkl')
    print('pipeline pickled')
    
    parameters = {'clf__estimator__n_estimators': [100, 150],\
                  'clf__estimator__min_samples_split': [3,4,5]\
                  }

    
    gscv = GridSearchCV(estimator=pipeline,\
                        param_grid=parameters,\
                        scoring=['accuracy','precision_micro','recall_micro','f1_micro'],\
                        refit='precision_micro',\
                        cv=3\
                        )
    
    gscv.fit(X_train, Y_train)
    print(gscv.best_params_)
    
    return gscv.best_estimator_


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    
    performance_report = display_results(Y_test, Y_pred, category_names)
    print(performance_report)    
    return    


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
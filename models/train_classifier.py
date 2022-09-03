# import libraries
import sys
import re
import pickle
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,label_ranking_average_precision_score ,confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



nltk.download(['wordnet', 'punkt', 'stopwords','omw-1.4'])


def load_data(database_filepath):
    """load data from database

    Args:
        database_filepath (string): path of database

    Returns:
        X: feature
        y: target feature 
        categories: columns names
    """

    from unicodedata import category
    engine = create_engine('sqlite:///DisasterMessages.db')
    df = pd.read_sql_table('disastertab',engine)
    df.dropna(inplace=True)
    X=df['message']
    Y = df[df.columns[4:]]
    categories=Y.columns
    return X,Y,categories


def tokenize(text):
    """toknizing text messages

    Args:
        text (string): [string that will be tokenized]

    Returns:
        [clean_tokens]: [string after being tooknize]
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """building the pipeline and building ML model

    Returns:
        Model
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200)))
    ])
    
    parameters = {
    'clf__estimator__min_samples_leaf': [1,10],
    'clf__estimator__max_features': ['auto','log2'],
    }


    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluating ML model

    Args:
        model (ML model): the model
        X_test ():  X test sets
        Y_test (): Y test sets
        category_names (string): categories
    """
    y_pred=model.predict(X_test)
    result=precision_recall_fscore_support(Y_test, y_pred)
    for i, col in enumerate(Y_test.columns.values):
        accu=accuracy_score(Y_test.loc[:,col],y_pred[:,i])
        score = ('{}\n Accuracy:  {:.4f}   % Precision: {:.4f}   % Recall {:.4f} '.format(
                  col,accu,result[0][i],result[1][i]))
        print(score)
    avgerage_precision = label_ranking_average_precision_score(Y_test, y_pred)
    average_score= ('average precision: {}'.format(avgerage_precision))
    print(average_score)


def save_model(model, model_filepath):
    """ saving the model as pkl file

    Args:
        model (ML model): the ML model
        model_filepath (string): place to save the model
    """
    with open('classifer.pkl', 'wb') as f:
        pickle.dump(model, f)


def main():
    #if len(sys.argv) == 3:
        database_filepath="DisasterMessages.db"
        model_filepath="model.pkl"
        #= sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

   
    #else:
        #print('Please provide the filepath of the disaster messages database '\
         #     'as the first argument and the filepath of the pickle file to '\
          #    'save the model to as the second argument. \n\nExample: python '\
           #   'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
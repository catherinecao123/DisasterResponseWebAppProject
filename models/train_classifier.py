import sys
import os
import re
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report

import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the data
def load_data(database_filepath):
    """
    INPUT:

    OUTPUT:
    X: 
    y: 
    category_names:
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).split('.')[0]
    df = pd.read_sql_table(table_name, con=engine)

    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns

    return X, y, category_names

def tokenize(text):
    """
    INPUT:

    OUTPUT: 
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")

    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='v').lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """
    """ 

    def sentence_length(self, text):
        return len(word_tokenize(text))

    def fit(self, X, y=None):
        return self

    def transform(self, X):        
        X_tagged = pd.Series(X).apply(self.sentence_length)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    """
    # instantiate the pipeline
    model = Pipeline([
    ('features',FeatureUnion([
        ('text_pipeline',Pipeline([
            ('vect',CountVectorizer(tokenizer=tokenize)),
            ('tfidf',TfidfTransformer())
            ])),
        ('text_length', TextLengthExtractor())
        ])),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])

    return model

def evaluate_model(model, X_test, y_test, category_names):
    """    
    INPUT:
        model: 
        X_test: 
        y_test: 
    OUTPUT
        Classification report and accuracy score
    """
    # predict
    y_pred = model.predict(X_test)

    # classification report
    print(classification_report(y_test.values, y_pred, target_names=category_names))

    # accuracy score
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy score is {:.3f}'.format(accuracy))

def save_model(model, model_filepath):
    """ This function saves the pipeline to local disk """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names= load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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
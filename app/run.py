# import libraries
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import PorterStemmer
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

import sys
import os
# from models folder import functions tokenize, TextLengthExtractor
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "models"))
from train_classifier import tokenize, TextLengthExtractor

app = Flask(__name__)
#load data from sql table
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
#load model
model = joblib.load("../models/classifier.pkl")

@app.route('/')
@app.route('/index')

def index():
    # prepare data needed for viz
    #data for graph 1: bar chart of the number of counts in each type of genre
    GenreCounts = df.groupby('genre').count()['message']
    GenreNames = list(GenreCounts.index)

    # data for graph 2: Genre distribution in the 36 categories
    CatesLabels = df[df.columns[4:]].sum().sort_values(ascending=False).index
    df_genre = df.groupby('genre')[CatesLabels].sum().reset_index()

    # graphs
    graphs = [{
        #graph1
            'data': [
                Bar(
                    x=GenreNames,
                    y=GenreCounts
                    )],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "# of counts"
                },
                'xaxis': {
                    'title': "genre"
                }
            }
        },
        #graph 2
        {
            'data': [
               Bar(
                x=CatesLabels,
                y=df_genre.iloc[0],
                name='Direct'
                ),
                Bar(
                    x=CatesLabels,
                    y=df_genre.iloc[1],
                    name='News'
                ),
                Bar(
                    x=CatesLabels,
                    y=df_genre.iloc[2],
                    name='Social'
                )
            ],
            'layout': {
                'title': 'Number of Message Types in each Categories',
                'yaxis': {
                    'title': "# of count"
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle': -40
                },
                'barmode': 'stack'
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
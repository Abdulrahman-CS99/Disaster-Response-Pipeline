import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterMessages.db')
df = pd.read_sql_table('disastertab', engine)

# load model
model = joblib.load("../models/classifer.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    #----------------------------vis 2 ------------------------------------------------
    df_1=df.drop(columns={'id','message','original','genre'})
    columns=[]
    number_of_occureance=[]
    for column in df_1.columns:
        columns.append(column)
        number_of_occureance.append(df_1[column].sum())
    vis_2=pd.DataFrame({'Feature':columns,'number_of_occureance':number_of_occureance})
    vis_2=vis_2.sort_values(by='number_of_occureance',ascending=False).head(10)
        #----------------------------vis 3 ------------------------------------------------
    df_2=df[df['request']==1]
    vis_3= df_2.groupby('genre').count()['message']

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=vis_2['Feature'],
                    y=vis_2['number_of_occureance'],
                )
            ],

            'layout': {
                'title': 'Most feature inside dataset',
                'yaxis': {
                    'title': "Feature"
                },
                'xaxis': {
                    'title': "number of occureance "
                }
            }
        },
     {
            'data': [
                Bar(
                    x=genre_names,
                    y=vis_3
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres in Reqeust Feature',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
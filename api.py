import flask
import json
import sys
import io
import os
from flask import jsonify

import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece
import time
import numpy as np
from annoy import AnnoyIndex
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch_dsl import Search
import numpy as np
import unicodedata
import csv 
from os.path import basename
import unicodedata
from to_sentences import *
from requests_aws4auth import AWS4Auth
import sys 
from search import *

app = flask.Flask(__name__)


with open("config.json") as f:
    config = json.load(f)

def generate_embeddings (messages_in):
    return session.run(embedded_text, feed_dict={text_input: messages_in})

awsauth = AWS4Auth(
        config.get("AWS_ACCESS_KEY"),
        config.get("AWS_SECRET_KEY"),
        config.get("ELASTICSEARCH_REGION"),
        'es'
    )

es = Elasticsearch(
        hosts=[{'host': config.get("ELASTICSEARCH_URL"), 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        timeout=30,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )


use_module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/1"
g = tf.Graph()
with g.as_default():
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed_module = hub.Module(use_module_url)
    embedded_text = embed_module(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

session = tf.Session(graph=g)
session.run(init_op)

ES_INDEX_FULL_TEXT = config.get("ES_INDEX_FULL_TEXT")
ES_INDEX_CHUNK = config.get("ES_INDEX_CHUNK")
vector_dims = 512
vector_index = AnnoyIndex(vector_dims, 'angular')
annoy_fn = config.get("ANNOY_FN")
vector_index.load(annoy_fn) # super fast, will just mmap the file


with open(config.get("IDX_NAME"), 'r') as f:
    idx_name = json.load(f)
with open(config.get("NAME_IDX"), 'r') as f:
    name_idx = json.load(f)
vec_cnt = vector_index.get_n_items()

searcher = QzUSESearchFactory(vector_index, idx_name, name_idx, es, ES_INDEX_FULL_TEXT, ES_INDEX_CHUNK, generate_embeddings)

def resource(id):
    query = {
        "query": {
            "match": {
                "_id": id
            }
        }
    }
    res = es.search(index="hofeller-files", body=query)
    return res

@app.route("/")
def home():
    return "Hello World"

@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        id = flask.request.args.get('id', default="")
    recs_results= []
    search = searcher.query_by_doc_text(id, k=50)
    recomendations = search.show(show_seed_docs=False)
    for rec in recomendations:
        recData = resource(rec)
        if(recData):
            recs_results.append(json.dumps(recData))   			
    return jsonify({
            'recs' : recs_results 
        }), 200

@app.route("/predict-tester", methods=["POST"])
def predict_test():
    if flask.request.method == "POST":
        data = flask.request.args['data']
        print(data)
    return jsonify({
            'data' : data
        }), 200

if __name__ == "__main__":
	# load the function used to classify input images in a *separate*
	# thread than the one used for main classification
	print("* Starting web service...")
	app.run(host="0.0.0.0", port=80)

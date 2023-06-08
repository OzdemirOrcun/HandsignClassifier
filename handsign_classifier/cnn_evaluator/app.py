import json
import os
import asyncio
from pyexpat import model
import aiohttp
import pickle
from datetime import datetime, timedelta
import numpy as np

from flask import Flask, request,jsonify
from urllib.parse import urlencode
from cnn_evaluator import CNNEvaluator
from data_processor.data_processor import DataProcessor

import glob

app = Flask(__name__)

only_safe_operations = os.getenv("only_safe_operations", "false")[0] == "t"

port = os.getenv("cnn_evaluator_port", 8011)


folder = "./output/"
if not os.path.exists(folder):
    os.makedirs(folder)

cnn_evaluator = CNNEvaluator()
data_processor = DataProcessor()

def get_file_name():
    current_datetime = datetime.now()

    day = current_datetime.day
    month = current_datetime.month
    year = current_datetime.year

    file_path = f"./output/cnn_{day}_{month}_{year}.pkl"
    return file_path

def get_f1_score(CNNEvaluator):

    test_X,test_y = data_processor.load_test_data()
    reshaped_test_X = data_processor.reshape_data(test_X)

    file_name = get_file_name()
    with open(file_name, 'rb') as file:
        model = pickle.load(file)

    f1_score = CNNEvaluator.calculate_f1_score(model,reshaped_test_X,test_y)
    return f1_score

def evaluate_model(CNNEvaluator):

    test_X,test_y = data_processor.load_test_data()
    reshaped_test_X = data_processor.reshape_data(test_X)

    file_name = get_file_name()
    with open(file_name, 'rb') as file:
        model = pickle.load(file)

    loss, acc = CNNEvaluator.evaluate_model(model,reshaped_test_X,test_y)

    return [loss, acc]

@app.route("/cnnevaluator", methods=["POST"])
def cnn_evaluator_endpoint():
    action = request.json["action"]

    if request.method == "POST":
      
        if action == "f1":
            f1_score = get_f1_score(cnn_evaluator)
            return jsonify({'f1_score': str(f1_score)})
        
        elif action == "eval":
            lst = evaluate_model(cnn_evaluator)
            return jsonify({'loss and accuracy': lst})
            
        else:
            return jsonify({'summary': "proper action argument is required"})  
    else:
        return jsonify({'summary': 'pass'})


def build_url(hostname, specified_port, path, url_vars):
    return f"http://{hostname}:{specified_port}/{path}?{urlencode(url_vars)}"


def check_key(dict_, key):
    """ Checks dict key, if not exits return None """
    try:
        return dict_[key]
    except KeyError as e:
        print(e)
        return None


async def async_get_request(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=None) as resp:
            return await resp.json(content_type=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
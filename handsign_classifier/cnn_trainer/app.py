import json
import os
import asyncio
import aiohttp
import pickle
from datetime import datetime, timedelta
import pickle

from flask import Flask, request,jsonify
from urllib.parse import urlencode
from cnn_trainer import CNNTrainer
from data_processor.data_processor import DataProcessor
import glob

app = Flask(__name__)

only_safe_operations = os.getenv("only_safe_operations", "false")[0] == "t"

port = os.getenv("cnn_trainer_port", 8012)


folder = "./output/"
if not os.path.exists(folder):
    os.makedirs(folder)

cnn_trainer = CNNTrainer()
data_processor = DataProcessor()

def get_file_name():
    """Returns the created cnn model's filename 

    Returns
    -------
    str
        CNN model filename with day,month,and year.
    """  
    current_datetime = datetime.now()

    day = current_datetime.day
    month = current_datetime.month
    year = current_datetime.year

    file_path = f"./output/cnn_{day}_{month}_{year}.pkl"
    return file_path

def compile_model(CNNTrainer):
    """Compiles and Saves CNN Model

    Parameters
    ----------
    CNNTrainer : object
        CNNTrainer is responsible of compiling and fitting the CNN Model with given training data.
    Returns
    -------
    str
        signal
    """    
    with open(get_file_name(), 'rb') as file:
        model = pickle.load(file)
    model = CNNTrainer.compile_model(model)

    with open(get_file_name(), "wb") as file:
        pickle.dump(model, file)

    return "1"


def fit_model(CNNTrainer):
    """Fits and saves CNN Model

    Parameters
    ----------
    CNNTrainer : object
        CNNTrainer is responsible of compiling and fitting the CNN Model with given training data.

    Returns
    -------
    str
        signal
    """    
    file_name = get_file_name()
    with open(file_name, 'rb') as file:
        model = pickle.load(file)

    train_X,train_y = data_processor.load_train_data()
    test_X,test_y = data_processor.load_test_data()

    reshaped_train_X = data_processor.reshape_data(train_X)
    reshaped_test_X = data_processor.reshape_data(test_X)

    history = CNNTrainer.fit_model(model,reshaped_train_X,train_y,reshaped_test_X,test_y)

    with open(file_name, "wb") as file:
        pickle.dump(model, file)

    file_name = file_name.replace('.pkl','')
    file_name = file_name + "history" + ".pkl"
    
    with open(file_name, "wb") as file:
        pickle.dump(history, file)

    return "1"

@app.route("/cnntrainer", methods=["POST"])
def cnn_trainer_endpoint():
    """Endpoint function for CNNTrainer

    Returns
    -------
    Jsonified dictionary
        When POST requested based on the action endpoint returns the results.
    """    
    action = request.json["action"]

    if request.method == "POST":
      
        if action == "compile":
            signal = compile_model(cnn_trainer)

            if signal == "1":
                return jsonify({'summary': "model is compiled"})
            else:
                return jsonify({'summary': "model is not compiled"})

        elif action == "fit":
            signal = fit_model(cnn_trainer)
            if signal == "1":
                return jsonify({'summary': "model is fitted"})
            else:
                return jsonify({'summary': "model is not fitted"})

        elif action == "all":
            compile_signal = compile_model(cnn_trainer)
            fit_signal = fit_model(cnn_trainer)
            if compile_signal == "1" and fit_signal == "1":
                return jsonify({'summary': "model is compiled and then fitted"})     
            else:
                return jsonify({'summary': "model is either not compiled or not fitted"})  
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
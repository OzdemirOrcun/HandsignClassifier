import json
import os
import asyncio
import aiohttp
import pickle
from datetime import datetime, timedelta
from cnn_creator import CNNCreator
import pickle


from flask import Flask, request,jsonify
from urllib.parse import urlencode
import glob

app = Flask(__name__)

only_safe_operations = os.getenv("only_safe_operations", "false")[0] == "t"

port = os.getenv("cnn_creator_port", 8010)


folder = "./output/"
if not os.path.exists(folder):
    os.makedirs(folder)

cnn_creator = CNNCreator()


def build_default_model(CNNCreator):

    cnn_model,model_summary = CNNCreator.build_default_model(verbose=1)
    current_datetime = datetime.now()

    day = current_datetime.day
    month = current_datetime.month
    year = current_datetime.year

    # Path to the pickle file
    file_path = f"./output/cnn_{day}_{month}_{year}.pkl"

    # Save the object to the pickle file
    with open(file_path, "wb") as file:
        pickle.dump(cnn_model, file)

    return model_summary

@app.route("/cnncreator", methods=["POST"])
def cnn_creator_endpoint():
    default = request.args.get("default", None)

    if request.method == "POST":
        # post the okay signal
        # under construction
        model_summary = build_default_model(cnn_creator)
        return jsonify({'summary': str(model_summary)})
    else:
        model_summary = build_default_model(cnn_creator)
        return jsonify({'summary': str(model_summary)})




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
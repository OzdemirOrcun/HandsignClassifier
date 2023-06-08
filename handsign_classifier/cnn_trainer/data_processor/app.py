import json
import os
import asyncio
import aiohttp
import pickle
from datetime import datetime, timedelta

from flask import Flask, request,jsonify
from urllib.parse import urlencode
import glob

app = Flask(__name__)

only_safe_operations = os.getenv("only_safe_operations", "false")[0] == "t"

port = os.getenv("data_processor_port", 8013)


folder = "./output/"
if not os.path.exists(folder):
    os.makedirs(folder)


@app.route("/dataprocessor", methods=["POST"])
def data_processor_endpoint():
    return "success"


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
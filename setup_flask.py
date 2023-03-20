
from flask import Flask
from flask import request
from flask_cors import CORS
from flask_caching import Cache

config = {
    "DEBUG": False,
    "CACHE_TYPE": "FileSystemCache",
    "CACHE_DIR": "cache-and-stuff",
    "CACHE_DEFAULT_TIMEOUT": 60*60*24*30,
    "CACHE_THRESHOLD": 1000*1000*1000,
}

app = Flask(__name__)

app.config.from_mapping(config)
cache = Cache(app)

CORS(app)
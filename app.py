from data_processing import get_map_json
from flask import Flask
from flask import request
from flask_cors import CORS
from flask_caching import Cache
from http_client import get_bl_leadeboard
from infer_publish import getDiffLabel, getMapComplexityForHits4, getMultiplierForAcc2, predictHitsForMap, predictHitsForMapFull
from data_processing_stats import get_map_stats
from tech_calc import techCalculation


config = {
    "DEBUG": False,
    "CACHE_TYPE": "FileSystemCache",
    "CACHE_DEFAULT_TIMEOUT": 60*60*30,
    "CACHE_DIR": "cache-and-stuff"
}

app = Flask(__name__)

app.config.from_mapping(config)
cache = Cache(app)

CORS(app)

@app.after_request
def add_header(response):
    response.cache_control.max_age = 60*60
    return response

@app.route("/")
def index():
    return "hi"


def normalize_sr(sr):
    return sr*15/2.56606282-1


def normalize_passing_sr(sr):
    return sr*15/3.00357589-1


@app.route('/<hash>')
@cache.cached(query_string=True)
def simple_hash(hash):
    results = ""
    # ignore_dots = True
    ignore_dots = True if request.args.get('ignore-dots') == "True" else False
    for mapName, hash, difficulty, accs, speeds in predictHitsForMap(hash.lower(), [1, 3, 5, 7, 9], ignore_dots):
        diffLabel = getDiffLabel(difficulty)
        sr, expected_acc, passing_sr, acc2 = getMapComplexityForHits4(accs, speeds)
        base = getMultiplierForAcc2(0.96)
        curr_mult = base/getMultiplierForAcc2(acc2)
        result = f"<div>Star rating - {round(normalize_sr(sr), 8)}*, difficulty to pass - {round(normalize_passing_sr(passing_sr), 8)}*, expected top play FC accuracy: {expected_acc} --- {mapName} {diffLabel}</div>"
            
        results += result
    if results == "":
        return "Not found"
    return results


@app.route('/json/<hash>/<diff>/basic')
@cache.cached(query_string=True)
def json_basic(hash, diff):
    ignore_dots = True if request.args.get('ignore-dots') == "True" else False
    for mapName, hash, difficulty, accs, speeds in predictHitsForMap(hash.lower(), [int(diff)], ignore_dots):
        sr, expected_acc, passing_sr, acc2 = getMapComplexityForHits4(accs, speeds)
        
        base = getMultiplierForAcc2(0.96)
        curr_mult = base/getMultiplierForAcc2(acc2)
        
        return {
            "balanced": normalize_sr(sr),
            "passing_difficulty": normalize_passing_sr(passing_sr),
            "expected_acc": expected_acc,
        }
    return "Not found"


@app.route('/json/<hash>/<diff>/full/time-scale/<scale>')
@cache.cached(query_string=True)
def json_full(hash, diff, scale):
    # ignore_dots = True
    ignore_dots = True if request.args.get('ignore-dots') == "True" else False
    for mapName, hash, difficulty, notes in predictHitsForMapFull(hash.lower(), [int(diff)], float(scale)):
        
        return notes
    
    return "Not found"



@app.route('/json/<hash>/<diff>/time-scale/<scale>')
@cache.cached(query_string=True)
def json_time_scale(hash, diff, scale):
    ignore_dots = True if request.args.get('ignore-dots') == "True" else False
    for mapName, hash, difficulty, accs, speeds in predictHitsForMap(hash.lower(), [int(diff)], ignore_dots, float(scale)):
        sr, expected_acc, passing_sr, acc2 = getMapComplexityForHits4(accs, speeds)
        
        base = getMultiplierForAcc2(0.96)
        curr_mult = base/getMultiplierForAcc2(acc2)
        
        return {
            "balanced": normalize_sr(sr),
            "passing_difficulty": normalize_passing_sr(passing_sr),
            "expected_acc": expected_acc,
        }
    return "Not found"


@app.route('/json/<hash>/<diff>/fixed-time/<time>/<njs>')
@cache.cached(query_string=True)
def json_fixed_time(hash, diff, time, njs):
    ignore_dots = True if request.args.get('ignore-dots') == "True" else False
    for mapName, hash, difficulty, accs, speeds in predictHitsForMap(hash.lower(), [int(diff)], ignore_dots, 1, float(time), float(njs)):
        sr, expected_acc, passing_sr, acc2 = getMapComplexityForHits4(accs, speeds)
        
        base = getMultiplierForAcc2(0.96)
        curr_mult = base/getMultiplierForAcc2(acc2)
        
        return {
            "balanced": normalize_sr(sr),
            "passing_difficulty": normalize_passing_sr(passing_sr),
            "expected_acc": expected_acc,
        }
    return "Not found"



@app.route('/stats/<hash>/<mode>/<diff>')
@cache.cached()
def api_get_map_stats(hash, mode, diff):
    res = get_map_stats(hash.lower(), mode, int(diff))
    if res is None:
        return "Not found"
    return res



@app.route('/bl-leaderboard/<hash>/<diff>/<mode>')
@cache.cached()
def api_get_map_leaderboard(hash, diff, mode):
    bl_leaderboard = get_bl_leadeboard(hash, diff, mode)
    return bl_leaderboard



@app.route('/lack-tech-calculator/<hash>/<characteristic>/<diff>')
@cache.cached()
def lack_tech_calculator(hash, characteristic, diff):
    map_json = get_map_json(hash.lower(), characteristic, int(diff))
    return {
        "tech": techCalculation(map_json, False)
    }



if __name__ == '__main__':
    app.run()
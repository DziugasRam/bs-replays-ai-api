from data_processing import get_map_info
from flask import Flask
from flask import request
from flask_cors import CORS
from flask_caching import Cache
from http_client import get_bl_leadeboard
from infer_publish import getDiffLabel, getMapComplexityForHits4, getMultiplierForAcc2, predictHitsForMap, predictHitsForMapFull, scaleFarmability
from data_processing_stats import get_map_stats
from tech_calc import mapCalculation


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
    return sr*14/3.7807760790332994


def normalize_passing_sr(sr):
    return sr*14/4.319542203926959


@app.route('/<hash>')
@cache.cached(query_string=True)
def simple_hash(hash):
    results = ""
    # ignore_dots = True
    ignore_dots = True if request.args.get('ignore-dots') == "True" else False
    for mapName, hash, difficulty, accs, speeds, noteTimes in predictHitsForMap(hash.lower(), None, [1, 3, 5, 7, 9], ignore_dots):
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
def json_basic_deprecated(hash, diff):
    ignore_dots = True if request.args.get('ignore-dots') == "True" else False
    for mapName, hash, difficulty, accs, speeds, noteTimes in predictHitsForMap(hash.lower(), None, [int(diff)], ignore_dots):
        sr, expected_acc, passing_sr, acc2 = getMapComplexityForHits4(accs, speeds)
        
        base = getMultiplierForAcc2(0.96)
        curr_mult = base/getMultiplierForAcc2(acc2)
        
        return {
            "balanced": normalize_sr(sr),
            "passing_difficulty": normalize_passing_sr(passing_sr),
            "expected_acc": expected_acc,
            "speed": sum(speeds)/len(speeds),
            "square_root_speed": (sum([s**2 for s in speeds])/len(speeds))**0.5
        }
    return "Not found"


@app.route('/json/<hash>/<characteristic>/<diff>/basic')
@cache.cached(query_string=True)
def json_basic(hash, characteristic, diff):
    ignore_dots = True if request.args.get('ignore-dots') == "True" else False
    for mapName, hash, difficulty, accs, speeds, noteTimes in predictHitsForMap(hash.lower(), characteristic, [int(diff)], ignore_dots):
        sr, expected_acc, passing_sr, acc2 = getMapComplexityForHits4(accs, speeds)
        
        base = getMultiplierForAcc2(0.96)
        curr_mult = base/getMultiplierForAcc2(acc2)
        
        return {
            "balanced": normalize_sr(sr),
            "passing_difficulty": normalize_passing_sr(passing_sr),
            "expected_acc": expected_acc,
            "speed": sum(speeds)/len(speeds),
            "square_root_speed": (sum([s**2 for s in speeds])/len(speeds))**0.5
        }
    return "Not found"


@app.route('/json/<hash>/<characteristic>/<diff>/adjust-farming/<farmminutes>')
@cache.cached(query_string=True)
def json_adjust_farming(hash, characteristic, diff, farmminutes):
    ignore_dots = True if request.args.get('ignore-dots') == "True" else False
    for mapName, hash, difficulty, accs, speeds, noteTimes in predictHitsForMap(hash.lower(), characteristic, [int(diff)], ignore_dots):
        sr, expected_acc, passing_sr, acc2 = getMapComplexityForHits4(accs, speeds)

        AIacc = (sum(accs)/len(accs)*15+100)/115
        adjustedAIacc = scaleFarmability(AIacc, len(accs), (noteTimes[-1] - noteTimes[0])/60+0.25, farm_session_length=int(farmminutes)*60)
        
        return {
            "balanced": normalize_sr(sr),
            "passing_difficulty": normalize_passing_sr(passing_sr),
            "expected_acc": adjustedAIacc,
            "speed": sum(speeds)/len(speeds),
            "square_root_speed": (sum([s**2 for s in speeds])/len(speeds))**0.5
        }
    return "Not found"


@app.route('/json/<hash>/<characteristic>/<diff>/full/time-scale/<scale>')
@cache.cached(query_string=True)
def json_full(hash, characteristic, diff, scale):
    full_notes = []
    basic_stats = {
        "balanced": 0,
        "passing_difficulty": 0,
        "acc": 0,
        "speed": 0,
        "lack_tech": 0,
        "lack_passing_difficulty": 0,
    }
    
    for mapName, hash, difficulty, notes in predictHitsForMapFull(hash.lower(), characteristic, [int(diff)], float(scale)):
        
        full_notes = notes
    
    for mapName, hash, difficulty, accs, speeds, noteTimes in predictHitsForMap(hash.lower(), characteristic, [int(diff)], False, float(scale)):
        sr, expected_acc, passing_sr, acc2 = getMapComplexityForHits4(accs, speeds)
        
        basic_stats["balanced"] = normalize_sr(sr)
        basic_stats["passing_difficulty"] = normalize_passing_sr(passing_sr)
        basic_stats["acc"] = (sum(accs)/len(accs)*15+100)/115
        basic_stats["speed"] = sum(speeds)/len(speeds)
        
    map_info = get_map_info(hash.lower(), characteristic, int(diff))
    lack_map_calculation = mapCalculation(map_info["map_json"], map_info["bpm"], False, False)
    basic_stats["tech"] = lack_map_calculation["balanced_tech"]
    basic_stats["lack_passing_difficulty"] = lack_map_calculation["balanced_pass_diff"]
        
    return {
        "stats": basic_stats,
        "notes": full_notes,
    }



@app.route('/json/<hash>/<diff>/time-scale/<scale>')
@cache.cached(query_string=True)
def json_time_scale(hash, diff, scale):
    ignore_dots = True if request.args.get('ignore-dots') == "True" else False
    for mapName, hash, difficulty, accs, speeds, noteTimes in predictHitsForMap(hash.lower(), None, [int(diff)], ignore_dots, float(scale)):
        sr, expected_acc, passing_sr, acc2 = getMapComplexityForHits4(accs, speeds)
        
        return {
            "balanced": normalize_sr(sr),
            "passing_difficulty": normalize_passing_sr(passing_sr),
            "expected_acc": expected_acc,
            "speed": sum(speeds)/len(speeds),
            "square_root_speed": (sum([s**2 for s in speeds])/len(speeds))**0.5
        }
    return "Not found"


@app.route('/json/<hash>/<diff>/time-scale/<scale>/<njs>')
@cache.cached(query_string=True)
def json_time_scale_njs(hash, diff, scale, njs):
    ignore_dots = True if request.args.get('ignore-dots') == "True" else False
    for mapName, hash, difficulty, accs, speeds, noteTimes in predictHitsForMap(hash.lower(), None, [int(diff)], ignore_dots, float(scale), fixed_njs=float(njs)):
        sr, expected_acc, passing_sr, acc2 = getMapComplexityForHits4(accs, speeds)
        
        return {
            "balanced": normalize_sr(sr),
            "passing_difficulty": normalize_passing_sr(passing_sr),
            "expected_acc": expected_acc,
            "speed": sum(speeds)/len(speeds),
            "square_root_speed": (sum([s**2 for s in speeds])/len(speeds))**0.5
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
def api_get_lack_tech_calculator(hash, characteristic, diff):
    map_info = get_map_info(hash.lower(), characteristic, int(diff))
    return mapCalculation(map_info["map_json"], map_info["bpm"], False, False)







def curveAccMulti(acc):
    pointList = [[1, 7],[0.999, 5.8],[0.9975, 4.7],[0.995, 3.76],[0.9925, 3.17],[0.99, 2.73],[0.9875, 2.38],[0.985, 2.1],[0.9825, 1.88],[0.98, 1.71],[0.9775, 1.57],[0.975, 1.45],[0.9725, 1.37],[0.97, 1.31],[0.965, 1.20],[0.96, 1.11],[0.955, 1.045],[0.95, 1],[0.94, 0.94],[0.93, 0.885],[0.92, 0.835],[0.91, 0.79],[0.9, 0.75],[0.875, 0.655],[0.85, 0.57],[0.825, 0.51],[0.8, 0.47],[0.75, 0.40],[0.7, 0.34],[0.65, 0.29],[0.6, 0.25],[0.0, 0.0]] # An array of pairs of (ACC, Multiplier)
    for i in range(0, len(pointList)):
        if pointList[i][0] <= acc:  # Searches the acc portion of each pair in the array until it finds a pair with a lower acc than the players acc, then breaks
            break
    
    if i == 0:  # Special case for 100% acc scores
        i = 1
    
    middle_dis = (acc - pointList[i-1][0]) / (pointList[i][0] - pointList[i-1][0]) 
    return pointList[i-1][1] + middle_dis * (pointList[i][1] - pointList[i-1][1])



@app.route('/bl-reweight/<hash>/<characteristic>/<diff>')
@cache.cached(query_string=True)
def api_get_bl_reweight(hash, characteristic, diff):
    modifiers = [
        ["SS", 0.85],
        ["none", 1],
        ["FS", 1.2],
        ["SFS", 1.5],
    ]
    results = {}
    for name, scale in modifiers:
        AIacc = 0
        for mapName, hash, difficulty, accs, speeds, noteTimes in predictHitsForMap(hash.lower(), characteristic, [int(diff)], False, skip_speed=True, time_scale=float(scale)):
            AIacc = (sum(accs)/len(accs)*15+100)/115
            adjustedAIacc = scaleFarmability(AIacc, len(accs), (noteTimes[-1] - noteTimes[0])/60+0.25)
            AIacc = adjustedAIacc
        map_info = get_map_info(hash.lower(), characteristic, int(diff))
        lack_map_calculation = mapCalculation(map_info["map_json"], map_info["bpm"] * float(scale), False, False)

        results[name] = { "lack_map_calculation": lack_map_calculation, "AIacc": AIacc }
    return results

if __name__ == '__main__':
    app.run()
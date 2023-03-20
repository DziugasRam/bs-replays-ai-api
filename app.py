from data_processing import get_map_info
from infer_publish import getMapAccForHits, predictHitsForMap, scaleFarmability
from tech_calc import mapCalculation

from setup_flask import app
from setup_flask import cache

@app.after_request
def add_header(response):
    response.cache_control.max_age = 60*60
    return response


@app.route("/")
def index():
    return "hi"


@app.route('/<hash>/<characteristic>/<diff>')
def api_get_ratings(hash, characteristic, diff):
    modifiers = [
        ["SS", "0.85"],
        ["none", "1"],
        ["FS", "1.2"],
        ["SFS", "1.5"],
    ]
    results = {}
    for name, timescale in modifiers:
        results[name] = get_bl_ratings(hash, characteristic, diff, timescale)
    return results


@app.route('/<hash>/<characteristic>/<diff>/<timescale>')
def api_get_rating(hash, characteristic, diff, timescale):
    return get_bl_ratings(hash, characteristic, diff, timescale)


@cache.memoize()
def get_bl_ratings(hash, characteristic, diff, timescale):
    accs, noteTimes, free_points = predictHitsForMap(hash.lower(), characteristic, int(diff), exclude_dots=False, time_scale=float(timescale))
    AIacc = getMapAccForHits(accs, free_points)
    adjustedAIacc = scaleFarmability(AIacc, len(accs), (noteTimes[-1] - noteTimes[0])+15)
    AIacc = adjustedAIacc

    map_info = get_map_info(hash.lower(), characteristic, int(diff))
    lack_map_calculation = mapCalculation(map_info["map_json"], map_info["bpm"] * float(timescale), False, False)
    return { "lack_map_calculation": lack_map_calculation, "AIacc": AIacc }


if __name__ == '__main__':
    app.run()

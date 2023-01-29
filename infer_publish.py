import os
import math
import datetime

import numpy as np
import tensorflow as tf
from data_processing import preprocess_map

from data_processing import pre_segment_size
from data_processing import post_segment_size

import requests

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def custom_loss(y_true, y_pred):
    return (tf.abs(y_true - y_pred)**1.5)

model_acc = tf.keras.models.load_model("model_sleep_lstm_acc", custom_objects={"custom_loss": custom_loss})
model_speed = tf.keras.models.load_model("model_sleep_lstm_speed", custom_objects={"custom_loss": custom_loss})


def getDiffLabel(difficulty):
    diffLabel = difficulty
    if difficulty == 1:
        diffLabel = "Easy"
    if difficulty == 3:
        diffLabel = "Normal"
    if difficulty == 5:
        diffLabel = "Hard"
    if difficulty == 7:
        diffLabel = "Expert"
    if difficulty == 9:
        diffLabel = "ExpertPlus"
    return diffLabel


def getMultiplierForCombo(combo):
    if combo <= 1:
        return 1
    elif combo <= 1 + 4:
        return 2
    elif combo <= 1 + 4 + 8:
        return 4
    else:
        return 8


def getMaxScoreForNotes(note_count):
    totalScore = 0
    
    for i in range(note_count):
        multiplier = getMultiplierForCombo(i + 1)
        totalScore += 115 * multiplier
    
    return totalScore


def getMapAccForHits(hits):
    totalMultipliers = 0
    totalScore = 0
    
    for i, hit in enumerate(hits):
        multiplier = getMultiplierForCombo(i + 1)
        totalScore += ((hit*15+100)/115) * multiplier
        totalMultipliers += multiplier
    
    return totalScore/totalMultipliers


def getTopScore(hash, difficulty):
    url = f"https://scoresaber.com/api/leaderboard/by-hash/{hash}/scores?difficulty={difficulty}"
    r = requests.get(url = url)
    leaderboard = r.json()
    score = leaderboard["scores"][0]["baseScore"]
    number_of_notes = max(leaderboard["scores"][0]["maxCombo"], 0)
    time_set = leaderboard["scores"][0]["timeSet"]
    player_name = leaderboard["scores"][0]["leaderboardPlayerInfo"]["name"]
    max_score = getMaxScoreForNotes(number_of_notes)
    age = (datetime.datetime.now() - datetime.datetime.strptime(time_set, "%Y-%m-%dT%H:%M:%S.000Z")).days
    return score/max_score, player_name, age


def predictHitsForMap(hash, characteristic, difficulties, exclude_dots, time_scale = 1, fixed_time_distance = None, fixed_njs = None, skip_speed = False):
    try:
        for difficulty in difficulties:
            segments, scores, songName, note_times = preprocess_map(hash, characteristic, difficulty, time_scale, fixed_time_distance, fixed_njs)
            if len(segments) == 0:
                continue

            predictions_arrays_acc = model_acc.predict(np.array(segments), verbose=0)
            if skip_speed:
                predictions_arrays_speed = predictions_arrays_acc
            else:
                predictions_arrays_speed = model_speed.predict(np.array(segments), verbose=0)
                
            accs = []
            speeds = []

            for batch_pred, batch_pred_speed, batch_inp in zip(predictions_arrays_acc, predictions_arrays_speed, segments):
                for pred, pred_speed, inp in zip(batch_pred, batch_pred_speed, batch_inp[pre_segment_size:-post_segment_size]):
                    if sum(inp) == 0.0:
                        continue
                    if exclude_dots and (inp[4*3+8] > 0 or inp[4*3+10+4*3+8] > 0):
                        continue
                    
                    accs.append(pred[0])
                    speeds.append(pred_speed[0])
                    
            
            yield songName, hash, difficulty, accs, speeds, note_times
    except Exception as e:
        print(e)
        print(hash, difficulties)


def predictHitsForMapFull(hash, characteristic, difficulties, time_scale = 1, fixed_time_distance = None, fixed_njs = None):
    try:
        for difficulty in difficulties:
            segments, scores, songName, note_times = preprocess_map(hash, characteristic, difficulty, time_scale, fixed_time_distance, fixed_njs)
            if len(segments) == 0:
                continue

            predictions_arrays_acc = model_acc.predict(np.array(segments), verbose=0)
            predictions_arrays_speed = model_speed.predict(np.array(segments), verbose=0)

            notes = {
                "columns": ["acc", "speed", "note_color", "is_dot", "note_time"],
                "rows": []
            }
            note_times_iterator = 0
            for batch_pred, batch_pred_speed, batch_inp in zip(predictions_arrays_acc, predictions_arrays_speed, segments):
                for pred, pred_speed, inp in zip(batch_pred, batch_pred_speed, batch_inp[pre_segment_size:-post_segment_size]):
                    if sum(inp) == 0.0:
                        continue
                    
                    notes["rows"].append([
                        round(float(pred[0]), 5),
                        round(float(pred_speed[0]), 5),
                        0 if sum(inp[:4*3+10]) > 1 else 1,
                        1 if inp[4*3 + 8] == 1 or inp[4*3 + 10 + 4*3 + 8] == 1 else 0,
                        note_times[note_times_iterator]
                    ])
                    note_times_iterator += 1
                    
            
            yield songName, hash, difficulty, notes
    except Exception as e:
        print(e)
        print(hash, difficulties)

curve2 = [
    [0, 0],
 [0.9349050106584025, 0.12752369975021405],
 [0.9361096414526044, 0.1338810660310752],
 [0.9373168654378441, 0.14023889931041011],
 [0.9385245851320524, 0.14659082820408054],
 [0.9397223493859982, 0.1528866740035726],
 [0.9409324581850277, 0.15924843852972081],
 [0.9421364984358537, 0.16558418281585519],
 [0.943340858876901, 0.17193243240220124],
 [0.9445528451823, 0.17833692668472112],
 [0.9457521950057954, 0.1846956949599493],
 [0.9469652757511278, 0.19115401643801677],
 [0.9481613689691947, 0.19755379454404964],
 [0.9493682127874609, 0.20404903478780123],
 [0.9505744372202971, 0.21058515128474997],
 [0.9517783524884541, 0.2171592678874248],
 [0.9529892330175649, 0.22382918760822046],
 [0.9541947185853665, 0.23053419320410745],
 [0.9554044516127758, 0.23733558864799276],
 [0.9566054381494079, 0.24416812135896532],
 [0.957807698698665, 0.25109683767385865],
 [0.9590221672423604, 0.2581958900445771],
 [0.9602231628864696, 0.26532531816166977],
 [0.961433998563471, 0.2726343255815458],
 [0.9626279572859802, 0.27997264641647096],
 [0.963842342827163, 0.28758364824299726],
 [0.965050103040447, 0.29531513712077556],
 [0.96624960935703, 0.30316989818393675],
 [0.9674529869368587, 0.3112442874173377],
 [0.9686591348667645, 0.3195525631025941],
 [0.9698668993297, 0.3281104611939074],
 [0.9710750806787853, 0.3369354081638326],
 [0.9722824425660789, 0.34604677365953185],
 [0.9734877233230004, 0.3554661722556236],
 [0.9746896497445529, 0.3652178262195114],
 [0.9759017934808459, 0.375457006140961],
 [0.977108280868104, 0.3861001218643532],
 [0.9783078742661878, 0.3971842458807886],
 [0.9795145289674262, 0.4089026020635034],
 [0.9807121922419519, 0.42117208266483336],
 [0.981930401543003, 0.4343941913166449],
 [0.9831227248036967, 0.44816346163162424],
 [0.9843344315883069, 0.46312172007358454],
 [0.9855345565106794, 0.47904648122941484],
 [0.9867538435462135, 0.4965454829250852],
 [0.9879462160499057, 0.5151729970360741],
 [0.989158767583543, 0.535945456051768],
 [0.9903616429313051, 0.5587509916921046],
 [0.9915723216173138, 0.5844172501512462],
 [0.9927779343719173, 0.6133568698454486],
 [0.9939826353978779, 0.6465866613825485],
 [0.9951928260723995, 0.6856877355734612],
 [0.99638391362715, 0.7318561252640624],
 [0.9975978174482817, 0.7905788225140824],
 [0.9988016676122579, 0.8703332399757727],
 [0.9997988680153226, 1.0],
 [1, 1]
]
curve = [
    (0.0, 0.0),
    (0.2553285380807287, 0.0),
 (0.7800594896882996, 0.13929361210187519),
 (0.8900188541854286, 0.19574786858546012),
 (0.9200109260611372, 0.2212116529231017),
 (0.9350291474407364, 0.23858302501929107),
 (0.9460175689234617, 0.25509974796057455),
 (0.9580166418422442, 0.2802612249088514),
 (0.9690021380342436, 0.3182800817388961),
 (0.9750066550650179, 0.35326901832837027),
 (0.9820033004409089, 0.41777520435196047),
 (0.9870075083185075, 0.4848591933969141),
 (0.9915132116476629, 0.5704049016338111),
 (0.995013841615207, 0.6700865738809406),
 (0.9999793102866608, 1.0),
 (1, 1.0)
]

def getMultiplierForAcc(acc):
    if acc > 1:
        return 1
    previousCurvePointAcc = 0
    previousCurvePointMultiplier = 0
    
    for curvePointAcc, curvePointMultiplier in curve:
        if acc <= curvePointAcc:
            accDiff = (curvePointAcc - previousCurvePointAcc)
            multiplierDiff = (curvePointMultiplier - previousCurvePointMultiplier)
            slope = multiplierDiff/accDiff
            multiplier = previousCurvePointMultiplier + slope * (acc - previousCurvePointAcc)
            return multiplier
        
        previousCurvePointAcc = curvePointAcc
        previousCurvePointMultiplier = curvePointMultiplier

def getMultiplierForAcc2(acc):
    if acc > 1:
        return 1
    previousCurvePointAcc = 0
    previousCurvePointMultiplier = 0
    
    for curvePointAcc, curvePointMultiplier in curve2:
        if acc <= curvePointAcc:
            accDiff = (curvePointAcc - previousCurvePointAcc)
            multiplierDiff = (curvePointMultiplier - previousCurvePointMultiplier)
            slope = multiplierDiff/accDiff
            multiplier = previousCurvePointMultiplier + slope * (acc - previousCurvePointAcc)
            return multiplier
        
        previousCurvePointAcc = curvePointAcc
        previousCurvePointMultiplier = curvePointMultiplier



def getMultiplierForAcc(acc):
    if acc > 1:
        return 1
    curve = [
        [0.0, 0.0], [0.78006, 0.42749], [0.89002, 0.60075], [0.92001, 0.6789], [0.93503, 0.73221], [0.94602, 0.7829], [0.95802, 0.86012], [0.969, 0.9768], [0.97501, 1.08418], [0.982, 1.28215], [0.98701, 1.48803], [0.99151, 1.75057], [0.99501, 2.0565], [0.99999999, 3.069], [1, 1]
    ]
    previousCurvePointAcc = 0
    previousCurvePointMultiplier = 0
    
    for curvePointAcc, curvePointMultiplier in curve:
        if acc <= curvePointAcc:
            accDiff = (curvePointAcc - previousCurvePointAcc)
            multiplierDiff = (curvePointMultiplier - previousCurvePointMultiplier)
            slope = multiplierDiff/accDiff
            multiplier = previousCurvePointMultiplier + slope * (acc - previousCurvePointAcc)
            return multiplier
        
        previousCurvePointAcc = curvePointAcc
        previousCurvePointMultiplier = curvePointMultiplier
        

def getAccForMultiplier(multiplier):
    curve = [
        [0.0, 0.0], [0.78006, 0.42749], [0.89002, 0.60075], [0.92001, 0.6789], [0.93503, 0.73221], [0.94602, 0.7829], [0.95802, 0.86012], [0.969, 0.9768], [0.97501, 1.08418], [0.982, 1.28215], [0.98701, 1.48803], [0.99151, 1.75057], [0.99501, 2.0565], [0.99999999, 3.069], [1, 1]
    ]
    previousCurvePointMultiplier = 0
    previousCurvePointAcc = 0
    
    for curvePointAcc, curvePointMultiplier in curve:
        if multiplier <= curvePointMultiplier:
            multDiff = (curvePointMultiplier - previousCurvePointMultiplier)
            accDiff = (curvePointAcc - previousCurvePointAcc)
            slope = accDiff/multDiff
            acc = previousCurvePointAcc + slope * (multiplier - previousCurvePointMultiplier)
            return acc
        
        previousCurvePointMultiplier = curvePointMultiplier
        previousCurvePointAcc = curvePointAcc


def scaleFarmability(acc, noteCount, mapLength):
    farm_session_length = 30*60
    base_map_length = 60
    base_attempts = farm_session_length/base_map_length
    base_note_count = 200
    base_multiplier = 0.030963633
    
    if noteCount > base_note_count * 5:
        return acc
    
    total_attempts = max(1, farm_session_length/(mapLength))
    attempts_scale = total_attempts/base_attempts
    note_scale = (noteCount + 5)/base_note_count # my segment count is in 8 notes, so this is in case I cut off the end of the map
    
    note_scaler = (((math.log(note_scale, 0.05)+0.46275642631951835)**2+0.2431278387816179)/2.2641367447629013)/0.20
    attempts_scaler = ((math.log(attempts_scale)+2.7081502061025433)**0.75/2.1110793685981553)
    multiplier = getMultiplierForAcc(acc) + base_multiplier*(note_scaler*attempts_scaler)
    return getAccForMultiplier(multiplier)


def getComplexity(acc, speed):
    normalized_acc = ((acc + 0.0)*15+100)/115
    mult = (getMultiplierForAcc(normalized_acc) + speed**0.69/6.9)/(1+speed**0.69/6.9)
    return (speed**0.69)/mult


def getExpectedAcc(acc, speed):
    return ((acc + 0.05*(1-min(1, speed)))*15+100)/115

def getExpectedAcc1(acc, speed):
    return ((acc)*15+100)/115

def getExpectedAcc2(acc, speed):
    return ((acc + 0.069*(1-min(1, speed/1.3)))*15+100)/115


def getMapComplexityForHits4(accs, speeds):
    window_size, top_window, skip_window = (50, 25, 5)
    complexities = [getComplexity(acc, speed*2) for acc, speed in zip(accs, speeds)]
    expected_accs = [getExpectedAcc1(acc, speed*2) for acc, speed in zip(accs, speeds)]
    expected_accs2 = [getExpectedAcc2(acc, speed*2) for acc, speed in zip(accs, speeds)]

    passing_averages = []
    top_comp = 0
    
    for i in range(max(len(complexities)-window_size, 1)):
        if i % skip_window != 0:
            continue
        window = sorted(complexities[i:i+window_size], reverse=True)
        curr = sum(window[skip_window:top_window])/(min(top_window-skip_window, len(window)))
        passing_averages.append(curr)
        top_comp = max(top_comp, curr)
        
    power = 3
    return (sum([comp**power for comp in passing_averages])/len(passing_averages))**(1/power), sum(expected_accs)/len(expected_accs), top_comp, sum(expected_accs2)/len(expected_accs2)

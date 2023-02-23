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

model_acc = tf.keras.models.load_model("model_sleep_3gru_acc")
model_speed = tf.keras.models.load_model("model_sleep_3gru_speed")

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
            model_input = tf.convert_to_tensor(np.array(segments), dtype=tf.float32)
            predictions_arrays_acc = model_acc.predict(model_input, verbose=0)
            if skip_speed:
                predictions_arrays_speed = predictions_arrays_acc
            else:
                predictions_arrays_speed = model_speed.predict(model_input, verbose=0)
                
            accs = []
            speeds = []

            for batch_pred, batch_pred_speed, batch_inp in zip(predictions_arrays_acc, predictions_arrays_speed, segments):
                for pred, pred_speed, inp in zip(batch_pred, batch_pred_speed, batch_inp[pre_segment_size:-post_segment_size]):
                    if sum(inp) == 0.0:
                        continue
                    if exclude_dots and (inp[4*3+8] > 0 or inp[4*3+10+4*3+8] > 0):
                        continue
                    if pred_speed[0] < 0:
                        print(inp)
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
            model_input = tf.convert_to_tensor(np.array(segments), dtype=tf.float32)
            predictions_arrays_acc = model_acc.predict(model_input, verbose=0)
            predictions_arrays_speed = model_speed.predict(model_input, verbose=0)

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

scaleCurve = [
    (0.0, 0.0),
 (0.6003503520303082, 0.10734016938160607),
 (0.8000625740167697, 0.1670550225942915),
 (0.9000130232242154, 0.22043326958200493),
 (0.9004830061775256, 0.22079472143247547),
 (0.9008578584881118, 0.22108433778130776),
 (0.9012317019288727, 0.22137435968435726),
 (0.9016975863665102, 0.2217374591070017),
 (0.9020691571275727, 0.22202839767296267),
 (0.9025321984028848, 0.22239264654902058),
 (0.902901492813263, 0.22268450759080094),
 (0.9032697743659337, 0.22297678050218772),
 (0.903728700925489, 0.2233427026069449),
 (0.9040947010422176, 0.22363590648868426),
 (0.9045507736831233, 0.22400299601642126),
 (0.9049144889963928, 0.22429713681826335),
 (0.9053677035885798, 0.22466540124182088),
 (0.9057291308582497, 0.22496048497073595),
 (0.9061794834320446, 0.225329931834937),
 (0.9065386195472332, 0.22562596455585027),
 (0.906986106295722, 0.22599660147842737),
 (0.9074319993347607, 0.22636790286330488),
 (0.9077875655580991, 0.22666542398750644),
 (0.9082305871790576, 0.2270379274277119),
 (0.9085838548982231, 0.22733641331166182),
 (0.9090240017209117, 0.22771012661557138),
 (0.9094625500357997, 0.22808451546802871),
 (0.9098122371154416, 0.2283845145887214),
 (0.9102479057151944, 0.2287601256033323),
 (0.910681973192462, 0.2291364190510281),
 (0.9110280737760723, 0.2294379468299907),
 (0.9114592570446779, 0.2298154749267708),
 (0.9118888368138129, 0.23019369244695265),
 (0.9123168122931197, 0.23057260191312606),
 (0.9126580370766751, 0.23087622939121183),
 (0.9130831229422995, 0.2312563907614374),
 (0.9135066024717686, 0.231637251215024),
 (0.913928474994476, 0.23201881332781527),
 (0.914348739871884, 0.23240107968991697),
 (0.9147673964977523, 0.23278405290580323),
 (0.9151844442983702, 0.23316773559442294),
 (0.9155169238200572, 0.23347519433492692),
 (0.9159310743941844, 0.23386016072083338),
 (0.916343614711684, 0.23424584399045378),
 (0.916754544324557, 0.23463224681872935),
 (0.9171638628184423, 0.23501937189560096),
 (0.9175715698128453, 0.23540722192612146),
 (0.9179776649613693, 0.2357957996305694),
 (0.9183821479519463, 0.23618510774456292),
 (0.9187850185070672, 0.2365751490191757),
 (0.9191862763840127, 0.23696592622105395),
 (0.9196656568111918, 0.23743583420593833),
 (0.9200633661146927, 0.23782824026388932),
 (0.9204594621997773, 0.23822139120839592),
 (0.920853944972351, 0.23861528987280595),
 (0.9212468143743018, 0.2390099391066638),
 (0.9216380703837297, 0.2394053417758345),
 (0.9221054479399644, 0.23988082357291024),
 (0.9224931545886673, 0.24027789396884247),
 (0.9228792480207663, 0.24067572708199075),
 (0.9233404308828069, 0.24115413773927785),
 (0.9237229757580672, 0.24155365918819754),
 (0.9241039079732349, 0.24195395280882098),
 (0.924558898279205, 0.24243532863220346),
 (0.924936283586876, 0.24283733158199783),
 (0.9253120571789561, 0.24324011633487347),
 (0.9257608585568515, 0.24372449430951199),
 (0.9261330877524391, 0.24412900977896695),
 (0.926577637134521, 0.24461547354801821),
 (0.926946324168112, 0.245021734671595),
 (0.9273866244870451, 0.24551030227615733),
 (0.9278246085523616, 0.2460000251153689),
 (0.9281878267087355, 0.2464090139018808),
 (0.9286215672428673, 0.24690086892216662),
 (0.9289812508322953, 0.24731164226993396),
 (0.9294107520707715, 0.24780564811889622),
 (0.9298379429380284, 0.24830083509456546),
 (0.9302628249351512, 0.24879720885843115),
 (0.9306151306883049, 0.24921176434865655),
 (0.9310357839415226, 0.2497103287307138),
 (0.9314541330468524, 0.250210096164748),
 (0.9318701798381535, 0.25071107247081265),
 (0.9322839262373341, 0.2512132635112973),
 (0.932695374254855, 0.25171667519133983),
 (0.9331045259902306, 0.25222131345924215),
 (0.9335113836325192, 0.25272718430689306),
 (0.9339159494608191, 0.2532342937701939),
 (0.9343182258447497, 0.2537426479294916),
 (0.9347182152449318, 0.2542522529100157),
 (0.9351159202134625, 0.2547631148823213),
 (0.9355113433943836, 0.25527524006273705),
 (0.9359697901686557, 0.25587432435208246),
 (0.9363602789878844, 0.2563892080297637),
 (0.9367484950041108, 0.2569053749077684),
 (0.9371985451695712, 0.2575091999783638),
 (0.937581847306741, 0.2580281691847736),
 (0.9379628865664157, 0.25854844206211497),
 (0.9384045767942713, 0.25915708354370215),
 (0.9387807247110888, 0.2596802036336343),
 (0.9392167179267369, 0.2602921850685731),
 (0.9396496517145889, 0.26090598016804445),
 (0.940018307281584, 0.2614335419524784),
 (0.940445575241413, 0.2620507339641663),
 (0.9408698013000969, 0.2626697707025034),
 (0.9412310074412941, 0.2632018502542468),
 (0.9416496015963305, 0.2638243423332387),
 (0.9420651728615702, 0.2644487110058738),
 (0.942477728244985, 0.26507496762090643),
 (0.9428872749392461, 0.2657031236303513),
 (0.9432938203223926, 0.2663331905907398),
 (0.9436973719584751, 0.26696518016439574),
 (0.9440979375981774, 0.2675991041207309),
 (0.9444955251794026, 0.26823497433755933),
 (0.9449462748165486, 0.26896408171560354),
 (0.9453375084475107, 0.2696041629994546),
 (0.9457257901926643, 0.2702462285948241),
 (0.9461659373982003, 0.2709824634526558),
 (0.9465479232779083, 0.2716288228500322),
 (0.9469808988941315, 0.27236999778592597),
 (0.9473566308160905, 0.2730207088014467),
 (0.9477824838094953, 0.27376689055838577),
 (0.9482045588345974, 0.27451577037906605),
 (0.9486228707579505, 0.2752673678457759),
 (0.9489858187078611, 0.2759272605053248),
 (0.9493971160253739, 0.2766840070860189),
 (0.9498046948070988, 0.2774435288146157),
 (0.9502085711897503, 0.27820584612015026),
 (0.9506585271099185, 0.2790668204043074),
 (0.9510545909720712, 0.27983514717960706),
 (0.9514470049779814, 0.2806063348472998),
 (0.9518841298062041, 0.281477367269969),
 (0.9522688463492481, 0.2822547056384368),
 (0.9526973572927545, 0.2831327129046036),
 (0.9530744584949815, 0.28391630087407405),
 (0.9534944497046729, 0.28480139558853956),
 (0.9539099697844494, 0.28569028900372434),
 (0.9543210466282215, 0.28658301386719387),
 (0.9547277086479302, 0.2874796033518093),
 (0.9551299847716085, 0.2883800910631242),
 (0.9555279044412088, 0.28928451104694164),
 (0.955921497610191, 0.2901928977970392),
 (0.956353786130413, 0.2912069113125735),
 (0.9567383462278782, 0.2921237876845015),
 (0.9571606754166833, 0.2931473224905135),
 (0.9575778262760373, 0.29417594067776554),
 (0.9579898435289135, 0.2952096929915714),
 (0.9583967726457088, 0.2962486309409123),
 (0.9587986598369218, 0.29729280681383824),
 (0.959195552045315, 0.29834227369325816),
 (0.9596264211809225, 0.2995028627201676),
 (0.9600129799333861, 0.3005636171263851),
 (0.9604326006146046, 0.30173675635642944),
 (0.9608464253791357, 0.3029165784443814),
 (0.9612545216244297, 0.30410315996526427),
 (0.9616569577292293, 0.30529657881783284),
 (0.9620538030343048, 0.30649691425526027),
 (0.9624804315118866, 0.3078143542258363),
 (0.9628658151550471, 0.30902941385896815),
 (0.9632801110282952, 0.3103631142649431),
 (0.9636881103349972, 0.3117054587504845),
 (0.9640899091651707, 0.31305656009627203),
 (0.9645183069814164, 0.3145302685311633),
 (0.9649075017817169, 0.31589998534744534),
 (0.9653224779095899, 0.3173941394097498),
 (0.9657306610292263, 0.31889915102882704),
 (0.9661321803554704, 0.3204151791574757),
 (0.9665572830594981, 0.3220603315075376),
 (0.9669453804567639, 0.3235997633667379),
 (0.9673563360182557, 0.3252705104053172),
 (0.9677602002707602, 0.3269548448445557),
 (0.9681852341976657, 0.3287748199947597),
 (0.9685749464453822, 0.33048801484182827),
 (0.9689852085123076, 0.33233944408293),
 (0.9693881387329771, 0.33420757314799165),
 (0.9698100882245698, 0.3362189937545381),
 (0.9702241944182711, 0.3382501405245402),
 (0.9706307156745811, 0.3403014042053906),
 (0.9710299106122553, 0.3423731872709509),
 (0.971446316968697, 0.34459740397805805),
 (0.9718550541467265, 0.3468457665431811),
 (0.9722564308833218, 0.3491188049749892),
 (0.9726737369463366, 0.35155305631696804),
 (0.9730834983375098, 0.3540162587921992),
 (0.9734860747468062, 0.35650910934340296),
 (0.9738818223293286, 0.3590323303865494),
 (0.9742925363298273, 0.3617295068189604),
 (0.9746964420022868, 0.36446227201260806),
 (0.9751146918106288, 0.367378361441084),
 (0.9755263016894108, 0.3703360958116499),
 (0.9759317205504938, 0.3733366818489754),
 (0.9763313860505544, 0.3763813794997551),
 (0.9767453081157705, 0.37962722829436657),
 (0.9771538329589436, 0.3829247568938137),
 (0.9775574204695061, 0.3862756376332149),
 (0.9779754120173848, 0.38984522080424994),
 (0.9783702476996005, 0.3933109178318859),
 (0.9787799759037069, 0.3970049436708181),
 (0.9791861352746015, 0.4007660540910947),
 (0.9796073882532946, 0.4047725472765047),
 (0.9800075050623989, 0.4086787582217678),
 (0.980423255679185, 0.4128428598436148),
 (0.9808188327090664, 0.41690581665674603),
 (0.9812305338401699, 0.4212404350391626),
 (0.9816406502548699, 0.42566771485407556),
 (0.9820494695680838, 0.430191704460561),
 (0.9824572497713367, 0.43481672346199657),
 (0.9828642204258444, 0.4395473874922171),
 (0.9832705841135677, 0.44438863589917466),
 (0.9836765181270832, 0.4493457627423146),
 (0.9840821763735244, 0.45442445159134615),
 (0.9845053211258176, 0.4598601744120041),
 (0.9849108076293412, 0.4652067915709929),
 (0.9853163652344854, 0.4706950980712935),
 (0.9857220759264133, 0.4763328085670776),
 (0.9861280099591841, 0.48212828545504993),
 (0.9865342282968508, 0.48809061348242394),
 (0.9869407850077634, 0.4942296854160182),
 (0.9873477295817633, 0.5005563008006252),
 (0.9877551091480568, 0.507082280282374),
 (0.988180715358901, 0.5141186011395664),
 (0.9885891314936663, 0.5210937416529832),
 (0.9889803371592701, 0.527992878980742),
 (0.989389952289551, 0.535460099280927),
 (0.9898002729013508, 0.5432066452513751),
 (0.9902113731352855, 0.5512542291320531),
 (0.9906233382268411, 0.5596271974591991),
 (0.9910362666808646, 0.5683529752428318),
 (0.9914322483611784, 0.5770580926654209),
 (0.991847409348529, 0.5865678864281907),
 (0.9922457886974023, 0.5960919424390858),
 (0.9926637759754662, 0.6065405821646188),
 (0.993065215406563, 0.6170536982964379),
 (0.9934868390015257, 0.6286469166315802),
 (0.9938922466963461, 0.6403783844945216),
 (0.9943000243543926, 0.6528144710471044),
 (0.994710519555982, 0.666045257568372),
 (0.9951052614592826, 0.6795154667361197),
 (0.9955222897068845, 0.6946348004319254),
 (0.9959241609946798, 0.7101754583858633),
 (0.9963302320906318, 0.7269774985228425),
 (0.9967409716074388, 0.7452639179514023),
 (0.9971567328192074, 0.7653226332347454),
 (0.9975573858065462, 0.7864209093867824),
 (0.9979619597970514, 0.8099147885553253),
 (0.9983683891380578, 0.8364188081105579),
 (0.9987916298728423, 0.8684630018343936),
 (0.999198672672967, 0.9063889338931731),
 (0.9995946244774474, 0.9578734354807452),
 (1, 1.0)
]

def getMultiplierForAccScale(acc):
    if acc > 1:
        return 1

    previousCurvePointAcc = 0
    previousCurvePointMultiplier = 0
    
    for curvePointAcc, curvePointMultiplier in scaleCurve:
        if acc <= curvePointAcc:
            accDiff = (curvePointAcc - previousCurvePointAcc)
            multiplierDiff = (curvePointMultiplier - previousCurvePointMultiplier)
            slope = multiplierDiff/accDiff
            multiplier = previousCurvePointMultiplier + slope * (acc - previousCurvePointAcc)
            return multiplier
        
        previousCurvePointAcc = curvePointAcc
        previousCurvePointMultiplier = curvePointMultiplier
        

def getAccForMultiplierScale(multiplier):

    previousCurvePointMultiplier = 0
    previousCurvePointAcc = 0
    
    for curvePointAcc, curvePointMultiplier in scaleCurve:
        if multiplier <= curvePointMultiplier:
            multDiff = (curvePointMultiplier - previousCurvePointMultiplier)
            accDiff = (curvePointAcc - previousCurvePointAcc)
            slope = accDiff/multDiff
            acc = previousCurvePointAcc + slope * (multiplier - previousCurvePointMultiplier)
            return acc
        
        previousCurvePointMultiplier = curvePointMultiplier
        previousCurvePointAcc = curvePointAcc


def scaleFarmability(acc, noteCount, mapLength, farmSessionLength=60*60):
    base_attempts_count = 30
    base_multiplier = 0.030963633
    base_note_count = 200
    note_scale = noteCount/base_note_count
    attempts_scale = (farmSessionLength/mapLength)/base_attempts_count
    note_scaler = 1/(1+5*(note_scale**0.69))*6.9

    attempts_scaler = ((math.log(attempts_scale)+2.7081502061025433)**0.69/2.0)
    multiplier = getMultiplierForAccScale(acc) + note_scaler*attempts_scaler*base_multiplier*(min(1, base_note_count * 10/noteCount))
    return getAccForMultiplierScale(multiplier)
    
    
    base_map_length = 60/map_length_scale
    attempts_scale = base_map_length/mapLength
    note_scale = (noteCount)/base_note_count
    note_scaler = (((math.log(note_scale, 0.05)+0.46275642631951835)**2+0.2431278387816179)/2.2641367447629013)/0.20
    attempts_scaler = ((math.log(attempts_scale)+2.7081502061025433)**0.75/2.1110793685981553)
    multiplier = getMultiplierForAccScale(acc) + base_multiplier*(note_scaler*attempts_scaler)*(min(1, base_note_count * 2/noteCount))
    return getAccForMultiplierScale(multiplier)


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
    window_size, top_window, skip_window = (50, 12, 5)
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

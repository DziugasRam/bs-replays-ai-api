from http_client import download_map
import numpy as np
import glob
import json


maps_dir = "/home/maps"


def read_json_file(file):
    try:
        with open(file, "r", encoding="utf8", errors="ignore") as f:
            file_content = f.read()
            if len(file_content) < 100:
                return None
            json_content = json.loads(file_content)
            return json_content
    except Exception as e:
        print(e)
        print(file)


def preprocess_map_notes(map_notes, njs, time_scale):
    notes = []
    note_times = []

    prev_zero_note_time = 0
    prev_one_note_time = 0

    for note_time, note_info in map_notes:
        type = note_info[-1]

        delta_to_zero = note_time - prev_zero_note_time
        delta_to_one = note_time - prev_one_note_time

        if delta_to_zero < 0 or delta_to_one < 0:
            print(f"{delta_to_zero} {delta_to_one}")

        if type == "0":
            prev_zero_note_time = note_time
            note = preprocess_note(delta_to_zero, delta_to_one, note_info, njs, time_scale)
            notes.append(note)
            note_times.append(note_time)
        if type == "1":
            prev_one_note_time = note_time
            note = preprocess_note(delta_to_one, delta_to_zero, note_info, njs, time_scale)
            notes.append(note)
            note_times.append(note_time)

    return notes, note_times


def preprocess_note(delta, delta_other, note_info, njs, time_scale):
    delta = delta/time_scale
    delta_other = delta_other/time_scale
    njs = njs*time_scale

    delta_long = max(0, 2 - delta)/2
    delta_other_long = max(0, 2 - delta_other)/2
    delta_short = max(0, 0.5 - delta)*2
    delta_other_short = max(0, 0.5 - delta_other)*2

    col_number = int(note_info[0])
    row_number = int(note_info[1])
    direction_number = int(note_info[2])
    color = int(note_info[3])

    row_col = [0] * 4 * 3
    direction = [0] * 10
    
    row_col2 = [0] * 4 * 3
    direction2 = [0] * 10
    
    row_col[col_number * 3 + row_number] = 1
    direction[direction_number] = 1

    # color_arr = [0] * 2
    # color_arr[color] = 1

    response = []

    if color == 0:
        response.extend(row_col)
        response.extend(direction)
        response.extend(row_col2)
        response.extend(direction2)
        response.extend([
            delta_short,
            delta_long,
        ])
        response.extend([
            delta_other_short,
            delta_other_long,
        ])
    if color == 1:
        response.extend(row_col2)
        response.extend(direction2)
        response.extend(row_col)
        response.extend(direction)
        response.extend([
            delta_other_short,
            delta_other_long,
        ])
        response.extend([
            delta_short,
            delta_long,
        ])
        
    # response.extend(row_col)
    # response.extend(direction)
    # response.extend(color_arr)

    response.extend([
        njs/30
    ])

    return response


def create_segments(notes):
    empty_res = ([], [])
    if len(notes) < prediction_size:
        return empty_res

    segments = []
    for i in range(len(notes)-prediction_size+1):
        if i % prediction_size != 0:
            continue

        pre_slice = notes[max(0, i-pre_segment_size):i]
        slice = notes[i:i+prediction_size]
        post_slice = notes[i+prediction_size:i +
                           prediction_size+post_segment_size]

        # NOTE: using relative score can be good to find relative difficulty of the notes more fairly
        # because good players will always get higher acc and worse players will do badly even on easy patterns

        pre_segment = [np.array(note) for note in pre_slice]
        if len(pre_segment) < pre_segment_size:
            pre_segment[0:0] = [np.zeros(note_size, dtype=np.float32) for i in range(
                pre_segment_size - len(pre_segment))]

        segment = [np.array(note) for note in slice]

        post_segment = [np.array(note) for note in post_slice]
        if len(post_segment) < post_segment_size:
            post_segment.extend([np.zeros(note_size, dtype=np.float32)
                                for i in range(post_segment_size - len(post_segment))])


        final_segment = []
        final_segment.extend(pre_segment)
        final_segment.extend(segment)
        final_segment.extend(post_segment)
        segments.append(final_segment)

    return segments


pre_segment_size = 12
post_segment_size = 12
prediction_size = 8
note_size = 49

segment_size = pre_segment_size + post_segment_size + prediction_size


direction_to_angle = {
    0: 180,
    1: 0,
    2: 90,
    3: 270,
    4: 135,
    5: 225,
    6: 45,
    7: 315
}

angle_to_direction = {
    180:0,
    0:1,
    90:2,
    270:3,
    135:4,
    225:5,
    45:6,
    315:7
}

def get_note_direction(direction, angle):
    if direction == 8:
        return 8
    
    note_angle = (direction_to_angle[direction] - round(angle/45)*45) % 360
    return angle_to_direction[note_angle]


def get_map_notes_from_json(map_json, bpm_time_scale):
    if "version" in map_json and map_json["version"].split(".")[0] == "3":
        map_notes = sorted(list(map(lambda n: (n["b"]*bpm_time_scale, f"{n['x']}{n['y']}{get_note_direction(n['d'], n['a'])}{n['c']}"), filter(
                            lambda n: n['c'] == 1 or n['c'] == 0, map_json["colorNotes"]))), key=lambda x: (x[0], x[1]))
    else:
        map_notes = sorted(list(map(lambda n: (n["_time"]*bpm_time_scale, f"{n['_lineIndex']}{n['_lineLayer']}{n['_cutDirection']}{n['_type']}"), filter(
                            lambda n: n['_type'] == 1 or n['_type'] == 0, map_json["_notes"]))), key=lambda x: (x[0], x[1]))
    return map_notes


def get_free_points_for_map(map_json):
    if "version" in map_json and map_json["version"].split(".")[0] == "3" and map_json["burstSliders"]:
        segment_count = sum([burst_slider["sc"] for burst_slider in map_json["burstSliders"]])
        return segment_count*20*8
    else:
        return 0


def get_map_data(hash, characteristic, difficulty):
    if characteristic is None:
        characteristic = "Standard"

    map_info_files = []
    map_info_files.extend(glob.glob(f'{maps_dir}/{hash}/Info.dat'))
    map_info_files.extend(glob.glob(f'{maps_dir}/{hash}/info.dat'))
    map_info_file = map_info_files[0]
    njs = None
    map_notes = None
    songName = None

    with open(map_info_file, "r", encoding="utf8", errors="ignore") as f:
        file_content = f.read()
        map_info = json.loads(file_content)
        bpm = map_info["_beatsPerMinute"]
        bpm_time_scale = 60/bpm
        songName = map_info["_songName"]

        for beatmap_set in map_info["_difficultyBeatmapSets"]:
            if beatmap_set["_beatmapCharacteristicName"] != characteristic:
                continue

            for beatmap in beatmap_set["_difficultyBeatmaps"]:
                if beatmap["_difficultyRank"] == difficulty:
                    njs = float(beatmap["_noteJumpMovementSpeed"])
                    map_file_name = beatmap["_beatmapFilename"]
                    with open(map_info_file.replace("Info.dat", map_file_name).replace("info.dat", map_file_name), "r", encoding="utf8", errors="ignore") as map_file:
                        map_file_content = map_file.read()
                        map_json = json.loads(map_file_content)
                        map_notes = get_map_notes_from_json(map_json, bpm_time_scale)
                        free_points = get_free_points_for_map(map_json)

    return njs, map_notes, songName, free_points


def preprocess_map(hash, characteristic, difficulty, time_scale):
    download_map(hash)
    
    empty_response = ([], [], "", [])
    njs, map_notes, songName, free_points = get_map_data(hash, characteristic, difficulty)
    if njs == None or map_notes == None:
        return empty_response

    notes, note_times = preprocess_map_notes(map_notes, njs, time_scale)
    segments = create_segments(notes)
    return segments, songName, note_times, free_points


def get_map_info(hash, characteristic, difficulty):
    download_map(hash)

    map_info_files = []
    map_info_files.extend(glob.glob(f'{maps_dir}/{hash}/Info.dat'))
    map_info_files.extend(glob.glob(f'{maps_dir}/{hash}/info.dat'))
    map_info_file = map_info_files[0]

    with open(map_info_file, "r", encoding="utf8", errors="ignore") as f:
        file_content = f.read()
        map_info = json.loads(file_content)
        bpm = map_info["_beatsPerMinute"]

        for beatmap_set in map_info["_difficultyBeatmapSets"]:
            if beatmap_set["_beatmapCharacteristicName"] != characteristic:
                continue

            for beatmap in beatmap_set["_difficultyBeatmaps"]:
                if beatmap["_difficultyRank"] == difficulty:
                    map_file_name = beatmap["_beatmapFilename"]
                    with open(map_info_file.replace("Info.dat", map_file_name).replace("info.dat", map_file_name), "r", encoding="utf8", errors="ignore") as map_file:
                        map_file_content = map_file.read()
                        return { "map_json": json.loads(map_file_content), "bpm": bpm }

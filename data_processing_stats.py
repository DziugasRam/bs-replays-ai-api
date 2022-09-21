import os
import glob
import json
import requests

from io import BytesIO
import urllib
from zipfile import ZipFile

from http_client import download_map


maps_dir = "/home/maps"


def get_map_note_counts(hash, mode, diff):
    map_info_files = []
    map_info_files.extend(glob.glob(f'{maps_dir}/{hash}/Info.dat'))
    map_info_files.extend(glob.glob(f'{maps_dir}/{hash}/info.dat'))
    map_info_file = map_info_files[0]
    map_notes = None
    songName = None

    with open(map_info_file, "r", encoding="utf8", errors="ignore") as f:
        file_content = f.read()
        map_info = json.loads(file_content)
        songName = map_info["_songName"]

        for beatmap_set in map_info["_difficultyBeatmapSets"]:
            if beatmap_set["_beatmapCharacteristicName"] != mode:
                continue

            for beatmap in beatmap_set["_difficultyBeatmaps"]:
                if beatmap["_difficultyRank"] == diff:
                    map_file_name = beatmap["_beatmapFilename"]
                    with open(map_info_file.replace("Info.dat", map_file_name), "r", encoding="utf8", errors="ignore") as map_file:
                        map_file_content = map_file.read()
                        map_file_json = json.loads(map_file_content)
                        map_notes = map_file_json["_notes"]
                        zero_notes_count = len(
                            list(filter(lambda n: int(n["_type"]) == 0, map_notes)))
                        one_notes_count = len(
                            list(filter(lambda n: int(n["_type"]) == 1, map_notes)))
                        return zero_notes_count, one_notes_count, songName

    return 0, 0, songName
    

def get_map_stats(hash, mode, difficulty):
    download_map(hash)

    zero_notes_count, one_notes_count = get_map_note_counts(
        hash, mode, difficulty)
    return {
        "zero_notes_count": zero_notes_count,
        "one_notes_count": one_notes_count,
    }


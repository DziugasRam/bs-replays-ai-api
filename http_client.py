import os
import glob
import json
import requests

from io import BytesIO
import urllib
from zipfile import ZipFile


maps_dir = "/home/maps"

def download_map(hash):
    map_dir = f"{maps_dir}/{hash}"

    if os.path.exists(map_dir):
        return

    beatsaver_url = f"https://beatsaver.com/api/maps/hash/{hash}"
    r = requests.get(url = beatsaver_url)
    beatsaver_data = r.json()
    downloadURL = ""
    
    for version in beatsaver_data["versions"]:
        if version["hash"] == hash:
            downloadURL = version["downloadURL"]
    
    if downloadURL == "":
        raise Exception()
    
    req = urllib.request.Request(downloadURL, headers={'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"})
    with urllib.request.urlopen(req) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(map_dir)
            song_file = glob.glob(f"{map_dir}/*gg")[0]
            try:
                os.remove(song_file)
            except:
                return

def get_bl_leadeboard(hash, diff, mode):
    bl_url = f"https://api.beatleader.xyz/leaderboards/hash/{hash}"
    r = requests.get(url = bl_url)
    bl_data = r.json()
    for difficulty in bl_data['song']['difficulties']:
        if difficulty['difficultyName'] == diff and difficulty['modeName'].replace("Solo", "") == mode.replace("Solo", ""):
            song_mode = difficulty['mode']
            value = difficulty['value']
            song_id = bl_data['song']['id']
            return f"https://www.beatleader.xyz/leaderboard/global/{song_id}{value}{song_mode}"
            
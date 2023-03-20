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
            extracted_files = glob.glob(f"{map_dir}/*")
            for extracted_file in extracted_files:
                if extracted_file.endswith(".dat"):
                    continue
                else:
                    try:
                        os.remove(extracted_file)
                    except:
                        continue

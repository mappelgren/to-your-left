import json
import sys
from glob import glob

scenes = []
for file in glob(sys.argv[1] + '*', recursive=True):
    with open(file, 'r', encoding='utf-8') as f:
        scenes.append(json.load(f))

with open(sys.argv[2], 'w', encoding='utf-8') as f:
    json.dump({
        'scenes': scenes
    }, f)

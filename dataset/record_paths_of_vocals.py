import json
import os

vocal_paths = "val_seperation_results"

record_path = "val_vocal.json"

data = []

for sources in os.listdir(vocal_paths):
    print(f"processing {sources}")
    for source in os.listdir(os.path.join(vocal_paths, sources)):
        print(f"  checking {source}")
        if source == "vocals.wav":
            print(f"loading {source} from {sources}")
            path = os.path.join("./dataset/", vocal_paths, sources, source)
            data.append(path)

with open(record_path, "w") as f:
    json.dump(data, f, indent=4)
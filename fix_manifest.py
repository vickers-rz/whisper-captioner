import json

path = "/Users/vickers/Documents/temp/batch_asr_staging/manifest.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for slug, info in data.items():
    if "02 docker下的AI开发与大模型部署.mp4" in info["mp4"]:
        info["srt_done"] = False
        print("Set srt_done to False for:", info["mp4"])

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

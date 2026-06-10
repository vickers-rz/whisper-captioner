import json
path = "/Users/vickers/Documents/temp/batch_asr_staging/_Volumes_02_HDD_unTar_NFS_华清元宇宙_yyzlab_05_第五阶段_进阶实战-大模型实战应用_02_大模型的部署与应用基础_11__环境_Ubuntu下CUDA和cuDNN安装-Docker_02_docker下的AI开发与大模型部署.mp4.json"
with open(path, "r") as f:
    data = json.load(f)
for i, seg in enumerate(data["segments"][:25]):
    print(f"{i}: {seg['text']}")

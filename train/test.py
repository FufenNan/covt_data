import json
import os
data_path = "/u/yli8/jiaqi/CoVT-Dataset/data.json"
# data_path_2 = "/u/yli8/jiaqi/CoVT-Dataset/data_81.json"  # 改成你的第二个文件
data1 = json.load(open(data_path, "r"))
out_path = "/u/yli8/jiaqi/CoVT-Dataset/data_new.json"
with open(data_path, "r") as f:
    data = json.load(f)

# 修改 image 字段
for item in data:
    img = item["image"]
    item["image"] = os.path.join("/u/yli8/jiaqi/CoVT-Dataset/images", img)

# 写入新文件
with open(out_path, "w") as f:
    json.dump(data, f)
    
# data2 = json.load(open(data_path_2, "r"))
# merged = data1 + data2
# out_file = "/u/yli8/jiaqi/CoVT-Dataset/data.json"
# with open(out_file, "w") as f:
#     json.dump(merged, f)
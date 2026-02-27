import pandas as pd
import json
from pathlib import Path

# 输入输出路径
dataset_folder = "/work/nvme/bdqf/yli8/CoVT-Dataset/part5"  # parquet 文件夹
image_folder = "/work/nvme/bdqf/yli8/CoVT-Dataset/images"
output_json = "/work/nvme/bdqf/yli8/CoVT-Dataset/llava_format.json"

all_entries = []

# 遍历 parquet 文件
for parquet_file in Path(dataset_folder).glob("*.parquet"):
    print(f"Processing {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    for _, row in df.iterrows():
        entry_id = str(row["id"])
        image_file = f"{entry_id}.jpg"  # 确保和 image_folder 中一致
        
        conversations = []
        for i, (q, a) in enumerate(zip(row["questions"], row["answers"])):
            # human 的第一个问题加 <image>
            human_text = q
            if i == 0:
                human_text = "<image>\n" + human_text
            
            conversations.append({"from": "human", "value": human_text})
            conversations.append({"from": "gpt", "value": a})
        
        all_entries.append({
            "id": entry_id,
            "image": image_file,
            "conversations": conversations
        })

# 保存为 JSON 文件
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(all_entries, f, ensure_ascii=False, indent=2)

print(f"Converted {len(all_entries)} entries to LLaVA format and saved to {output_json}")

from transformers import AutoProcessor, AutoModel
path = "/u/yli8/jiaqi/CoVT/train/output/lora_merged/lora_stage234_merged"
AutoProcessor.from_pretrained(path)
AutoModel.from_pretrained("/u/yli8/jiaqi/CoVT/train/output/lora_merged/lora_stage234_merged")
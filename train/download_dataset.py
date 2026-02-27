from huggingface_hub import snapshot_download

print("开始下载数据集...")
snapshot_download(
    repo_id="Wakals/CoVT-Dataset",
    repo_type="dataset",
    revision="main",
    local_dir="/u/yli8/jiaqi/CoVT-Dataset",
    local_dir_use_symlinks=False
)
print("下载完成！")
import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, HfArgumentParser
from training.trainer import QwenTrainer, UnfreezeLoRACallback, ResumeDatasetCallback
from training.data import make_supervised_data_module,ImagePathDataset
from training.params import DataArguments, ModelArguments, TrainingArguments
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward

from training.covt_qwen2_5_vl import CoVTForConditionalGeneration,AnchorOutputExtractor
from training.constants import *
from deepspeed import zero
from PIL import Image

local_rank = None

# set seed 42
torch.manual_seed(42)

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad
        
def set_anchor_requires_grad(model, anchor_model_id):
    # set_requires_grad(model.sam_projection.parameters(), True)
    # set_requires_grad(model.dino_projection.parameters(), True)
    # set_requires_grad(model.depth_projection.parameters(), True)
    # set_requires_grad(model.SD_projection.parameters(), True)
    # set_requires_grad(model.internvit_projection.parameters(), True)
    # set_requires_grad(model.pidinet_projection.parameters(), True)
    # set_requires_grad(model.siglip_projection.parameters(), True)
    # set_requires_grad(model.metaclip_projection.parameters(), True)
    
    # set_requires_grad(model.sam_cross_attention.parameters(), True)
    # set_requires_grad(model.dino_cross_attention.parameters(), True)
    # set_requires_grad(model.depth_cross_attention.parameters(), True)
    # set_requires_grad(model.SD_cross_attention.parameters(), True)
    # set_requires_grad(model.internvit_cross_attention.parameters(), True)
    # set_requires_grad(model.pidinet_cross_attention.parameters(), True)
    # set_requires_grad(model.siglip_cross_attention.parameters(), True)
    # set_requires_grad(model.metaclip_cross_attention.parameters(), True)

    # model.dino_query_vectors.requires_grad = True
    # model.sam_query_vectors.requires_grad = True
    # model.depth_query_vectors.requires_grad = True
    # model.SD_query_vectors.requires_grad = True
    # model.internvit_query_vectors.requires_grad = True
    # model.pidinet_query_vectors.requires_grad = True
    # model.siglip_query_vectors.requires_grad = True
    # model.metaclip_query_vectors.requires_grad = True
    
    if "sam" in anchor_model_id:
        set_requires_grad(model.sam_projection.parameters(), True)
        set_requires_grad(model.sam_cross_attention.parameters(), True)
        model.sam_query_vectors.requires_grad = True
    if "dino" in anchor_model_id:
        set_requires_grad(model.dino_projection.parameters(), True)
        set_requires_grad(model.dino_cross_attention.parameters(), True)
        model.dino_query_vectors.requires_grad = True
    if "depth" in anchor_model_id:
        set_requires_grad(model.depth_projection.parameters(), True)
        set_requires_grad(model.depth_cross_attention.parameters(), True)
        model.depth_query_vectors.requires_grad = True
        set_requires_grad(model.depth_token_generator.parameters(), True)
    if "sd" in anchor_model_id:
        set_requires_grad(model.SD_projection.parameters(), True)
        set_requires_grad(model.SD_cross_attention.parameters(), True)
        model.SD_query_vectors.requires_grad = True
    if "internvit" in anchor_model_id:
        set_requires_grad(model.internvit_projection.parameters(), True)
        set_requires_grad(model.internvit_cross_attention.parameters(), True)
        model.internvit_query_vectors.requires_grad = True
    if "pidinet" in anchor_model_id:
        set_requires_grad(model.pidinet_projection.parameters(), True)
        set_requires_grad(model.pidinet_cross_attention.parameters(), True)
        model.pidinet_query_vectors.requires_grad = True
    if "siglip" in anchor_model_id:
        set_requires_grad(model.siglip_projection.parameters(), True)
        set_requires_grad(model.siglip_cross_attention.parameters(), True)
        model.siglip_query_vectors.requires_grad = True
    if "metaclip" in anchor_model_id:
        set_requires_grad(model.metaclip_projection.parameters(), True)
        set_requires_grad(model.metaclip_cross_attention.parameters(), True)
        model.metaclip_query_vectors.requires_grad = True

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, training_args.tune_merger)
    
def configure_llava_vision_tower(model, model_args, training_args, compute_dtype, processor):
    model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
    )
    
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    model.config.image_aspect_ratio = "pad"
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = False
    model.config.mm_projector_lr = 2e-5
    training_args.use_im_start_end = False
    model.config.mm_use_im_patch_token = False
    model.initialize_vision_tokenizer(model_args, tokenizer=processor.tokenizer)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    anchor_model_id = ast.literal_eval(model_args.anchor_model_id)
    
    if model_args.model_path is None:
        print("\033[91mWARNING: model_path is not provided, using model_id instead\033[0m")
        model_args.model_path = model_args.model_id
    
    # Liger-kernel for Qwen2.5 is not supported yet.
    replace_qwen2_5_with_mixed_modality_forward(use_liger=training_args.use_liger)\
    

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))
    
    # Create AnchorOutputExtractor for feature extraction
    anchor_extractor = AnchorOutputExtractor(anchor_model_id=anchor_model_id)
    dataset = ImagePathDataset(data_path=data_args.data_path, data_args=data_args, anchor_model_id=anchor_model_id)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=2, collate_fn=lambda x: x)
    
    rank0_print("Starting anchor model feature extraction...")
    
    # Extract all unique image paths from all datasets
    image_paths = set()
    
    for batch_idx, batch in enumerate(dataloader):
        for data_point in batch:
            if 'image' in data_point:
                images = data_point["image"]

                if isinstance(images, str):
                    image_paths.add(images)
                elif isinstance(images, list):
                    for img in images:
                        if isinstance(img, str):
                            image_paths.add(img)
        
        # Show progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            rank0_print(f"Processed {batch_idx + 1} data batches, found {len(image_paths)} unique images")

    rank0_print(f"Dataset scanning complete. Found {len(image_paths)} unique images to process")

    if len(image_paths) > 0:
        # Create output directory
        anchor_output_dir = training_args.output_dir
        os.makedirs(anchor_output_dir, exist_ok=True)
        
        # Convert set to sorted list for consistent processing order
        image_paths = sorted(list(image_paths))
        
        # Determine optimal batch size dynamically
        batch_size = 4096
        rank0_print(f"Using batch size: {batch_size} for processing {len(image_paths)} images")
        
        # Process images in batches with proper error handling and progress tracking                
        anchor_extractor.batch_process_images(
            image_paths,
            save_dir=anchor_output_dir,
            batch_size=batch_size,
            save_original=False,
            save_features=False,
        )
    
    else:
        rank0_print("No images found in dataset for anchor model processing")


if __name__ == "__main__":
    train()
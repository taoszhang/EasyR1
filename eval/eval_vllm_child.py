import os
import re
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration, CLIPModel, CLIPProcessor
from data.mbeir_data_utils import build_mbeir_dataset_from_config, DatasetType, MBEIRDataConfig
from torch.utils.data import DataLoader, DistributedSampler
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
import numpy as np
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch.nn as nn
from unittest.mock import patch
from qwen_vl_utils import process_vision_info
import csv
from gme_inference import GmeQwen2VL
from vllm import LLM, SamplingParams
import torch.nn.functional as F
from data.child_dataset import ChildDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def extract_gme_features(model, data_loader, device,rank):
    features = []
    ids = []
    print("Extracting CLIP features...")
    with torch.no_grad():
        for index,batch in enumerate(tqdm(data_loader)):
            images=[]
            for img,id in zip(batch['img'],batch["did"]):
                image = Image.open(img).convert("RGB")
                ids.append(id)
                images.append(image)
            e_image = model.get_image_embeddings(images=images).to(device)
            features.extend([torch.tensor(feat) for feat in e_image.tolist()])
    with open(f"/data/sxli/projects/open-r1-multimodal/datasets/M-BEIR/features/gme_cirr_{rank}.jsonl",'a') as file:
        for id,feature in zip(ids,features):
            file.write(f"{id}\t{feature.tolist()}\n")
    features = torch.stack(features).to(device)
    return features, ids


def calculate_recall_from_file(file_path, true_indices, q_ids, candidate_ids, k=10):
    recall = 0
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid = int(parts[0])
            top_k_entries = parts[2:]
            top_k_candidate_ids = [int(entry.split(':')[0]) for entry in top_k_entries]
            if qid in q_ids:
                true_index = true_indices[q_ids == qid][0]
                if true_index in top_k_candidate_ids[:k]:
                    recall += 1
    return recall / len(true_indices)
def identity_collate(batch):
    return batch  # 直接返回原始列表，不进行任何处理
def main(rank, world_size):
    setup(rank, world_size)

    # Load configurations
    query_data_path = "/data/sxli/projects/open-r1-multimodal/datasets/Child/val/val.jsonl"
    image_pool_path = "/data/sxli/projects/open-r1-multimodal/datasets/Child/images/"
    # Load models
    # Build datasets
    val_dataset=ChildDataset(query_data_path,image_pool_path)

    qwen2vl_model="/data/sxli/projects/open-r1-multimodal/checkpoints/Qwen2.5-VL-7B-GRPO"
    query_batch_size = 1
    torch.cuda.set_device(rank)
    llm = LLM(
        model=qwen2vl_model,
        device=f"cuda:{rank}",
        gpu_memory_utilization=0.9,
        dtype=torch.bfloat16,
        tensor_parallel_size=1,
        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
        # This is particularly useful here because we generate completions from the same prompts.
        enable_prefix_caching=True,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=1,
        max_tokens=2048,
    )
    processing_class=AutoProcessor.from_pretrained(qwen2vl_model)
    query_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    query_data_loader = DataLoader(val_dataset, batch_size=1, sampler=query_sampler,collate_fn=identity_collate)
    # Create an iterator for the query data loader
    query_iter = iter(query_data_loader)
    # Process queries in batches
    with open(f"/data/sxli/projects/open-r1-multimodal/eval_records/child/{rank}.jsonl",'w') as wfile:
        with tqdm(total=len(query_data_loader), desc="Processing Queries") as pbar:  # 添加进度条
            while True:
                # Process each query in the current batch
                for _ in range(query_batch_size):
                    try:
                        batch = next(query_iter)
                    except StopIteration:
                        break
                    image_data, _ = process_vision_info(batch[0]["prompt"])
                    toys=batch[0]["toys"]
                    text = processing_class.apply_chat_template(batch[0]["prompt"], tokenize=False, add_generation_prompt=True)
                    all_multimodal_inputs=[{"prompt": text,"multi_modal_data": {"image": image_data},}]
                    outputs = llm.generate(
                            all_multimodal_inputs,
                            sampling_params=sampling_params,
                            use_tqdm=False,
                        )
                    generations = [output.text for content in outputs for output in content.outputs]
                    for generation in generations:
                        nums=0
                        for toy in toys:
                            for t in toy:
                                if t in generation:
                                    nums+=1
                                    break
                        reward=nums/len(toys)
                        wfile.write(f"{generation}\t{reward}\n")
                # Append batch results to the main lists
                pbar.update(query_batch_size)  # 更新查询处理进度
    cleanup()
if __name__ == "__main__":
    world_size = 8
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
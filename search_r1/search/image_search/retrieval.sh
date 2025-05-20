#!/bin/bash

# 设置可见 GPU 设备
export CUDA_VISIBLE_DEVICES=4,5

# 输入参数
MODEL_PATH="ViT-L/14@336px"
INDEX_PATH="/data/tzhang/project/Infoseek_multi_hop/search_engine/image/CLIP_Flat.index"
TITLE_PATH="/data/tzhang/project/Infoseek_multi_hop/search_engine/image/CLIP_Flat_titles.txt"
QUERY_LIST="/data/tzhang/project/EasyR1/search_r1/search/image_search/test.txt"  # 一行一个图像路径
TOPK=10

# 执行检索脚本（batch 模式）
python retrieval.py \
  --model_path "$MODEL_PATH" \
  --index_path "$INDEX_PATH" \
  --title_path "$TITLE_PATH" \
  --image_path "$QUERY_LIST" \
  --topk $TOPK \
  --batch

#!/bin/bash
set -x
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_PATH=/data/project/zyma/tzhang/model/Qwen2.5-VL-7B-Instruct
experiment_name=multi_hop_bridge_ret_img_clip_right_1_compare_part_1_search_time+4*accuracy+4*format
save_path=/data/project/zyma/tzhang/outputs/EasyR1/checkpoints/multi_hop_bridge_ret_img_clip_right_1_compare_part_1_search_time+4*accuracy+4*format
load_checkpoint_path=
mkdir -p ${save_path}

python3 -m verl.trainer.main \
    config=examples/config_dapo.yaml \
    data.train_files=/data/project/zyma/project/Infoseek_multi_hop/infoseek_multi_hop/parquet/bridge_ret_img_clip_right_1_compare_part_1/train \
    data.val_files=/data/project/zyma/project/Infoseek_multi_hop/infoseek_multi_hop/parquet/bridge_ret_img_clip_right_1_compare_part_1/validation \
    data.max_prompt_length=6144 \
    data.max_response_length=512 \
    data.max_start_length=2048 \
    data.max_obs_length=1024 \
    data.max_end_length=4096 \
    data.mini_rollout_batch_size=128 \
    data.val_batch_size=256 \
    data.format_prompt=./examples/format_prompt/search_r1_compare_image.jinja \
    worker.actor.global_batch_size=64 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.n=8 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=${save_path} \
    trainer.load_checkpoint_path=${load_checkpoint_path} \
    trainer.val_freq=5 \
    trainer.val_before_train=true \
    trainer.val_generations_to_log=20 \
    trainer.save_freq=5 \
    retriever.url=http://127.0.0.1:8000/retrieve \
    retriever.topk=2
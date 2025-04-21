set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_PATH=/data/tzhang/model/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
experiment_name=acc_only_debug
save_path=/data/tzhang/outputs/EasyR1/checkpoints/temp
load_checkpoint_path=/data/tzhang/outputs/EasyR1/checkpoints/infoseek_bridge_acc_reward/global_step_160
mkdir -p ${save_path}

SYSTEM_PROMPT=""""""

python3 -m verl.trainer.main \
    config=examples/config_test.yaml \
    data.train_files=/data/tzhang/dataset/infoseek_bridge/infoseek_bridge_train \
    data.val_files=/data/tzhang/dataset/infoseek_bridge/infoseek_bridge_validation \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.max_start_length=2048 \
    data.max_obs_length=512 \
    data.max_end_length=2048 \
    data.rollout_batch_size=256 \
    data.val_batch_size=512 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=${save_path} \
    trainer.load_checkpoint_path=${load_checkpoint_path} \
    trainer.val_freq=10 \
    trainer.val_before_train=true \
    trainer.save_freq=40 \
    retriever.url=http://127.0.0.1:8000/retrieve \
    retriever.topk=2 
    
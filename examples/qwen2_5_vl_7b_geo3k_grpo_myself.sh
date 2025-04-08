set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_PATH=/data/tzhang/model/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

SYSTEM_PROMPT=""""""

python3 -m verl.trainer.main \
    config=examples/config_test.yaml \
    data.train_files=/data/tzhang/dataset/infoseek_bridge/infoseek_bridge_train \
    data.val_files=/data/tzhang/dataset/infoseek_bridge/infoseek_bridge_test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_infoseek_bridge_grpo \
    trainer.n_gpus_per_node=8 \
    retriever.url=http://127.0.0.1:8000/retrieve \
    retriever.topk=2 
    
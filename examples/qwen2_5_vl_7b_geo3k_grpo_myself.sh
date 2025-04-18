set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_PATH=/data/tzhang/model/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=examples/config_test.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.n_gpus_per_node=4 \
    retriever.url=http://127.0.0.1:8000/retrieve \
    retriever.topk=3 
    
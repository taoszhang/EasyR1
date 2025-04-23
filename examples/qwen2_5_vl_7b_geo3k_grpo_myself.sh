set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_PATH=/data/tzhang/model/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
experiment_name=acc_only_prompt_new_rollout_8
save_path=/data/tzhang/outputs/EasyR1/checkpoints/infoseek_bridge_acc_reward_prompt_new_rollout_8
load_checkpoint_path=
mkdir -p ${save_path}

SYSTEM_PROMPT="""<image>\nAnswer the given question, you should break down the question and perform searches multiple times, step by step.
  You must conduct reasoning inside <think> and </think> first every time you get start.
  After reasoning, if you find that you lack certain knowledge or are unsure about specific knowledge, you can use the <search>query</search> format to call a search engine.
  It will return the top search results between the <information> and </information> tags.
  If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> tags, using a single word or phrase.
  For example, <answer>Beijing</answer>."""

python3 -m verl.trainer.main \
    config=examples/config_test.yaml \
    data.train_files=/data/tzhang/dataset/infoseek_bridge/infoseek_bridge_with_entity/infoseek_bridge_train \
    data.val_files=/data/tzhang/dataset/infoseek_bridge/infoseek_bridge_with_entity/infoseek_bridge_validation \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.max_start_length=2048 \
    data.max_obs_length=512 \
    data.max_end_length=2048 \
    data.rollout_batch_size=128 \
    data.val_batch_size=256 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.gpu_memory_utilization=0.8 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.n=8 \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=${save_path} \
    trainer.load_checkpoint_path=${load_checkpoint_path} \
    trainer.val_freq=10 \
    trainer.val_before_train=false \
    trainer.val_generations_to_log=20 \
    trainer.save_freq=10 \
    retriever.url=http://127.0.0.1:8000/retrieve \
    retriever.topk=2 
    
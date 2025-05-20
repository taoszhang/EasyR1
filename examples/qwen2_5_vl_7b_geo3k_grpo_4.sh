set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_PATH=/data/tzhang/model/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
experiment_name=accuracy+format_new_data_debug
save_path=/data/tzhang/outputs/EasyR1/checkpoints_format_num_redo/accuracy+format_new_data_debug
load_checkpoint_path=/data/tzhang/outputs/EasyR1/checkpoints_format_num_redo/accuracy+format_new_data_debug/global_step_40
mkdir -p ${save_path}

SYSTEM_PROMPT="""<image>
 Answer the given question.
 You must conduct reasoning inside <think> and </think> first every time you get new information.
 After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.
 You can search as many times as your want.
 If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question"""


python3 -m verl.trainer.main \
    config=examples/config_test.yaml \
    data.train_files=/data/tzhang/project/Infoseek_multi_hop/infoseek_bridge_refine/infoseek_bridge_train_all_right \
    data.val_files=/data/tzhang/project/Infoseek_multi_hop/infoseek_bridge_parquet/infoseek_bridge_with_entity/infoseek_bridge_validation \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.max_start_length=2048 \
    data.max_obs_length=512 \
    data.max_end_length=2048 \
    data.rollout_batch_size=128 \
    data.val_batch_size=256 \
    worker.actor.global_batch_size=64 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.n=8 \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=4 \
    trainer.save_checkpoint_path=${save_path} \
    trainer.load_checkpoint_path=${load_checkpoint_path} \
    trainer.val_freq=20 \
    trainer.val_before_train=false \
    trainer.val_generations_to_log=20 \
    trainer.save_freq=20 \
    retriever.url=http://127.0.0.1:8000/retrieve \
    retriever.topk=2 
    

input_file=/data/tzhang/project/Infoseek_multi_hop/search_engine/raw_data/infoseek_bridge_wiki_with_image.jsonl
save_dir=/data/tzhang/project/Infoseek_multi_hop/search_engine/image

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 index_builder_image.py \
    --model_name CLIP \
    --model_path /data/tzhang/model/clip-vit-large-patch14-336 \
    --input_file $input_file \
    --save_dir $save_dir \
    --image_weight 0.5 --title_weight 0.5 \
    --faiss_type Flat \
    --feature_dim 768
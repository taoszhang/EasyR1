
corpus_file=/data/tzhang/project/Infoseek_multi_hop/search_engine/raw_data_chunk_2000/infoseek_bridge_wiki.jsonl # jsonl
save_dir=/data/tzhang/project/Infoseek_multi_hop/search_engine/raw_data_chunk_2000
retriever_name=NV-Embed-v2 # this is for indexing naming
retriever_model=/data/tzhang/model/NV-Embed-v2

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0 python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding

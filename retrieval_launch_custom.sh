file_path=/data/tzhang/project/Infoseek_multi_hop/search_engine/raw_data
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/infoseek_bridge_wiki_contents.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
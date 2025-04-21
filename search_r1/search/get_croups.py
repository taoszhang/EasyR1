import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# # 加载数据
# data = []
# with open("/group/40034/taoszhang/project/KG-RAG/Multi-hop/infoseek_compare/raw_data/infoseek_all_entity.jsonl", "r") as f:
#     for line in f:
#         data.append(json.loads(line))

merged_infoseek_bridge = []
with open('/group/40034/taoszhang/project/KG-RAG/Multi-hop/infoseek_bridge/processed_data/merged_bridge_question_first_from_answer.jsonl', 'r', encoding='utf-8') as f:
    merged_infoseek_bridge = [json.loads(line) for line in f]


# Step 1: Load the JSONL file and create a mapping from ID to content
wiki_data_file = '/group/40034/taoszhang/datasets/Wikipedia/Wikipedia_Text_Information/Wiki6M_ver_1_0.jsonl'
title_to_content = {}

with open(wiki_data_file, 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line)
        entity_title = entry['wikipedia_title']
        title_to_content[entity_title] = entry

# for item in merged_infoseek_bridge:
merged_infoseek_bridge_with_kb = []

infoseek_bridge_all = []
for item in tqdm(merged_infoseek_bridge, total=len(merged_infoseek_bridge)):
    try:
        infoseek_bridge_all.append(title_to_content[item['entity_text']])
        infoseek_bridge_all.append(title_to_content[item['bridge_entity']])
    except:
        print(f"Error: {item['entity_text']} or {item['bridge_entity']} not found in Wikipedia data.")
        continue

print(len(infoseek_bridge_all))
# 去重
infoseek_bridge_all = set([json.dumps(item, ensure_ascii=False) for item in infoseek_bridge_all])
infoseek_bridge_all = [json.loads(item) for item in infoseek_bridge_all]
print(len(infoseek_bridge_all))


docs = [Document(page_content=item["wikipedia_title"] + '##' + item["wikipedia_content"]) for item in infoseek_bridge_all]
print(f"load {len(docs)} wikipeidia articlas.")

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
print(f"Split wikipeidia articlas into {len(all_splits)} sub-documents.")

# 加载嵌入模型到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/group/40034/taoszhang/model/NV-Embed-v2"
embedding_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(device)
embedding_model = torch.nn.DataParallel(embedding_model)  # 多 GPU 并行
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def embed_documents(model, tokenizer, documents, batch_size=64):
    embeddings = []
    texts = []  # 用于存储原始文本
    with torch.no_grad():  # 禁用梯度计算
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
            batch = documents[i:i + batch_size]
            batch_content = [doc.page_content for doc in batch]
            texts.extend(batch_content)  # 记录文本顺序
            
            # 对文本进行 Tokenizer 编码
            inputs = tokenizer(batch_content, padding=True, truncation=False, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 模型前向传播获取嵌入
            outputs = model(**inputs)
            # import pdb; pdb.set_trace()
            batch_embeddings = outputs['sentence_embeddings'].mean(dim=1).cpu().numpy()  # 平均池化
            embeddings.extend(batch_embeddings)
    return np.array(embeddings), texts

# 批量嵌入文档
batch_size = 256  # 根据显存调整批量大小
all_embeddings, all_texts = embed_documents(embedding_model, tokenizer, all_splits, batch_size=batch_size)

# 保存嵌入和原始文本
embedding_file = "/group/40034/taoszhang/project/KG-RAG/Multi-hop/evaluation/models/self_ask/infoseek_bridge_embeddings.npy"
texts_file = "/group/40034/taoszhang/project/KG-RAG/Multi-hop/evaluation/models/self_ask/infoseek_bridge_texts.jsonl"

np.save(embedding_file, all_embeddings)
print(f"Saved embeddings to {embedding_file}.")

with open(texts_file, "w") as f:
    for text in all_texts:
        f.write(json.dumps({"content": text}) + "\n")
print(f"Saved original texts to {texts_file}.")

import pdb; pdb.set_trace()
print(f"Generated embeddings for {len(all_embeddings)} sub-documents.")

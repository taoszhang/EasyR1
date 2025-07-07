import os
import argparse
import torch
import faiss
import clip
from PIL import Image
from typing import List, Union
from tqdm import tqdm
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from gme_inference import GmeQwen2VL
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRetriever:
    def __init__(self, config):
        self.config = config

        # Load FAISS index
        assert os.path.exists(config.index_path), f"Index file not found: {config.index_path}"
        index = faiss.read_index(config.index_path)
        if torch.cuda.device_count() > 1:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co)
        self.index = index

        # Load titles
        assert os.path.exists(config.title_path), f"Title file not found: {config.title_path}"
        with open(config.title_path, 'r') as f:
            self.titles = [line.strip() for line in f]

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = device
        # self.gme_model = GmeQwen2VL(config.model_path, device=device)
        self.batch_size = config.batch_size

        self.device_ids = list(range(torch.cuda.device_count()))
        assert len(self.device_ids) > 0, "No CUDA devices available."

        # 多卡加载多个模型副本（独立模型）
        self.models = {
            device_id: GmeQwen2VL(model_path=config.model_path, device=f'cuda:{device_id}')
            for device_id in self.device_ids
        }

    # @torch.no_grad()
    # def encode_image(self, image_path: str):
    #     image_features = self.gme_model.get_image_embeddings([image_path])
    #     return image_features.cpu().numpy()

    # @torch.no_grad()
    # def encode_image_batch(self, images: List[str]):
    #     image_features = []
    #     for i in range(0, len(images), self.batch_size):
    #         embeds = self.gme_model.get_image_embeddings(images[i:i + self.batch_size])
    #         image_features.append(embeds.cpu().numpy())
    #     return np.concatenate(image_features, axis=0)

    @torch.no_grad()
    def encode_image_batch(self, image_paths: List[str]):
        total = len(image_paths)
        results = [None] * total
        
        def encode_on_device(images, texts, idx_range, device_id):
            model = self.models[device_id]
            # embs = model.get_image_embeddings(batch).cpu()
            embs = model.get_fused_embeddings(texts=texts, images=images).cpu()
            for i, idx in enumerate(idx_range):
                results[idx] = embs[i]

        with ThreadPoolExecutor(max_workers=len(self.device_ids)) as executor:
            futures = []
            for i in range(0, total, self.batch_size):
                images = image_paths[i:i + self.batch_size]
                texts = ["Find the entity in the image."] * len(images)  # Dummy texts, not used in image retrieval
                idx_range = list(range(i, i + len(images)))
                device_id = self.device_ids[(i // self.batch_size) % len(self.device_ids)]
                futures.append(executor.submit(encode_on_device, images, texts, idx_range, device_id))

            for future in futures:
                future.result()
        return torch.stack(results, dim=0).numpy()

    # def search(self, image, topk: int = 10):
    #     image_features = self.encode_image(image)
    #     scores, indices = self.index.search(image_features, topk)
    #     # return [(self.titles[i], scores[0][rank]) for rank, i in enumerate(indices[0])]
    #     return [(self.titles[i], float(scores[0][rank])) for rank, i in enumerate(indices[0])]


    def batch_search(self, image_paths: List[str], topk: int = 10):
        image_features = self.encode_image_batch(image_paths)
        scores, indices = self.index.search(image_features, topk)
        batch_results = []
        for i in range(len(image_paths)):
            # result = [(self.titles[idx], scores[i][rank]) for rank, idx in enumerate(indices[i])]
            result = [(self.titles[idx], float(scores[i][rank])) for rank, idx in enumerate(indices[i])]
            batch_results.append(result)
        return batch_results

class Config:
    def __init__(self, model_path, index_path, title_path, batch_size):
        self.model_path = model_path
        self.index_path = index_path
        self.title_path = title_path
        self.batch_size = batch_size

class RetrievalRequest(BaseModel):
    image_paths: Union[str, List[str]]
    topk: int = 10

@app.on_event("startup")
def load_model():
    global retriever
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--title_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args, _ = parser.parse_known_args()
    config = Config(args.model_path, args.index_path, args.title_path, args.batch_size)
    retriever = ImageRetriever(config)

@app.post("/image_search")
def retrieve(request: RetrievalRequest):
    if isinstance(request.image_paths, str):
        result = retriever.search(request.image_paths, topk=request.topk)
        return {"mode": "single", "results": result}
    elif isinstance(request.image_paths, list):
        result = retriever.batch_search(request.image_paths, topk=request.topk)
        return {"mode": "batch", "results": result}
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid input type. Must be a string or list of strings."})

if __name__ == '__main__':
    import sys
    sys.argv.append("--model_path")
    sys.argv.append("/data/tzhang/model/gme-Qwen2-VL-7B-Instruct")
    sys.argv.append("--index_path")
    sys.argv.append("/data/tzhang/project/Infoseek_multi_hop/search_engine/GME_7B_text_new/GME_Flat.index")
    sys.argv.append("--title_path")
    sys.argv.append("/data/tzhang/project/Infoseek_multi_hop/search_engine/GME_7B_text_new/GME_Flat_titles.txt")
    sys.argv.append("--batch_size")
    sys.argv.append("256")
    uvicorn.run("image_retrieval_server:app", host="0.0.0.0", port=8888, reload=False)

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

        # Load CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(config.model_path, device=device)
        model.eval()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.batch_size = config.batch_size

    @torch.no_grad()
    def encode_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError("Input must be a file path or PIL Image.")
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.module.encode_image(image) if isinstance(self.model, torch.nn.DataParallel) else self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    @torch.no_grad()
    def encode_image_batch(self, images):
        if isinstance(images[0], str):
            images = [Image.open(p).convert("RGB") for p in images]
        elif isinstance(images[0], Image.Image):
            images = [img.convert("RGB") for img in images]
        else:
            raise ValueError("Input must be a list of file paths or PIL Images.")
        image_features = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i+self.batch_size]
            batch = [self.preprocess(img) for img in batch]
            image_tensor = torch.stack(batch).to(self.device)
            encode_fn = self.model.module.encode_image if isinstance(self.model, torch.nn.DataParallel) else self.model.encode_image
            feats = encode_fn(image_tensor)
            feats /= feats.norm(dim=-1, keepdim=True)
            image_features.append(feats.cpu().numpy())
        return np.concatenate(image_features, axis=0)

    def search(self, image, topk: int = 10):
        image_features = self.encode_image(image)
        scores, indices = self.index.search(image_features, topk)
        # return [(self.titles[i], scores[0][rank]) for rank, i in enumerate(indices[0])]
        return [(self.titles[i], float(scores[0][rank])) for rank, i in enumerate(indices[0])]


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
    sys.argv.append("ViT-L/14@336px")
    sys.argv.append("--index_path")
    sys.argv.append("/data/tzhang/project/Infoseek_multi_hop/search_engine/image/CLIP_Flat.index")
    sys.argv.append("--title_path")
    sys.argv.append("/data/tzhang/project/Infoseek_multi_hop/search_engine/image/CLIP_Flat_titles.txt")
    sys.argv.append("--batch_size")
    sys.argv.append("64")
    uvicorn.run("image_retrieval_server:app", host="0.0.0.0", port=8888, reload=False)

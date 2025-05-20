import os
import argparse
import torch
import faiss
import clip
from PIL import Image
from typing import List
from tqdm import tqdm

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
        all_features = []
        batch_size = self.config.batch_size

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            if isinstance(batch[0], str):
                batch = [Image.open(p).convert("RGB") for p in batch]
            elif isinstance(batch[0], Image.Image):
                batch = [img.convert("RGB") for img in batch]
            else:
                raise ValueError("Input must be a list of file paths or PIL Images.")

            batch = [self.preprocess(img) for img in batch]
            image_tensor = torch.stack(batch).to(self.device)
            encode_fn = self.model.module.encode_image if isinstance(self.model, torch.nn.DataParallel) else self.model.encode_image
            image_features = encode_fn(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_features.append(image_features.cpu())

        return torch.cat(all_features, dim=0).numpy()
    
    def search(self, image, topk: int = 10):
        image_features = self.encode_image(image)
        scores, indices = self.index.search(image_features, topk)
        return [(self.titles[i], scores[0][rank]) for rank, i in enumerate(indices[0])]

    def batch_search(self, image_paths: List[str], topk: int = 10):
        image_features = self.encode_image_batch(image_paths)
        scores, indices = self.index.search(image_features, topk)
        batch_results = []
        for i in range(len(image_paths)):
            result = [(self.titles[idx], scores[i][rank]) for rank, idx in enumerate(indices[i])]
            batch_results.append(result)
        return batch_results

def main():
    parser = argparse.ArgumentParser(description="Image-based retrieval using FAISS and CLIP")
    parser.add_argument('--model_path', type=str, required=True, help='Path to CLIP model')
    parser.add_argument('--index_path', type=str, required=True, help='Path to FAISS index')
    parser.add_argument('--title_path', type=str, required=True, help='Path to title list file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to query image or a text file of paths')
    parser.add_argument('--topk', type=int, default=10, help='Number of results to return')
    parser.add_argument('--batch', action='store_true', help='Enable batch mode if input is a text file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for batch search')
    
    args = parser.parse_args()
    retriever = ImageRetriever(args)

    if args.batch:
        assert os.path.isfile(args.image_path), "For batch mode, provide a file with image paths."
        with open(args.image_path, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        batch_results = retriever.batch_search(image_paths, args.topk)
        for i, result in enumerate(batch_results):
            print(f"\nQuery Image: {image_paths[i]}")
            for rank, (title, score) in enumerate(result, 1):
                print(f"{rank}. {title} (score: {score:.4f})")
    else:
        results = retriever.search(args.image_path, args.topk)
        print("Top-k similar images:")
        for rank, (title, score) in enumerate(results, 1):
            print(f"{rank}. {title} (score: {score:.4f})")

if __name__ == '__main__':
    main()

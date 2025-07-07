import os
import json
import torch
import faiss
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import clip
import argparse
import torch.distributed as dist
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, file_path, preprocess=None):
        self.image_data = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.image_data.append(data)
        self.image_data = [data for data in self.image_data if data['image_path'] is not None]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image_path = self.image_data[idx]['image_path']
        title = self.image_data[idx]['title']
        image = Image.open(image_path).convert('RGB')
        if self.preprocess:
            image = self.preprocess(image)
        return title, image

class FeatureExtractor:
    def __init__(self, args):
        # Initialize the distributed environment
        # dist.init_process_group(backend="nccl", init_method='env://')
        # self.rank = dist.get_rank()
        # self.world_size = dist.get_world_size()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load the model and dataset
        self.model, self.preprocess = clip.load(args.model_path, self.device)
        self.model.to(self.device)
        self.model.eval()
    
        self.dataset = ImageDataset(args.input_file, self.preprocess)
        self.data_loader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
        
        # # Dataset and DataLoader setup
        # self.dataset = ImageDataset(args.input_file, self.preprocess)
        # sampler = DistributedSampler(self.dataset, num_replicas=args.world_size, rank=args.rank)
        # self.data_loader = DataLoader(self.dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)

        # Parameters
        self.args = args
        self.batch_size = args.batch_size
        self.features_dir = args.save_dir
        self.faiss_type = args.faiss_type
        self.faiss_gpu = args.faiss_gpu
        self.all_features = []
        self.all_titles = []  # Store titles
        self.index_save_path = os.path.join(self.features_dir, f"{args.model_name}_{self.faiss_type}.index")

    def encode_image_and_title(self, image, title):
        # Perform encoding for both image and title
        image = image.to(self.device)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize

        # Tokenize title and encode
        title = clip.tokenize(title).to(self.device)
        title_features = self.model.encode_text(title)
        title_features /= title_features.norm(dim=-1, keepdim=True)  # Normalize
        
        return image_features, title_features

    def extract_features(self):
        all_features = []
        all_titles = []  # Collect titles
        
        with torch.no_grad():
            for titles, images in tqdm(self.data_loader, desc="Extracting Features", leave=True):
                images = images.to(self.device)
                # import pdb; pdb.set_trace()
                # images_tensor = torch.stack(images).to(self.device)
                image_features, title_features = self.encode_image_and_title(images, titles)

                # Combine image and title features with weight
                weight_image = self.args.image_weight
                weight_title = self.args.title_weight
                combined_features = weight_image * image_features + weight_title * title_features

                all_features.append(combined_features.cpu())
                all_titles.extend(titles)  # Collect titles corresponding to the images
                    
        all_features = torch.cat(all_features, dim=0).numpy()

        # Save features as memmap
        # if self.rank == 0:
        self._save_features(all_features, all_titles)


    def _save_features(self, features, titles):
        if not os.path.exists(self.features_dir):
            os.makedirs(self.features_dir)

        # Save features as memmap file
        feature_file_path = os.path.join(self.features_dir, f"{self.args.model_name}_{self.faiss_type}_features.memmap")
        memmap = np.memmap(feature_file_path, 
                           dtype=np.float32, 
                           mode='w+', 
                           shape=features.shape)
        length = features.shape[0]
        
        # 批量写入特征数据
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i: j] = features[i: j]
        else:
            memmap[:] = features

        # 保存标题
        titles_file_path = os.path.join(self.features_dir, f"{self.args.model_name}_{self.faiss_type}_titles.txt")
        with open(titles_file_path, 'w') as f:
            f.write('\n'.join(titles))
        print("Feature extraction completed and saved.")

    def create_faiss_index(self, features):
        # # Create FAISS index based on features
        # dim = features.shape[-1]
        # index = faiss.IndexFlatL2(dim)  # Using L2 distance for similarity search
        # index.add(features)  # Add features to the index

        # # Save index
        # faiss.write_index(index, self.index_save_path)
        # print("FAISS index created and saved.")
        dim = features.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)
        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(features)
            faiss_index.add(features)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(features)
            faiss_index.add(features)
        faiss.write_index(faiss_index, self.index_save_path)
        print("Finish!")

    @torch.no_grad()
    def build_index(self):
        features_file_path = os.path.join(self.features_dir, f"{self.args.model_name}_{self.args.faiss_type}_features.memmap")
        features = np.memmap(features_file_path, dtype=np.float32, mode='r', shape=(len(self.dataset), self.args.feature_dim))

        self.create_faiss_index(features)

class Retriever:
    def __init__(self, args):
        self.args = args
        self.features_dir = args.save_dir
        self.index_file_path = os.path.join(self.features_dir, f"{args.model_name}_{self.faiss_type}.index")
        self.index = faiss.read_index(self.index_file_path)  # Load the FAISS index

    def search(self, query_image, query_title, model, device, k=10):
        query_image_features, query_title_features = model.encode_image(query_image), model.encode_text(clip.tokenize([query_title]).to(device))
        
        # Combine query features based on weights
        query_combined = self.args.image_weight * query_image_features + self.args.title_weight * query_title_features

        # Perform search using FAISS index
        D, I = self.index.search(query_combined.cpu().numpy(), k)

        return D, I

def main():
    parser = argparse.ArgumentParser(description='CLIP Feature Extraction and Indexing')
    parser.add_argument('--model_name', type=str, default='CLIP', help='Name of the CLIP model used')
    parser.add_argument('--model_path', type=str, default='ViT-L/14@336px', help='Path to the CLIP model')
    parser.add_argument('--input_file', type=str, default='/data/tzhang/project/Infoseek_multi_hop/search_engine/raw_data/infoseek_bridge_wiki_with_image.jsonl', help='Path to the image paths text file')
    parser.add_argument('--save_dir', type=str, default='/data/tzhang/project/Infoseek_multi_hop/search_engine/image', help='Root directory to save features')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes for data loading')
    parser.add_argument('--image_weight', type=float, default=0.5, help='Weight for image features')
    parser.add_argument('--title_weight', type=float, default=0.5, help='Weight for title features')
    parser.add_argument('--feature_dim', type=int, default=768, help='Feature dimension size (e.g., 768 for ViT-L/14)')
    parser.add_argument('--faiss_type', type=str, default='Flat', help='FAISS index type (e.g., "Flat", "IVF", etc.)')
    parser.add_argument('--faiss_gpu', action='store_true', default=False, help='Use FAISS GPU indexing')
    # parser.add_argument('--world_size', type=int, default=1, help='Total number of processes (GPUs)')
    # parser.add_argument('--rank', type=int, default=0, help='Rank of the current process')

    args = parser.parse_args()

    # Feature extraction and indexing
    feature_extractor = FeatureExtractor(args)
    feature_extractor.extract_features()
    feature_extractor.build_index()

if __name__ == '__main__':
    main()

from PIL import Image
import io
import json
import pandas as pd
import os
from typing import Dict
import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_jsonl(file_path: str) -> list:
    """Load a .jsonl file line by line."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data: list, file_path: str):
    """Save a list of dictionaries to a .jsonl file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_infoseek_image_bytes(image_id: str):
    """Try to load an image from multiple folders and return bytes + path."""
    image_folders = [
        '/data/tzhang/dataset/Infoseek/infoseek_images/infoseek_human_images',
        '/data/tzhang/dataset/Infoseek/infoseek_images/infoseek_train_images',
        '/data/tzhang/dataset/Infoseek/infoseek_images/infoseek_val_images',
        '/data/tzhang/dataset/Infoseek/infoseek_images/infoseek_test_images'
    ]
    suffixes = ['.jpg', '.JPEG', '.jpeg', '.JPG']

    for folder in image_folders:
        for suffix in suffixes:
            image_path = os.path.join(folder, image_id + suffix)
            if os.path.exists(image_path):
                try:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    return image_bytes, image_path
                except Exception as e:
                    print(f"Error reading image as bytes from {image_path}: {e}")
    return None, None

def make_prefix(question, template_type='base'):
    if template_type == 'base':
        return (
            f"<image>\nAnswer the given question. "
            f"You must conduct reasoning inside <think> and </think> first every time you get new information. "
            f"After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
            f"and it will return the top searched results between <information> and </information>. "
            f"You can search as many times as your want. "
            f"If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
            f"without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
            f"Answer the question with a single word or a phrase. "
        )
    else:
        raise NotImplementedError

def process_item(i: int, item: dict):
    try:
        image_bytes, image_path = load_infoseek_image_bytes(random.choice(item['image_ids']))
        if image_bytes is None:
            print(f"Image not found for ID: {item['image_ids']}")
            return None

        entity_name = item['entity_text']
        question = f"This is {entity_name}.\n{item['two_hop_question']}"

        if 'answer_2' not in item or (not isinstance(item['answer_2'], str) and not isinstance(item['answer_2'], list)):
            print(f"Skipping item {i} due to missing or invalid 'answer_2'.")
            return None
        answer = item['answer_2'] if isinstance(item['answer_2'], str) else item['answer_2'][0]

        ground_truth = (
            [str(float(x)) for x in item['answer_eval_2']['range']]
            if isinstance(item['answer_eval_2'], dict)
            else item['answer_eval_2']
        )

        return {
            'id': int(str(i).zfill(5)),
            'images': np.array([{'bytes': image_bytes, 'path': image_path}], dtype=object),
            'problem': question,
            'answer': answer,
            'bridge_entity': item['bridge_entity_text'],
            'ground_truth': np.array(ground_truth, dtype=object),
            'problem_type': item['question_type']
        }
    except Exception as e:
        print(f"Error processing item {i}: {e}")
        # import pdb; pdb.set_trace()
        return None
    
def is_valid_sample(result):
    """Check if the result contains valid data for all required fields."""
    if result is None:
        return False

    # Check if all necessary fields are present and not empty or invalid
    required_fields = ['images', 'problem', 'answer', 'ground_truth', 'problem_type']
    
    for field in required_fields:
        
        if field == 'images':
            if not isinstance(result[field], np.ndarray) or len(result[field]) == 0:
                print(f"Skipping sample due to invalid 'images' field.")
                return False
            for image in result[field]:
                if not isinstance(image, dict) or 'bytes' not in image or 'path' not in image or not image['bytes'] or not image['path']:
                    print(f"Skipping sample due to invalid image format.")
                    return False
                
        if field == 'ground_truth':
            if not isinstance(result[field], np.ndarray) or len(result[field]) == 0:
                print(f"Skipping sample due to invalid 'ground_truth' field.")
                return False
            for gt in result[field]:
                if not isinstance(gt, str):
                    print(f"Skipping sample due to invalid ground truth format.")
                    return False
                
        if field == 'problem_type':
            if not isinstance(result[field], str) or not result[field]:
                print(f"Skipping sample due to invalid 'problem_type' field.")
                return False
            
        if field == 'answer':
            if not isinstance(result[field], str) or not result[field]:
                print(f"Skipping sample due to invalid 'answer' field.")
                return False
        
        if field == 'problem':
            if not isinstance(result[field], str) or not result[field]:
                print(f"Skipping sample due to invalid 'problem' field.")
                return False

    return True

if __name__ == "__main__":
    data_path = '/data/tzhang/project/KG-RAG/Multi-hop/infoseek_bridge/bridge_results/three_stages_results/3_merge_question/a_v2_merge/infoseek_bridge_question_entity.jsonl'
    all_data = load_jsonl(data_path)

    processed_data = []
    random.seed(0)

    print("Processing with multithreading...")

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_item, i, item) for i, item in enumerate(all_data)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is not None:
                processed_data.append(result)

    processed_data = [result for result in processed_data if is_valid_sample(result)]
    # 切片十份保存
    for i in range(10):
        start = i * len(processed_data) // 10
        end = (i + 1) * len(processed_data) // 10
        df = pd.DataFrame(processed_data[start:end])
        output_dir = '/data/tzhang/dataset/infoseek_bridge_with_entity'
        os.makedirs(output_dir, exist_ok=True)
        if i < 7:
            save_path = os.path.join(output_dir, f'infoseek_bridge_train', f'train-{i:05d}-of-00010.parquet')
        elif i == 7:
            save_path = os.path.join(output_dir, f'infoseek_bridge_validation', f'validation-{i:05d}-of-00010.parquet')
        else:
            save_path = os.path.join(output_dir, f'infoseek_bridge_test', f'test-{i:05d}-of-00010.parquet')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path, index=True)
        print(f"Saved part {i} to {save_path}")




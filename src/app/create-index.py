import pickle
import argparse
import os

import faiss
import numpy as np
from PIL import Image

import torch
from torch import nn

from transformers import ViTImageProcessor, ViTModel

from ai_hack.models import Embedder

from config import *



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dimension = 64


    image_processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
    trunk = ViTModel.from_pretrained('facebook/dino-vits16')
    trunk_output_size = trunk.config.hidden_size  
    trunk = nn.DataParallel(trunk).to(device)
    embedder = nn.DataParallel(Embedder(input_dim=trunk_output_size, 
                                        embedding_dim=dimension)).to(device)
    models = {'trunk': trunk, 'embedder': embedder}
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
    models['trunk'].load_state_dict(checkpoint['trunk_state_dict'])
    models['embedder'].load_state_dict(checkpoint['embedder_state_dict'])


    def get_embedding(image_path):
        image = Image.open(image_path).convert('RGB')
        with torch.no_grad():
            image = image_processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
            image_vit = trunk(image.unsqueeze(0))
            embedding = embedder(image_vit).cpu().numpy()
        return embedding


    index = faiss.index_factory(dimension, "Flat", faiss.METRIC_INNER_PRODUCT)
    image_paths = {}
    
    idx = 0
    for root, dirs, files in os.walk(DATASET_PATH):
        print(dirs)
        print(files)
        for filename in files:
            if filename.endswith('.jpg'):
                image_path = os.path.join(root, filename)
                embedding = get_embedding(image_path)
                faiss.normalize_L2(embedding)
                index.add(embedding)

                image_paths[idx] = image_path
                idx += 1


    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(IMAGE_IDX_MAP_PATH, 'wb') as f:
        pickle.dump(image_paths, f)

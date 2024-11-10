import os
import random
from enum import Enum
from tqdm import tqdm
from PIL import Image
from transformers import ViTModel, ViTImageProcessor

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
from torch.optim import lr_scheduler
import json

from torchvision import transforms
from transformers import AutoModel
import numpy as np
from torch.utils.data import random_split
import faiss
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.samplers import MPerClassSampler
import logging
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.trainers import TrainWithClassifier


def create_image_label_txt(image_dir, output_file):
    with open(output_file, 'w') as file:
        for label, folder in enumerate(os.listdir(image_dir)):
            folder_path = os.path.join(image_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for image_name in os.listdir(folder_path):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    file.write(f"{os.path.join(folder, image_name)};{label}\n")


LOGGER_NAME = "PML"
LOGGER = logging.getLogger(LOGGER_NAME)


def try_gpu(index, query, reference, k, is_cuda, gpus):
    # https://github.com/facebookresearch/faiss/blob/master/faiss/gpu/utils/DeviceDefs.cuh
    gpu_index = None
    gpus_are_available = faiss.get_num_gpus() > 0
    gpu_condition = (is_cuda or (gpus is not None)) and gpus_are_available
    if gpu_condition:
        max_k_for_gpu = 1024 if float(torch.version.cuda) < 9.5 else 2048
        if k <= max_k_for_gpu:
            gpu_index = convert_to_gpu_index(index, gpus)
    try:
        return add_to_index_and_search(gpu_index, query, reference, k)
    except (AttributeError, RuntimeError):
        if gpu_condition:
            c_f.LOGGER.warning(
                f"Using CPU for k-nn search because k = {k} > {max_k_for_gpu}, which is the maximum allowable on GPU."
            )
        cpu_index = convert_to_cpu_index(index)
        return add_to_index_and_search(cpu_index, query, reference, k)


def convert_to_gpu_index(index, gpus):
    if "Gpu" in str(type(index)):
        return index
    if gpus is None:
        return faiss.index_cpu_to_all_gpus(index)
    return faiss.index_cpu_to_gpus_list(index, gpus=gpus)


def convert_to_cpu_index(index):
    if "Gpu" not in str(type(index)):
        return index
    return faiss.index_gpu_to_cpu(index)


def add_to_index_and_search(index, query, reference, k):
    if reference is not None:
        index.add(reference.float().cpu())
    return index.search(query.float().cpu(), k)


class FaissKNN:
    def __init__(self, reset_before=True, reset_after=True, index_init_fn=None, gpus=None):
        self.reset()
        self.reset_before = reset_before
        self.reset_after = reset_after
        self.index_init_fn = faiss.IndexFlatL2 if index_init_fn is None else index_init_fn
        if gpus is not None:
            if not isinstance(gpus, (list, tuple)):
                raise TypeError("gpus must be a list")
            if len(gpus) < 1:
                raise ValueError("gpus must have length greater than 0")
        self.gpus = gpus

    def __call__(self, query, k, reference=None, ref_includes_query=False):
        if ref_includes_query:
            k = k + 1
        device = query.device
        is_cuda = query.is_cuda
        d = query.shape[1]
        if self.reset_before:
            self.index = self.index_init_fn(d)
        if self.index is None:
            raise ValueError("self.index is None. It needs to be initialized before being used.")
        distances, indices = try_gpu(self.index, query, reference, k, is_cuda, self.gpus)
        distances = to_device(distances, device=device)
        indices = to_device(indices, device=device)
        if self.reset_after:
            self.reset()
        return return_results(distances, indices, ref_includes_query)

    def train(self, embeddings):
        self.index = self.index_init_fn(embeddings.shape[1])
        self.add(numpy_to_torch(embeddings).cpu())

    def add(self, embeddings):
        self.index.add(numpy_to_torch(embeddings).cpu())

    def save(self, filename):
        faiss.write_index(self.index, filename)

    def load(self, filename):
        self.index = faiss.read_index(filename)

    def reset(self):
        self.index = None


def numpy_to_torch(v):
    try:
        return torch.from_numpy(v)
    except TypeError:
        return v


def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x


def to_device(x, tensor=None, device=None, dtype=None):
    dv = device if device is not None else tensor.device
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        x = x.to(dv)
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x


def return_results(D, I, ref_includes_query):
    if ref_includes_query:
        self_idx = torch.arange(len(I), device=I.device)
        matches_self_idx = I == self_idx.unsqueeze(1)
        row_has_match = torch.any(matches_self_idx, dim=1)
        # If every row has a match, then masking will work
        if not torch.all(row_has_match):
            # For rows that don't contain the self index
            # Remove the Nth value by setting matches_self_idx[N] to True
            matches_self_idx[~row_has_match, -1] = True
        I = mask_reshape_knn_idx(I, matches_self_idx)
        D = mask_reshape_knn_idx(D, matches_self_idx)
    return D, I


def mask_reshape_knn_idx(x, matches_self_idx):
    return x[~matches_self_idx].view(x.shape[0], -1)


class Difficulty(Enum):
    Easy = 1        # A - P < A - N
    SemiHard = 2    # min(A - N)
    Hard = 3        # max(A - P), min(A - N)


def _get_anchor_positive_triplet_mask(labels):
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):
    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


def _pairwise_distances(embeddings, squared=False, cosine=False):
    dot_product = torch.matmul(embeddings, embeddings.t())
    if cosine:  # Cosine range is -1 to 1. 1 - similarity makes 0 be closest, 2 = furthest
        norm = torch.norm(embeddings, dim=1, keepdim=True)
        similarity = dot_product / torch.matmul(norm, norm.t())
        return 1 - similarity

    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances[distances < 0] = 0
    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16
        distances = (1.0 - mask) * torch.sqrt(distances)
    return distances


def _masked_minimum(data, mask, dim=1):
    axis_maximums = data.max(dim, keepdim=True).values
    masked_minimums = ((data - axis_maximums) * mask.float()).min(dim, keepdim=True).values + axis_maximums
    return masked_minimums


def _masked_maximum(data, mask, dim=1):
    axis_minimums = data.min(dim, keepdim=True).values
    masked_maximums = ((data - axis_minimums) * mask.float()).max(dim, keepdim=True).values + axis_minimums
    return masked_maximums


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False, cosine=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared, cosine=cosine)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()
    return triplet_loss


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2, cosine=False, difficulty=Difficulty.Easy):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cosine = cosine
        self.difficulty = difficulty

    # e.g. loss.change_parameter(difficulty=Difficulty.Hard)
    def change_parameter(self, margin=None, cosine=None, difficulty=None):
        self.margin = self.margin if margin is None else margin
        self.cosine = self.cosine if cosine is None else cosine
        self.difficulty = self.difficulty if difficulty is None else difficulty

    def forward(self, labels, embeddings):
        if self.difficulty == Difficulty.Hard:
            return batch_hard_triplet_loss(labels, embeddings, self.margin, cosine=self.cosine)

        adjacency_not = _get_anchor_negative_triplet_mask(labels)
        batch_size = labels.size(0)

        pdist_matrix = _pairwise_distances(embeddings, cosine=self.cosine)
        pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
        mask = adjacency_not.repeat(batch_size, 1)

        if self.difficulty == Difficulty.Easy:
            mask = mask & torch.gt(pdist_matrix_tile, pdist_matrix.t().reshape(-1, 1))

        mask_final = torch.gt(mask.float().sum(dim=1, keepdim=True), 0.0).reshape(batch_size, batch_size)
        mask_final = mask_final.t()

        adjacency_not = adjacency_not.float()
        mask = mask.float()

        negatives_outside = (
            _masked_minimum(pdist_matrix_tile, mask)
            .reshape(batch_size, batch_size)
            .t()
        )

        negatives_inside = _masked_maximum(pdist_matrix, adjacency_not).repeat(1, batch_size)
        semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

        loss_mat = self.margin + pdist_matrix - semi_hard_negatives

        mask_positives = _get_anchor_positive_triplet_mask(labels)
        num_positives = torch.sum(mask_positives)
        triplet_loss = torch.sum(torch.clamp(loss_mat * mask_positives, min=0.0)) / (num_positives + 1e-8)
        return triplet_loss


class ImageLabelDataset(Dataset):
    def __init__(self, image_preprocessor, json_file, image_dir, transform=None):
        self.image_preprocessor = image_preprocessor
        self.image_dir = image_dir
        self.transform = transform

        with open(json_file, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        label = item['label']
        embedding = item['caption_embedding']

        image = Image.open(os.path.join(self.image_dir, image_path)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        image = self.image_preprocessor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)

        label = torch.tensor(label, dtype=torch.long)
        embedding = torch.tensor(embedding, dtype=torch.float32)

        return image, label, embedding


class Embedder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Embedder, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        return self.projection_head(x.last_hidden_state[:, 0, :])


class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class CosineEmbeddingLoss(nn.Module):
    def __init__(self):
        super(CosineEmbeddingLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, model_embeddings, target_embeddings):
        cosine_sim = self.cosine_similarity(model_embeddings, target_embeddings)
        loss = 1 - cosine_sim.mean()
        return loss


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def save_checkpoint(models, optimizers, epoch, epochs_dir):
    """Save model and optimizer states."""
    checkpoint = {
        'epoch': epoch,
        'trunk_state_dict': models['trunk'].state_dict(),
        'embedder_state_dict': models['embedder'].state_dict(),
        'classifier_state_dict': models['classifier'].state_dict(),
        'trunk_optimizer_state_dict': optimizers['trunk_optimizer'].state_dict(),
        'embedder_optimizer_state_dict': optimizers['embedder_optimizer'].state_dict(),
        'classifier_optimizer_state_dict': optimizers['classifier_optimizer'].state_dict(),
    }
    checkpoint_path = os.path.join(epochs_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_last_checkpoint(models, optimizers, epochs_dir):
    """Load the last checkpoint if available."""
    checkpoint_files = [f for f in os.listdir(epochs_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoint_files:
        print("No checkpoint found.")
        return 0

    last_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(epochs_dir, last_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    models['trunk'].load_state_dict(checkpoint['trunk_state_dict'])
    models['embedder'].load_state_dict(checkpoint['embedder_state_dict'])
    models['classifier'].load_state_dict(checkpoint['classifier_state_dict'])
    optimizers['trunk_optimizer'].load_state_dict(checkpoint['trunk_optimizer_state_dict'])
    optimizers['embedder_optimizer'].load_state_dict(checkpoint['embedder_optimizer_state_dict'])
    optimizers['classifier_optimizer'].load_state_dict(checkpoint['classifier_optimizer_state_dict'])
    return checkpoint['epoch'] + 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training script for a neural network")

    parser.add_argument("--image_dir", type=str, default="../sekrrno/dataset", help="image dir")
    parser.add_argument("--epochs_dir", type=str, default="./epochs_emb", help="epochs dir")
    parser.add_argument("--embedding_size", type=int, default=32, help="embedding size")
    parser.add_argument("--m_per_batch_size", type=int, default=4, help="m_per_batch_size")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--load_last", type=bool, default=False, help="load last")

    args = parser.parse_args()

    image_dir = args.image_dir
    epochs_dir = args.epochs_dir
    create_directory_if_not_exists(epochs_dir)
    batch_size = args.batch_size

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=10),
        # transforms.ToTensor()
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trunk = ViTModel.from_pretrained('facebook/dino-vits16')
    trunk_output_size = trunk.config.hidden_size

    trunk = nn.DataParallel(trunk).to(device)
    image_processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')

    dataset = ImageLabelDataset(image_preprocessor=image_processor, image_dir=image_dir,
                                json_file='captions_embed.json')  # , transform=transform)

    embedder = nn.DataParallel(Embedder(input_dim=trunk_output_size, embedding_dim=args.embedding_size)).to(device)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_labels = [dataset.data[idx]['label'] for idx in train_dataset.indices]
    train_labels = [dataset.data[idx]['label'] for idx in test_dataset.indices]

    metric_loss = losses.TripletMarginLoss(margin=0.5, distance=CosineSimilarity())

    miner = miners.TripletMarginMiner(margin=0.3, type_of_triplets='semihard')

    sampler = MPerClassSampler(train_labels, m=args.m_per_batch_size, length_before_new_iter=len(train_labels))

    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=5e-5, weight_decay=1e-5)
    embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=1e-4, weight_decay=1e-5)

    models = {'trunk': trunk, 'embedder': embedder}
    optimizers = {'trunk_optimizer': trunk_optimizer, 'embedder_optimizer': embedder_optimizer}
    loss_funcs = {
        'metric_loss': metric_loss
    }

    loss_weights = {
        'metric_loss': 1.0,
        'classifier_loss': 3.0,
        'contrastive_loss': 0.1
    }

    embedding_loss_func = CosineEmbeddingLoss()
    loss_funcs['embedding_loss'] = embedding_loss_func

    mining_funcs = {'tuple_miner': miner}

    num_classes = len(set(train_labels))

    classifier = nn.DataParallel(Classifier(embedding_dim=args.embedding_size, num_classes=num_classes)).to(device)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-5)
    trunk_scheduler = lr_scheduler.StepLR(trunk_optimizer, step_size=2, gamma=0.1)
    embedder_scheduler = lr_scheduler.StepLR(embedder_optimizer, step_size=2, gamma=0.1)
    classifier_scheduler = lr_scheduler.StepLR(classifier_optimizer, step_size=2, gamma=0.1)

    schedulers = {
        'trunk_scheduler_by_epoch': trunk_scheduler,
        'embedder_scheduler_by_epoch': embedder_scheduler,
        'classifier_scheduler_by_epoch': classifier_scheduler
    }

    optimizers['classifier_optimizer'] = classifier_optimizer

    classification_loss = nn.CrossEntropyLoss()
    loss_funcs['classifier_loss'] = classification_loss

    batch_size = args.batch_size

    models['classifier'] = classifier

    # trainer = TrainWithClassifier(
    #     models=models,
    #     optimizers=optimizers,
    #     batch_size=batch_size,
    #     loss_funcs=loss_funcs,
    #     mining_funcs=mining_funcs,
    #     lr_schedulers=schedulers,
    #     sampler=sampler,
    #     loss_weights=loss_weights,
    #     dataloader_num_workers=4,
    #     dataset=train_dataset,
    #     data_device=device,
    # )

    start_epoch = 0
    if args.load_last:
        start_epoch = load_last_checkpoint(models, optimizers, epochs_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    num_epochs = 10
    for epoch in range(start_epoch, num_epochs):
        print(f"Starting epoch {epoch}/{num_epochs}")

        models['trunk'].train()
        models['embedder'].train()
        models['classifier'].train()

        train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        train_embeddings = []
        train_labels = []

        for images, labels, caption_embeddings in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            caption_embeddings = caption_embeddings.to(device)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            trunk_output = models['trunk'](images)
            image_embeddings = models['embedder'](trunk_output)
            logits = models['classifier'](image_embeddings)

            metric_loss = loss_funcs['metric_loss'](image_embeddings, labels)
            classifier_loss = loss_funcs['classifier_loss'](logits, labels)

            # image_embeddings = F.normalize(image_embeddings, dim=1)
            # caption_embeddings = F.normalize(caption_embeddings, dim=1)

            # logits_per_image = image_embeddings @ caption_embeddings.T
            # logits_per_caption = logits_per_image.T

            # temperature = 0.07
            # logits_per_image /= temperature
            # logits_per_caption /= temperature

            # batch_size = images.size(0)
            # contrastive_labels = torch.arange(batch_size, dtype=torch.long).to(device)

            # loss_image = F.cross_entropy(logits_per_image, contrastive_labels)
            # loss_caption = F.cross_entropy(logits_per_caption, contrastive_labels)
            # contrastive_loss = (loss_image + loss_caption) / 2

            total_loss = (
                loss_weights['metric_loss'] * metric_loss +
                loss_weights['classifier_loss'] * classifier_loss
                # loss_weights['contrastive_loss'] * contrastive_loss
            )

            total_loss.backward()
            for optimizer in optimizers.values():
                optimizer.step()

            train_loss += total_loss.item()
            train_embeddings.append(image_embeddings.cpu())
            train_labels.append(labels.cpu())

            train_loader_tqdm.set_postfix(loss=total_loss.item())

        train_embeddings = torch.cat(train_embeddings)
        train_labels = torch.cat(train_labels)

        for scheduler in schedulers.values():
            scheduler.step()

        models['trunk'].eval()
        models['embedder'].eval()
        models['classifier'].eval()

        val_loss = 0
        val_embeddings = []
        val_labels = []
        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")

        with torch.no_grad():
            for images, labels, target_embeddings in test_loader:
                images = images.to(device)
                labels = labels.to(device, dtype=torch.long)
                target_embeddings = target_embeddings.to(device)

                trunk_output = models['trunk'](images)
                image_embeddings = models['embedder'](trunk_output)
                logits = models['classifier'](image_embeddings)

                metric_loss = loss_funcs['metric_loss'](image_embeddings, labels)
                classifier_loss = loss_funcs['classifier_loss'](logits, labels)

                # image_embeddings = F.normalize(image_embeddings, dim=1)
                # target_embeddings = F.normalize(target_embeddings, dim=1)

                # logits_per_image = image_embeddings @ target_embeddings.T
                # logits_per_caption = logits_per_image.T

                # temperature = 0.07
                # logits_per_image /= temperature
                # logits_per_caption /= temperature

                # batch_size = images.size(0)
                # contrastive_labels = torch.arange(batch_size, dtype=torch.long).to(device)
                # loss_image = F.cross_entropy(logits_per_image, contrastive_labels)
                # loss_caption = F.cross_entropy(logits_per_caption, contrastive_labels)
                # contrastive_loss = (loss_image + loss_caption) / 2

                total_loss = (
                    loss_weights['metric_loss'] * metric_loss +
                    loss_weights['classifier_loss'] * classifier_loss
                    # loss_weights['contrastive_loss'] * contrastive_loss
                )

                val_loss += total_loss.item()
                val_embeddings.append(image_embeddings.cpu())
                val_labels.append(labels.cpu())

                test_loader_tqdm.set_postfix(loss=total_loss.item())

        val_embeddings = torch.cat(val_embeddings)
        val_labels = torch.cat(val_labels)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(test_loader)}")
        save_checkpoint(models, optimizers, epoch, epochs_dir)

        matching_ratios = []
        average_precisions = []

        for i in range(len(val_embeddings)):
            val_emb = val_embeddings[i]
            val_label = val_labels[i]

            distances = torch.nn.functional.cosine_similarity(train_embeddings, val_emb.unsqueeze(0), dim=1)
            nearest_indices = torch.topk(distances, k=10, largest=False).indices
            nearest_labels = train_labels[nearest_indices]

            num_matching = (nearest_labels == val_label).sum().item()
            matching_ratio = num_matching / 10.0
            matching_ratios.append(matching_ratio)

            relevant_indices = (nearest_labels == val_label).nonzero(as_tuple=True)[0]
            precisions = [(rank + 1) / (index + 1) for rank, index in enumerate(relevant_indices.tolist())]
            average_precision = sum(precisions) / len(precisions) if precisions else 0
            average_precisions.append(average_precision)

        average_matching_ratio = sum(matching_ratios) / len(matching_ratios)
        map10 = sum(average_precisions) / len(average_precisions)
        print("\nValidation metric based on nearest neighbors:")
        print(f"Average matching ratio: {average_matching_ratio:.4f}")
        print(f"MAP@10: {map10:.4f}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import json
import socket
import time
import numpy as np
import argparse
import os
import sys
import psutil
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import gc
import h5py
from multiprocessing import shared_memory, Pool
from torch.utils.data import Dataset, DataLoader
import cuml
from cuml.manifold import UMAP as cumlUMAP
from cuml.manifold import TSNE
from cuml.metrics import trustworthiness as cuml_trustworthiness
import cupy as cp
import pandas as pd

# Dataset class for interacting with IPC Remote Psana
class IPCRemotePsanaDataset(Dataset):
    def __init__(self, server_address, requests_list):
        """Initialize the dataset with server address and requests."""
        self.server_address = server_address
        self.requests_list = requests_list

    def __len__(self):
        return len(self.requests_list)

    def __getitem__(self, idx):
        request = self.requests_list[idx]
        return self.fetch_event(*request)

    def fetch_event(self, exp, run, access_mode, detector_name, event):
        """Fetch an event from the server and access shared memory."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(self.server_address)
            request_data = json.dumps({
                'exp': exp,
                'run': run,
                'access_mode': access_mode,
                'detector_name': detector_name,
                'event': event,
                'mode': 'calib'
            })
            sock.sendall(request_data.encode('utf-8'))
            response_data = sock.recv(4096).decode('utf-8')
            response_json = json.loads(response_data)

            shm_name = response_json['name']
            shape = response_json['shape']
            dtype = np.dtype(response_json['dtype'])

            shm = shared_memory.SharedMemory(name=shm_name)
            try:
                data_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                result = np.array(data_array)
            finally:
                shm.close()
                shm.unlink()

            sock.sendall("ACK".encode('utf-8'))
            return result

# Process function for dimensionality reduction
def process(rank, proj, device_list, num_tries, threshold, type_of_embedding):
    """Perform t-SNE or UMAP on the projection data."""
    proj = torch.tensor(proj, device=device_list[rank % 4])
    torch.cuda.empty_cache()
    gc.collect()

    trustworthiness_threshold = threshold
    best_params = None
    best_score = 0

    if type_of_embedding == "umap":
        # UMAP
        for i in range(num_tries):
            if i % 10 == 0:
                print(f"UMAP fitting on GPU {rank} iteration {i}", flush=True)
            n_neighbors = np.random.randint(5, 200)
            min_dist = np.random.uniform(0.0, 0.99)
            umap = cumlUMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
            embedding = umap.fit_transform(proj)
            trustworthiness = cuml_trustworthiness(proj, embedding)

            if trustworthiness > trustworthiness_threshold:
                best_params = (n_neighbors, min_dist)
                best_score = trustworthiness
                break
            elif trustworthiness > best_score:
                best_params = (n_neighbors, min_dist)
                best_score = trustworthiness

    elif type_of_embedding == "t-sne":
        # t-SNE
        for i in range(num_tries):
            if i % 10 == 0:
                print(f"t-SNE fitting on GPU {rank} iteration {i}", flush=True)
            perplexity = np.random.randint(5, 50)
            n_neighbors = np.random.randint(3 * perplexity, 6 * perplexity)
            learning_rate = np.random.uniform(10, 1000)
            tsne = TSNE(n_components=2, perplexity=perplexity, n_neighbors=n_neighbors, learning_rate=learning_rate)
            embedding = tsne.fit_transform(proj)
            trustworthiness = cuml_trustworthiness(proj, embedding)

            if trustworthiness > trustworthiness_threshold:
                best_params = (n_neighbors, perplexity)
                best_score = trustworthiness
                break
            elif trustworthiness > best_score:
                best_params = (n_neighbors, perplexity)
                best_score = trustworthiness

    else:
        raise Exception("Sorry, dimension reduction algorithm not recognized. Choose between umap and t-sne")

    embedding = cp.asnumpy(embedding)

    return embedding, best_params, best_score

# Function to calculate projectors
def get_projectors(rank, imgs, V, device_list, mu, first_rank=0):
    """Compute projectors on given data."""
    V = torch.tensor(V, device=device_list[rank % 4])
    imgs = torch.tensor(imgs.reshape(imgs.shape[0], -1), device=device_list[rank % 4])
    mu = torch.tensor(mu, device=device_list[rank % 4])
    proj = torch.mm(imgs - mu[rank], V[:,first_rank:])
    return proj.cpu().detach().numpy()

# Function to bin embeddings
def binning_indices(embedding, grid_size=50):
    """Create bins for embeddings based on grid size, including empty bins."""
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    x_bin_size = (x_max - x_min) / grid_size
    y_bin_size = (y_max - y_min) / grid_size
    bins = {}

    # Initialize all bins as empty
    for i in range(grid_size):
        for j in range(grid_size):
            x_center = x_min + (i + 0.5) * x_bin_size
            y_center = y_min + (j + 0.5) * y_bin_size
            bin_key = (x_center, y_center)
            bins[bin_key] = []

    # Populate bins with embedding indices
    for index, (x, y) in enumerate(embedding):
        x_bin = min(int((x - x_min) / x_bin_size), grid_size - 1)
        y_bin = min(int((y - y_min) / y_bin_size), grid_size - 1)
        x_center = x_min + (x_bin + 0.5) * x_bin_size
        y_center = y_min + (y_bin + 0.5) * y_bin_size
        bin_key = (x_center, y_center)
        bins[bin_key].append(index)

    return bins
    
# Create average projections
def create_average_proj(proj_list, bins):
    """Calculate average projection for each bin."""
    proj_binned = {}
    weights = {}
    proj_list = np.array(proj_list)
    print("Number of keys (ie number of bins):",len(list(bins.keys())),flush=True)
    for key, indices in bins.items():
        proj_binned[key] = []
        if key in weights:
            weights[key]+=len(indices)
        else:
            weights[key]=len(indices)
            
        if indices:
            for rank in range(proj_list.shape[0]):
                list_proj = [proj_list[rank][indice//proj_list.shape[0]] for indice in indices]
                if list_proj:
                    proj_binned[key].append(np.mean(list_proj, axis=0))
                else:
                    proj_binned[key] = []
    return proj_binned, weights

# Unpack PiPCA model file
def unpack_ipca_pytorch_model_file(filename):
    """Read and unpack PiPCA model file."""
    data = {}
    with h5py.File(filename, 'r') as f:
        metadata = f['metadata']
        data['exp'] = str(np.asarray(metadata.get('exp')))[2:-1]
        data['run'] = int(np.asarray(metadata.get('run')))
        data['det_type'] = str(np.asarray(metadata.get('det_type')))[2:-1]
        data['start_offset'] = int(np.asarray(metadata.get('start_offset')))
        data['S'] = np.asarray(f['S'])
        data['num_images'] = int(np.asarray(metadata.get('num_images')))
    return data

# Parse command-line arguments
def parse_input():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", required=True, type=str)
    parser.add_argument("-n", "--num_images", required=True)
    parser.add_argument("-l", "--loading_batch_size", required=True, type=int)
    parser.add_argument("--num_tries", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--num_runs", type=list)
    parser.add_argument("--grid_size", type=int)
    parser.add_argument("--guiding_panel", type=int)
    parser.add_argument("--type_of_embedding", type=str)
    parser.add_argument("--first_rank", type=int)
    return parser.parse_args()

# Main execution logic
def main():
    params = parse_input()
    filename = params.filename
    num_images = json.loads(params.num_images)
    loading_batch_size = params.loading_batch_size
    threshold = params.threshold
    num_tries = params.num_tries
    num_runs = params.num_runs
    grid_size = params.grid_size
    guiding_panel = params.guiding_panel
    type_of_embedding = params.type_of_embedding
    first_rank = params.first_rank
    
    print("Unpacking model file...", flush=True)
    data = unpack_ipca_pytorch_model_file(filename)

    exp, run, det_type = data['exp'], data['run'], data['det_type']
    start_img, S = data['start_offset'], data['S']
    num_gpus, num_components = S.shape
    mp.set_start_method('spawn', force=True)

    list_proj, list_proj_rank = [], [[] for _ in range(num_gpus)]

    for current_run in range(run, run + len(num_runs)):
        for event in range(start_img, start_img + num_images[current_run - run], loading_batch_size):
            requests_list = [(exp, current_run, 'idx', det_type, img) for img in range(event, min(event + loading_batch_size, num_images[current_run - run]))]
            server_address = ('localhost', 5000)
            dataset = IPCRemotePsanaDataset(server_address=server_address, requests_list=requests_list)
            dataloader = DataLoader(dataset, batch_size=20, num_workers=4, prefetch_factor=None)

            list_images = [batch for batch in dataloader]
            list_images = np.concatenate(list_images, axis=0)
            list_images = list_images[[i for i in range (list_images.shape[0]) if not np.isnan(list_images[i : i + 1]).any()]]
            list_images = np.split(list_images, num_gpus, axis=1)

            device_list = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(num_gpus)]
            with Pool(processes=num_gpus) as pool:
                with h5py.File(filename, 'r') as f:
                    V, mu = f['V'], f['mu'][:]
                    proj = pool.starmap(get_projectors, [(rank, list_images[rank], V[rank, :, :], device_list, mu,first_rank) for rank in range(num_gpus)])

            rank_proj_list = [u for u in proj]
            list_proj.append(np.vstack(rank_proj_list))
            for i in range(num_gpus):
                list_proj_rank[i].append(rank_proj_list[i])

    list_proj = np.vstack(list_proj)
    for i in range(num_gpus):
        list_proj_rank[i] = np.vstack(list_proj_rank[i])

    with Pool(processes=num_gpus) as pool:
        embeddings, best_params, best_score = pool.apply(process, (0, list_proj, device_list, num_tries, threshold, type_of_embedding))
        results = pool.starmap(process, [(i, list_proj_rank[i], device_list, num_tries, threshold, type_of_embedding) for i in range(num_gpus)])
    
    embeddings_rank = [[] for _ in range(num_gpus)]
    for i, (embedding, params, score) in enumerate(results):
        embeddings_rank[i] = embedding
    
    if guiding_panel == -1:
        bins = binning_indices(embeddings, grid_size=grid_size)
    else:
        guiding_panel %= num_gpus
        bins = binning_indices(embeddings_rank[guiding_panel], grid_size=grid_size)
    
    proj_binned,weights = create_average_proj(list_proj_rank, bins)
    
    with h5py.File(f"../../visuals/embeddings/binned_data_{num_components}_{num_images}_{type_of_embedding}.h5", "w") as hdf:
        group = hdf.create_group(f'proj_binned')
        weights_group = hdf.create_group(f'weights')
        for key, value in proj_binned.items():
            group.create_dataset("_".join(map(str, key)), data=value)
            weights_group.create_dataset("_".join(map(str, key)), data=weights[key])
            
    data = {
        f"embeddings": embeddings,
        "S": S,
        f"embeddings_rank": embeddings_rank
    }
    with open(f"../../visuals/embeddings/embedding_data_{num_components}_{num_images}_{type_of_embedding}.pkl", "wb") as f:
        pickle.dump(data, f)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(server_address)
        sock.sendall("DONE".encode('utf-8'))

if __name__ == "__main__":
    main()

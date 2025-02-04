import json
import socket
import time
import requests
import io
import numpy as np
import argparse
import time
import os
import sys
import psutil
from multiprocessing import shared_memory, Pool
import torch 
import torch.nn as nn
import torch.multiprocessing as mp
import logging
import gc
import h5py
import csv
import ast

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from processing.pca_on_gpu.pca_module import IncrementalPCAonGPU

class IPCRemotePsanaDataset(Dataset):
    """
    Dataset for fetching data from a remote Psana server using IPC.
    
    Parameters:
        server_address (tuple or str): Address of the server. For UNIX sockets, it's the path to the socket. For TCP, it's a tuple (host, port).
        requests_list (list of tuples): Each tuple contains (exp, run, access_mode, detector_name, event).
    """

    def __init__(self, server_address, requests_list):
        self.server_address = server_address
        self.requests_list = requests_list

    def __len__(self):
        return len(self.requests_list)

    def __getitem__(self, idx):
        request = self.requests_list[idx]
        return self.fetch_event(*request)

    def fetch_event(self, exp, run, access_mode, detector_name, event):
        """Fetch event data from the server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(self.server_address)
            
            # Send request
            request_data = json.dumps({
                'exp': exp,
                'run': run,
                'access_mode': access_mode,
                'detector_name': detector_name,
                'event': event,
                'mode': 'calib',
            })
            sock.sendall(request_data.encode('utf-8'))

            # Receive response
            response_data = sock.recv(4096).decode('utf-8')
            response_json = json.loads(response_data)

            # Access shared memory
            shm_name = response_json['name']
            shape = response_json['shape']
            dtype = np.dtype(response_json['dtype'])

            shm = None
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                data_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                result = np.array(data_array)  # Create a copy of the data
            finally:
                if shm:
                    shm.close()
                    shm.unlink()

            # Acknowledge shared memory access
            sock.sendall("ACK".encode('utf-8'))
            return result

def parse_input():
    """
    Parses command-line arguments and returns them as a namespace object.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--exp", help="Experiment name.", required=True, type=str)
    parser.add_argument("-r", "--run", help="Run number.", required=True, type=int)
    parser.add_argument("-d", "--det_type", help="Detector name, e.g. epix10k2M or jungfrau4M.", required=True, type=str)
    parser.add_argument("--num_images", help="Total number of images per run to be incorporated into model.", required=True, type=str)
    parser.add_argument("--loading_batch_size", help="Batch size used when loading images on the client.", required=True, type=int)
    parser.add_argument("--batch_size", help="Batch size for iPCA algorithm.", required=True, type=int)
    parser.add_argument("--num_runs", help="Number of runs to process.", required=True, type=int)
    parser.add_argument("--lower_bound", help="Lower bound for the loss.", required=True, type=float)
    parser.add_argument("--upper_bound", help="Upper bound for the loss.", required=True, type=float)
    parser.add_argument("--model", help="Path to the model file.", required=True, type=str)
    parser.add_argument("--num_gpus", help="Number of GPUs to use.", required=True, type=int)
    parser.add_argument("--start_offset", help="Run index of first image to be incorporated into iPCA model.", type=int)
    parser.add_argument("--num_nodes", help="Number of nodes to use.", type=int)
    parser.add_argument("--id_current_node", help="ID of the current node.", type=int)
    parser.add_argument("--num_components", help="Number of components to create in case of new model", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    # Parse input arguments
    params = parse_input()

    # Extract parameters
    exp = params.exp
    init_run = params.run
    det_type = params.det_type
    start_offset = params.start_offset or 0
    batch_size = params.batch_size
    filename = params.model
    #num_gpus = params.num_gpus
    num_runs = params.num_runs
    lower_bound = params.lower_bound
    upper_bound = params.upper_bound
    id_current_node = params.id_current_node
    num_nodes = params.num_nodes
    num_tot_gpus = num_gpus * num_nodes
    # Parse number of images per run
    num_images = json.loads(params.num_images)
    num_images_to_add = sum(num_images)
    loading_batch_size = params.loading_batch_size

    # Initialize multiprocessing
    mp.set_start_method('spawn', force=True)

    # Load the initial model
    if not os.path.isdir(filename):
        data = read_model_file(filename, id_current_node, num_gpus)
    all_norm_diff = []
    all_init_norm = []
    last_batch = False

    # Use multiprocessing manager for shared state dictionaries
    with mp.Manager() as manager:
        model_state_dict = [manager.dict() for _ in range(num_gpus)]

        # Initialize model state for each GPU
        for rank in range(num_gpus):
            if not os.path.isdir(filename):
                model_state_dict[rank].update({
                    'V': data['V'][rank],
                    'mu': data['mu'][rank],
                    'S': data['S'][rank],
                    'num_images': data['num_images'],
                    'num_components': data['num_components']
                })
            else:
                model_state_dict[rank].update({'num_images' : 0, 'num_components' : params.num_components, 'V': None, 'S':None, 'mu':None})

        print("Model loaded", flush=True)

        # Create a pool of processes for parallel computation
        with Pool(processes=num_gpus) as pool:
            num_images_seen = 0

            # Process each run
            for run in range(init_run, init_run + num_runs):
                for event in range(start_offset, start_offset + num_images[run - init_run], loading_batch_size):
                    # Handle the last batch
                    if num_images_seen + loading_batch_size >= num_images_to_add:
                        last_batch = True

                    # Load and preprocess images
                    current_loading_batch = []
                    all_norm_diff.append([])
                    all_init_norm.append([])

                    requests_list = [
                        (exp, run, 'idx', det_type, img)
                        for img in range(event, min(event + loading_batch_size, start_offset+num_images[run - init_run]))
                    ]
                    server_address = ('localhost', 5000)
                    dataset = IPCRemotePsanaDataset(server_address=server_address, requests_list=requests_list)
                    dataloader = DataLoader(dataset, batch_size=20, num_workers=4, prefetch_factor=None)

                    # Collect batches
                    for batch in dataloader:
                        current_loading_batch.append(batch)
                        if num_images_seen + len(current_loading_batch) >= num_images_to_add:
                            last_batch = True
                            break
                        
                    # Concatenate and filter valid images              
                    current_loading_batch = np.concatenate(current_loading_batch, axis=0)
                    current_len = current_loading_batch.shape[0]
                    num_images_seen += current_len

                    print(f"Loaded {event + current_len} images from run {run}.", flush=True)
                    print("Number of images seen:", num_images_seen, flush=True)

                    current_loading_batch = current_loading_batch[
                        [i for i in range(current_len) if not np.isnan(current_loading_batch[i : i + 1]).any()]
                    ]
                    current_len = current_loading_batch.shape[0]

                    print(f"Number of non-None images in the current batch: {current_loading_batch.shape[0]}", flush=True)
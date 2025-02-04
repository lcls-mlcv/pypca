#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

def append_to_dataset(f, dataset_name, data):
    """Append or create a dataset in an HDF5 file."""
    if dataset_name not in f:
        f.create_dataset(dataset_name, data=np.array(data))
    else:
        existing_data = np.atleast_1d(f[dataset_name][()])
        new_data = np.atleast_1d(np.array(data))
        combined_data = np.concatenate([existing_data, new_data])
        del f[dataset_name]
        f.create_dataset(dataset_name, data=combined_data)

def create_or_update_dataset(f, name, data):
    """Create or update a dataset in an HDF5 file."""
    if name in f:
        del f[name]
    f.create_dataset(name, data=data)

def create_shared_images(images):
    """Create shared memory for a list of images."""
    shm_list = []
    for sub_imgs in images:
        chunk_size = np.prod(images[0].shape) * images[0].dtype.itemsize
        shm = shared_memory.SharedMemory(create=True, size=chunk_size)
        shm_images = np.ndarray(sub_imgs.shape, dtype=sub_imgs.dtype, buffer=shm.buf)
        np.copyto(shm_images, sub_imgs)
        shm_list.append(shm)
    return shm_list

def read_model_file(filename, id_current_node=0, num_gpus=4, num_components=1):
    """
    Read iPCA model information from an HDF5 file.
    
    Parameters:
        filename (str): Name of the HDF5 file.
        id_current_node (int): Current node ID.
        num_gpus (int): Number of GPUs.
    
    Returns:
        dict: A dictionary containing the model data.
    """
    data = {}
    with h5py.File(filename, 'r') as f:
        V = np.asarray(f['V'])
        data['V'] = V[id_current_node * num_gpus:(id_current_node + 1) * num_gpus]
        data['mu'] = np.asarray(f['mu'])[id_current_node * num_gpus:(id_current_node + 1) * num_gpus]
        data['S'] = np.asarray(f['S'])[id_current_node * num_gpus:(id_current_node + 1) * num_gpus]
        metadata = f['metadata']
        data['num_images'] = metadata['num_images'][()]
        data['num_components'] = data['V'].shape[2]
        
    return data

def compute_loss_process(rank, model_state_dict, shm_list, device_list, shape, dtype, batch_size):
    """
    Compute reconstruction loss for a batch of images.
    
    Parameters:
        rank (int): Rank of the process.
        model_state_dict (dict): State dictionary of the iPCA model.
        shm_list (list): List of shared memory objects containing images.
        device_list (list): List of devices for computation.
        shape (tuple): Shape of the images.
        dtype (numpy.dtype): Data type of the images.
        batch_size (int): Batch size for processing.
    
    Returns:
        tuple: Lists of reconstruction loss and initial norms.
    """
    device = device_list[rank]
    V = torch.tensor(model_state_dict[rank]['V'], device=device)
    mu = torch.tensor(model_state_dict[rank]['mu'], device=device)

    # Load shared memory images
    existing_shm = shared_memory.SharedMemory(name=shm_list[rank].name)
    images = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    list_norm_diff = torch.tensor([], device=device)
    list_init_norm = torch.tensor([], device=device)

    for start in range(0, images.shape[0], batch_size):
        end = min(start + batch_size, images.shape[0])
        batch_imgs = torch.tensor(images[start:end].reshape(end - start, -1), device=device)
        initial_norm = torch.norm(batch_imgs, dim=1, p='fro')
        
        transformed_batch = torch.mm(batch_imgs - mu, V)
        reconstructed_batch = torch.mm(transformed_batch, V.T) + mu

        diff = batch_imgs - reconstructed_batch
        norm_batch = torch.norm(diff, dim=1, p='fro')

        list_norm_diff = torch.cat((list_norm_diff, norm_batch), dim=0)
        list_init_norm = torch.cat((list_init_norm, initial_norm), dim=0)

    list_norm_diff = list_norm_diff.cpu().detach().numpy()
    list_init_norm = list_init_norm.cpu().detach().numpy()

    torch.cuda.empty_cache()
    gc.collect()
    return list_norm_diff, list_init_norm

def compute_total_loss(all_norm_diff, all_init_norm, num_gpus):
    """
    Compute total loss across all GPUs.
    
    Parameters:
        all_norm_diff (list): List of norm differences for each GPU.
        all_init_norm (list): List of initial norms for each GPU.
        num_gpus (int): Number of GPUs.
    
    Returns:
        numpy.ndarray: Total loss array.
    """
    all_losses = []
    for k in range(len(all_init_norm)):
        i = np.zeros_like(all_init_norm[k][0])
        d = np.zeros_like(all_norm_diff[k][0])
        
        for rank in range(num_gpus):
            i += all_init_norm[k][rank] ** 2
            d += all_norm_diff[k][rank] ** 2

        all_losses.append(np.sqrt(d) / np.sqrt(i))

    return np.concatenate(all_losses, axis=0)

def indices_to_update(losses, lower_bound=0, upper_bound=1e9):
    """
    Find indices of images requiring updates based on loss bounds.
    
    Parameters:
        losses (numpy.ndarray): Loss values for all images.
        lower_bound (float): Minimum loss threshold.
        upper_bound (float): Maximum loss threshold.
    
    Returns:
        numpy.ndarray: Indices of images within the specified loss bounds.
    """
    return np.where((losses >= lower_bound) & (losses <= upper_bound))[0]

def compute_new_model(model_state_dict, batch_size, device_list, rank, shm_list, shape, dtype, indices_to_update):
    """
    Update iPCA model using new data.
    
    Parameters:
        model_state_dict (dict): State dictionary of the iPCA model.
        batch_size (int): Batch size for processing.
        device_list (list): List of devices for computation.
        rank (int): Rank of the process.
        shm_list (list): List of shared memory objects containing images.
        shape (tuple): Shape of the images.
        dtype (numpy.dtype): Data type of the images.
        indices_to_update (numpy.ndarray): Indices of images to use for updating.
    
    Returns:
        dict: Updated model state dictionary.
    """

    # Initialize IncrementalPCAonGPU
    num_components = model_state_dict[rank]['num_components']
    num_images = model_state_dict[rank]['num_images']

    device = device_list[rank]
    ipca = IncrementalPCAonGPU(n_components=num_components, batch_size=batch_size, device=device)
    if model_state_dict[rank].get("V") is not None: ##### TEST
        ipca.components_ = torch.tensor(model_state_dict[rank]['V'].T, device=device)
        ipca.mean_ = torch.tensor(model_state_dict[rank]['mu'], device=device)
        ipca.n_samples_seen_ = num_images
        ipca.singular_values_ = torch.tensor(model_state_dict[rank]['S'], device=device)

    # Load images from shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_list[rank].name)
    images = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    images = images[indices_to_update]

    # Fit the updated model
    start_time = time.time()
    ipca.fit(images.reshape(len(indices_to_update), -1))

    existing_shm.close()
    existing_shm.unlink()
    
    # Update model state
    model_state_dict[rank]['V'] = ipca.components_.T.cpu().detach().numpy()
    model_state_dict[rank]['mu'] = ipca.mean_.cpu().detach().numpy()
    model_state_dict[rank]['S'] = ipca.singular_values_.cpu().detach().numpy()
    model_state_dict[rank]['num_images'] += len(indices_to_update)

    torch.cuda.empty_cache()
    gc.collect()
    
    return model_state_dict

def update_model(model, model_state_dict, id_current_node=0, metadata=None):
    """
    Updates the iPCA model and saves it to a file.
    
    Parameters:
        model (str): Path to the model file. If just a directory, means no previous model was used and have to create one.
        model_state_dict (list): List of state dictionaries for each GPU.
        id_current_node (int): ID of the current node.
    """
    # Gather updated model components from all GPUs
    S, V, mu = [], [], []
    for rank in range(num_gpus):
        S.append(model_state_dict[rank]['S'])
        V.append(model_state_dict[rank]['V'])
        mu.append(model_state_dict[rank]['mu'])

    new_num_images = model_state_dict[0]['num_images']

    # Define output path
    if not os.path.isdir(model):
        model_path = os.path.dirname(model)
        metadata=None
    else:
        model_path = model
    output_path = os.path.join(model_path, f'node_{id_current_node}_model.h5')

    # Save the updated model to file
    with h5py.File(output_path, 'w') as f:
        create_or_update_dataset(f, 'V', data=V)
        create_or_update_dataset(f, 'mu', data=mu)
        create_or_update_dataset(f, 'S', data=S)
        create_or_update_dataset(f, 'num_images', data=new_num_images)
        if metadata:
            create_or_update_dataset(f, 'num_runs', data=metadata['num_runs'])
            create_or_update_dataset(f, 'run', data=metadata['run'])
            create_or_update_dataset(f, 'exp', data=metadata['exp'])
            create_or_update_dataset(f, 'start_offset', data=metadata['start_offset'])
            create_or_update_dataset(f, 'det_type', data=metadata['det_type'])

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
    num_gpus = params.num_gpus
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

    # Initializes timers
    loading_time=0
    formatting_time=0
    fitting_time=0
    model_loading_time=0
    model_updating_time=0
    loss_computing_time=0

    # Load the initial model
    if not os.path.isdir(filename):
        model_loading_time-=time.time()
        data = read_model_file(filename, id_current_node, num_gpus)
        model_loading_time+=time.time()
        
    all_norm_diff = []
    all_init_norm = []
    last_batch = False

    # Use multiprocessing manager for shared state dictionaries
    with mp.Manager() as manager:
        model_state_dict = [manager.dict() for _ in range(num_gpus)]

        # Initialize model state for each GPU
        model_loading_time -= time.time()
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
        model_loading_time +=time.time()
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

                    prev_loading_time=loading_time
                    loading_time-=time.time()
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
                    dataloader = DataLoader(dataset, batch_size=50, num_workers=2, prefetch_factor=None)

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
                    loading_time+=time.time()
                    print(f"Loaded {event + current_len} images from run {run} in {loading_time-prev_loading_time} (s).", flush=True)
                    print("Number of images seen:", num_images_seen, flush=True)

                    formatting_time-=time.time()
                    current_loading_batch = current_loading_batch[
                        [i for i in range(current_len) if not np.isnan(current_loading_batch[i : i + 1]).any()]
                    ]
                    current_len = current_loading_batch.shape[0]

                    print(f"Number of non-None images in the current batch: {current_loading_batch.shape[0]}", flush=True)

                    # Split images across GPUs and nodes
                    current_loading_batch = np.split(current_loading_batch, num_tot_gpus, axis=1)
                    current_loading_batch = current_loading_batch[id_current_node * num_gpus : (id_current_node + 1) * num_gpus]

                    # Create shared memory for each batch
                    shm_list = create_shared_images(current_loading_batch)
                    print("Images split and on shared memory", flush=True)
                    datashape,datadtype= current_loading_batch[0].shape,current_loading_batch[0].dtype
                    current_loading_batch = []
                    gc.collect()
                    
                    formatting_time+=time.time()
                    
                    device_list = [
                        torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(num_gpus)
                    ]

                    # Compute loss or select indices to update based on bounds
                    if lower_bound == -1 and upper_bound == -1:
                        indices = range(current_len)
                    else:
                        loss_computing_time-=time.time()
                        results = pool.starmap(
                            compute_loss_process,
                            [
                                (rank, model_state_dict, shm_list, device_list, datashape, 
                                 datadtype, batch_size)
                                for rank in range(num_gpus)
                            ]
                        )

                        print("Loss computed", flush=True)

                        for rank in range(num_gpus):
                            list_norm_diff, list_init_norm = results[rank]
                            all_norm_diff[-1].append(list_norm_diff)
                            all_init_norm[-1].append(list_init_norm)

                        total_losses = compute_total_loss(all_norm_diff, all_init_norm)
                        indices = indices_to_update(total_losses, lower_bound, upper_bound)

                        all_norm_diff.clear()
                        all_init_norm.clear()
                        loss_computing_time+=time.time()
                        
                        print(f"Number of images to update: {len(indices)}", flush=True)

                    # Update the model if necessary
                    if indices:
                        fitting_time-=time.time()
                        print("Updating model", flush=True)
                        results = pool.starmap(
                            compute_new_model,
                            [
                                (model_state_dict, batch_size, device_list, rank, shm_list, datashape,
                                 datadtype, indices)
                                for rank in range(num_gpus)
                            ]
                        )

                        print("New model computed", flush=True)
                        fitting_time+=time.time()
                        if last_batch:
                            print("Last batch", flush=True)
                            break
                    else:
                        print("No images to update", flush=True)
                        if last_batch:
                            print("Last batch", flush=True)
                            break

                if last_batch:
                    break

        # Save the updated model
        if indices:
            model_updating_time-=time.time()
            metadata = {'exp':params.exp,'run':params.run,'num_runs':params.num_runs,'start_offset':params.start_offset,'det_type':params.det_type}
            update_model(filename, model_state_dict, id_current_node, metadata=metadata)
            model_updating_time+=time.time()
            print("Model updated and saved", flush=True)

        print("Process finished", flush=True)
        print(f"Image loading time: {loading_time}(s) \n Model loading time: {model_loading_time} (s) \n Model updating time: {model_updating_time} (s) \n Formatting time: {formatting_time} (s) \n Fitting time: {fitting_time} (s) \n Loss computing time: {loss_computing_time} (s)",flush=True)
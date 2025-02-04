#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import socket
import time
import numpy as np
import argparse
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
from itertools import chain
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class IPCRemotePsanaDataset(Dataset):
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
            
            # Prepare and send request
            request_data = json.dumps({
                'exp': exp, 'run': run, 'access_mode': access_mode,
                'detector_name': detector_name, 'event': event, 'mode': 'calib',
            })
            sock.sendall(request_data.encode('utf-8'))
            
            # Receive and process response
            response_data = sock.recv(4096).decode('utf-8')
            response_json = json.loads(response_data)
            
            # Extract shared memory information
            shm_name = response_json['name']
            shape = response_json['shape']
            dtype = np.dtype(response_json['dtype'])
            fiducial = int(response_json['fiducial'])
            time = int(response_json['time'])
            nanoseconds = int(response_json['nanoseconds'])
            seconds = int(response_json['seconds'])
            
            shm = None
            try:
                # Access shared memory
                shm = shared_memory.SharedMemory(name=shm_name)
                data_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                result = [np.array(data_array), fiducial, time, nanoseconds, seconds]
            finally:
                if shm:
                    shm.close()
                    shm.unlink()
            
            # Send acknowledgment
            sock.sendall("ACK".encode('utf-8'))
            
            return result

def append_to_dataset(f, dataset_name, data):
    """Append data to an existing dataset or create a new one."""
    if dataset_name not in f:
        f.create_dataset(dataset_name, data=np.array(data))
    else:
        if isinstance(f[dataset_name], h5py.Dataset) and f[dataset_name].shape == ():
            existing_data = np.atleast_1d(f[dataset_name][()])
        else:
            existing_data = f[dataset_name][:]
        new_data = np.atleast_1d(np.array(data))
        data_combined = np.concatenate([existing_data, new_data])
        del f[dataset_name]
        f.create_dataset(dataset_name, data=data_combined)

def create_shared_images(images):
    """Create shared memory for images."""
    shm_list = []
    for sub_imgs in images:
        chunk_size = np.prod(images[0].shape) * images[0].dtype.itemsize
        shm = shared_memory.SharedMemory(create=True, size=chunk_size)
        shm_images = np.ndarray(sub_imgs.shape, dtype=sub_imgs.dtype, buffer=shm.buf)
        np.copyto(shm_images, sub_imgs)
        shm_list.append(shm)
    return shm_list

def read_model_file(filename, id_current_node=0, num_gpus=4):
    """Read PiPCA model information from h5 file."""
    data = {}
    with h5py.File(filename, 'r') as f:
        data['V'] = np.asarray(f.get('V'))[id_current_node*num_gpus:(id_current_node+1)*num_gpus]
        data['mu'] = np.asarray(f.get('mu'))[id_current_node*num_gpus:(id_current_node+1)*num_gpus]
    return data

def reduce_images(V, mu, batch_size, device_list, rank, shm_list, shape, dtype):
    """Reduce images using the iPCA model."""
    device = device_list[rank]
    V = torch.tensor(V[rank], device=device)
    mu = torch.tensor(mu[rank], device=device)
    
    existing_shm = shared_memory.SharedMemory(name=shm_list[rank].name)
    images = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    transformed_images = []
    for start in range(0, images.shape[0], batch_size):
        end = min(start + batch_size, images.shape[0])
        batch = images[start:end]
        batch = torch.tensor(batch.reshape(end-start, -1), device=device)
        transformed_batch = torch.mm((batch - mu).float(), V.float())
        transformed_images.append(transformed_batch)
    
    transformed_images = torch.cat(transformed_images, dim=0)
    return transformed_images.cpu().numpy()

def parse_input():
    """Parse command line input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", help="Experiment name.", required=True, type=str)
    parser.add_argument("-r", "--run", help="Run number.", required=True, type=int)
    parser.add_argument("-d", "--det_type", help="Detector name, e.g epix10k2M or jungfrau4M.", required=True, type=str)
    parser.add_argument("--start_offset", help="Run index of first image to be incorporated into iPCA model.", required=False, type=int)
    parser.add_argument("--num_images", help="Total number of images per run to be incorporated into model.", required=True, type=str)
    parser.add_argument("--loading_batch_size", help="Size of the batches used when loading the images on the client.", required=True, type=int)
    parser.add_argument("--batch_size", help="Batch size for incremental transformation algorithm.", required=True, type=int)
    parser.add_argument("--num_runs", help="Number of runs to process.", required=True, type=int)
    parser.add_argument("--model", help="Path to the model file.", required=True, type=str)
    parser.add_argument("--num_gpus", help="Number of GPUs to use.", required=True, type=int)
    parser.add_argument("--num_nodes", help="Number of nodes to use.", required=False, type=int)
    parser.add_argument("--id_current_node", help="ID of the current node.", required=False, type=int)
    return parser.parse_args()

def process_batch(current_loading_batch, fiducials, times, nanoseconds, seconds):
    """Process a batch of loaded images."""
    current_len = current_loading_batch.shape[0]
    valid_indices = [i for i in range(current_len) if not np.isnan(current_loading_batch[i : i + 1]).any()]
    current_loading_batch = current_loading_batch[valid_indices]
    fiducials = [fiducials[i] for i in valid_indices]
    times = [times[i] for i in valid_indices]
    nanoseconds = [nanoseconds[i] for i in valid_indices]
    seconds = [seconds[i] for i in valid_indices]
    return current_loading_batch, fiducials, times, nanoseconds, seconds

def main():
    # Parse input parameters
    params = parse_input()
    exp, init_run, det_type = params.exp, params.run, params.det_type
    start_offset = params.start_offset if params.start_offset is not None else 0
    batch_size, filename = params.batch_size, params.model
    num_gpus, num_runs = params.num_gpus, params.num_runs
    id_current_node, num_nodes = params.id_current_node, params.num_nodes
    num_tot_gpus = num_gpus * num_nodes
    num_images = json.loads(params.num_images)
    num_images_to_add = sum(num_images)
    loading_batch_size = params.loading_batch_size

    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)

    # Read current model
    data = read_model_file(filename, id_current_node, num_gpus)
    V, mu = data['V'], data['mu']
    num_components=V.shape[2]
    fiducials_list, times_list, nanoseconds_list, seconds_list = [], [], [], []
    last_batch = False
    projected_images = [[] for _ in range(num_gpus)]

    loading_time=0
    formatting_time=0
    transforming_time=0
    saving_time=0
    
    with Pool(processes=num_gpus) as pool:
        num_images_seen = 0
        for run in range(init_run, init_run + num_runs):
            for event in range(start_offset, start_offset + num_images[run-init_run], loading_batch_size):
                if num_images_seen + loading_batch_size >= num_images_to_add:
                    last_batch = True

                prev_loading_time=loading_time
                loading_time-=time.time()
                # Prepare data loading
                requests_list = [(exp, run, 'idx', det_type, img) for img in range(event, min(event+loading_batch_size, num_images[run-init_run]))]
                
                # Set up dataset and dataloader
                server_address = ('localhost', 5000)
                dataset = IPCRemotePsanaDataset(server_address=server_address, requests_list=requests_list)
                dataloader = DataLoader(dataset, batch_size=50, num_workers=2, prefetch_factor=None)

                # Load data
                current_loading_batch, current_fiducials, current_times, current_nanoseconds, current_seconds = [], [], [], [], []
                for batch in dataloader:
                    current_loading_batch.append(batch[0])
                    current_fiducials.extend(batch[1])
                    current_times.extend(batch[2])
                    current_nanoseconds.extend(batch[3])
                    current_seconds.extend(batch[4])

                    if num_images_seen + len(current_loading_batch) >= num_images_to_add and current_loading_batch:
                        last_batch = True
                        break

                # Process loaded data
                current_loading_batch = np.concatenate(current_loading_batch, axis=0)
                current_loading_batch, current_fiducials, current_times, current_nanoseconds, current_seconds = process_batch(
                    current_loading_batch, current_fiducials, current_times, current_nanoseconds, current_seconds
                )

                loading_time+=time.time()
                num_images_seen += len(current_loading_batch)
                print(f"Loaded {event+len(current_loading_batch)} images from run {run} in {loading_time-prev_loading_time} (s)",flush=True)
                print(f"Number of images seen: {num_images_seen}",flush=True)
                print(f"Number of non-none images in the current batch: {current_loading_batch.shape[0]}",flush=True)

                fiducials_list.append(current_fiducials)
                times_list.append(current_times)
                nanoseconds_list.append(current_nanoseconds)
                seconds_list.append(current_seconds)

                formatting_time-=time.time()
                # Split images for GPUs
                current_loading_batch = np.split(current_loading_batch, num_tot_gpus, axis=1)
                current_loading_batch = current_loading_batch[id_current_node*num_gpus:(id_current_node+1)*num_gpus]
                shape, dtype = current_loading_batch[0].shape, current_loading_batch[0].dtype

                # Create shared memory for batches
                shm_list = create_shared_images(current_loading_batch)
                print("Images split and on shared memory",flush=True)

                current_loading_batch = []
                gc.collect()
                
                device_list = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(num_gpus)]
                formatting_time+=time.time()
                transforming_time-=time.time()
                # Reduce images
                results = pool.starmap(reduce_images, [(V, mu, batch_size, device_list, rank, shm_list, shape, dtype) for rank in range(num_gpus)])
                for rank in range(num_gpus):
                    projected_images[rank].append(results[rank])
                transforming_time+=time.time()
                
                # Clean up shared memory
                for shm in shm_list:
                    shm.close()
                    shm.unlink()

                torch.cuda.empty_cache()
                gc.collect()

                if last_batch:
                    break

            if last_batch:
                break

    formatting_time-=time.time()
    # Concatenate projected images
    for rank in range(num_gpus):
        projected_images[rank] = np.concatenate(projected_images[rank], axis=0)

    fiducials_list = np.concatenate(fiducials_list, axis=0)
    times_list = np.concatenate(times_list, axis=0)
    nanoseconds_list = np.concatenate(nanoseconds_list, axis=0)
    seconds_list = np.concatenate(seconds_list, axis=0)

    formatting_time+=time.time()
    saving_time-=time.time()
    
    # Save the projected images
    input_path = os.path.dirname(filename)
    output_path = os.path.join(input_path, f"projections_{init_run}_{init_run+num_runs-1}_{num_images_to_add}_{num_components}_node_{id_current_node}.h5")
    with h5py.File(output_path, 'w') as f:
        append_to_dataset(f, 'projected_images', projected_images)
        append_to_dataset(f, 'fiducials', fiducials_list)
        append_to_dataset(f, 'times', times_list)
        append_to_dataset(f, 'nanoseconds', nanoseconds_list)
        append_to_dataset(f, 'seconds', seconds_list)
        f.create_dataset('exp', data=exp)
        f.create_dataset('run', data=run)
        f.create_dataset('num_runs', data=num_runs)
        f.create_dataset('num_images', data=num_images_to_add)
        f.create_dataset('model_used', data=filename)
    saving_time+=time.time()
    print(f"Projections saved under the name projections_{init_run}_{init_run+num_runs-1}_{num_images_to_add}_{num_components}_node_{id_current_node}.h5",flush=True)
    print("Process finished",flush=True)
    print(f"Loading time: {loading_time}(s) \n Formatting time: {formatting_time} (s) \n Transforming time: {transforming_time} (s) \n Saving time: {saving_time} (s)",flush=True)

if __name__ == "__main__":
    main()

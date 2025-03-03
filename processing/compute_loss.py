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
import pickle

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
            shm = None
            try:
                # Access shared memory
                shm = shared_memory.SharedMemory(name=shm_name)
                data_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                result = np.array(data_array)
            finally:
                if shm:
                    shm.close()
                    shm.unlink()
            
            # Send acknowledgment
            sock.sendall("ACK".encode('utf-8'))
            
            return result

def append_to_dataset(f, dataset_name, data, chunks=None):
    """Append data to an existing dataset or create a new one."""
    if dataset_name not in f:
        if chunks:
            f.create_dataset(dataset_name, data=np.array(data), chunks=chunks)
        else:
            f.create_dataset(dataset_name, data=np.array(data))
    else:
        if isinstance(f[dataset_name], h5py.Dataset) and f[dataset_name].shape == ():
            existing_data = np.atleast_1d(f[dataset_name][()])
        else:
            existing_data = f[dataset_name][:]
        new_data = np.atleast_1d(np.array(data))
        data_combined = np.concatenate([existing_data, new_data])
        del f[dataset_name]
        if chunks:
            f.create_dataset(dataset_name, data=data_combined,chunks=chunks)
        else:
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

def compute_loss_for_components(projections_filename, V, mu, device_list, rank, shm_list, shape, dtype, id_current_node, event, component_steps):
    """Reconstructs images using the iPCA model then computes the loss for different numbers of components."""
    device = device_list[rank]
    V = torch.tensor(V[rank], device=device)
    mu = torch.tensor(mu[rank], device=device)
    
    existing_shm = shared_memory.SharedMemory(name=shm_list[rank].name)
    images = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    with h5py.File(projections_filename, 'r') as f:
        projected_imgs = f['projected_images']
        projected_imgs = torch.tensor(projected_imgs[rank+len(device_list)*id_current_node,event:event+images.shape[0],:],device=device)
    
    images_on_tensor = torch.tensor(images, device=device).reshape(projected_imgs.shape[0], -1)
    original_losses = torch.linalg.norm(images_on_tensor,dim=tuple(range(1, images_on_tensor.dim())))
    losses = {}
    
    prev_n_components=0
    for i, n_components in enumerate(sorted(component_steps)):
        projected_imgs_subset = projected_imgs[:, prev_n_components:n_components]
        V_subset = V[:, prev_n_components:n_components]
        if prev_n_components==0:
            rec_images = mu + torch.mm(projected_imgs_subset.float(), V_subset.T.float())
        else:
            rec_images += torch.mm(projected_imgs_subset.float(), V_subset.T.float())
        loss = torch.linalg.norm(rec_images - images_on_tensor, dim=1)#tuple(range(1, rec_images.dim()))  
        losses[n_components] = loss.cpu().numpy()
    
        prev_n_components = n_components
        if rank==0:
            print("Number of components treated for current loading batch:",n_components,flush=True)

    return (losses,original_losses.cpu().numpy())

def fuse_losses(rank_losses,device,key):
    losses = None
    for l in rank_losses:
        l_tensor = torch.tensor(l, device=device)
        if losses is not None:
            losses += l_tensor**2
        else:
            losses = l_tensor**2
    
    losses = torch.sqrt(losses)

    return key,losses.cpu().numpy()
        
def parse_input():
    """Parse command line input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", help="Experiment name.", required=True, type=str)
    parser.add_argument("-r", "--run", help="Run number.", required=True, type=int)
    parser.add_argument("-d", "--det_type", help="Detector name, e.g epix10k2M or jungfrau4M.", required=True, type=str)
    parser.add_argument("--start_offset", help="Run index of first image to be incorporated into iPCA model.", required=False, type=int)
    parser.add_argument("--num_images", help="Total number of images per run to be incorporated into model.", required=True, type=str)
    parser.add_argument("--loading_batch_size", help="Size of the batches used when loading the images on the client.", required=True, type=int)
    parser.add_argument("--num_runs", help="Number of runs to process.", required=True, type=int)
    parser.add_argument("--model", help="Path to the model file.", required=True, type=str)
    parser.add_argument("--num_gpus", help="Number of GPUs to use.", required=True, type=int)
    parser.add_argument("--num_nodes", help="Number of nodes to use.", required=False, type=int)
    parser.add_argument("--id_current_node", help="ID of the current node.", required=False, type=int)
    parser.add_argument("--projections_filename", help="Path to the projections file",required=True, type=str)
    return parser.parse_args()

def main():
    # Parse input parameters
    params = parse_input()
    exp, init_run, det_type = params.exp, params.run, params.det_type
    start_offset = params.start_offset if params.start_offset is not None else 0
    filename = params.model
    num_gpus, num_runs = params.num_gpus, params.num_runs
    id_current_node, num_nodes = params.id_current_node, params.num_nodes
    num_tot_gpus = num_gpus * num_nodes
    num_images = json.loads(params.num_images)
    num_images_to_add = sum(num_images)
    loading_batch_size = params.loading_batch_size
    projections_filename = params.projections_filename
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)

    # Read current model
    data = read_model_file(filename, id_current_node, num_gpus)
    V, mu = data['V'], data['mu']
    num_components=V.shape[2]
    last_batch = False
    component_steps = [1,2,3,4,5,10,20,30,50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000,12500,13000,13500,14000,14500,15000] #[1,2,3,4,5,10,20,30,50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500,3000,3500,4000,4500,5000] #[1,2,3,4,5,10,20,30,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
    dict_losses = {}
    for key in component_steps:
        dict_losses[key]= [[] for _ in range(num_gpus)]
    dict_losses['original']= [[] for _ in range(num_gpus)]
    loading_time=0
    formatting_time=0
    loss_computing_time=0
    num_skipped_events=0
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
                current_loading_batch = []
                for batch in dataloader:
                    current_loading_batch.append(batch)

                    if num_images_seen + len(current_loading_batch) >= num_images_to_add and current_loading_batch:
                        last_batch = True
                        break

                # Process loaded data
                current_loading_batch = np.concatenate(current_loading_batch, axis=0)

                loading_time+=time.time()
                num_images_seen += len(current_loading_batch)
                print(f"Loaded {event+len(current_loading_batch)} images from run {run} in {loading_time-prev_loading_time} (s)",flush=True)
                print(f"Number of images seen: {num_images_seen}",flush=True)
                print(f"Number of non-none images in the current batch: {current_loading_batch.shape[0]}",flush=True)
                num_skipped_events += loading_batch_size-current_loading_batch.shape[0]
                print(f"Total number of skipped events (none):{num_skipped_events}",flush=True)
                formatting_time-=time.time()
                # Split images for GPUs
                current_loading_batch = current_loading_batch.reshape(current_loading_batch.shape[0],num_tot_gpus, -1)
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
                loss_computing_time-=time.time()
                # Reduce images
                results = pool.starmap(compute_loss_for_components, [(projections_filename, V, mu, device_list, rank, shm_list, shape, dtype, id_current_node, event, component_steps) for rank in range(num_gpus)])

                for key in component_steps:
                    for rank in range(num_gpus):
                        losses,_= results[rank]
                        dict_losses[key][rank].append(losses[key])
                        
                for rank in range(num_gpus):
                    _,original_losses= results[rank]
                    dict_losses['original'][rank].append(original_losses)
                loss_computing_time+=time.time()
                
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
        print(dict_losses.keys())
        for key in dict_losses.keys():
            for rank in range(num_gpus):
                dict_losses[key][rank] = np.concatenate(dict_losses[key][rank], axis=0)
            
        formatting_time+=time.time()
        loss_computing_time-=time.time()
        results = pool.starmap(fuse_losses,[(dict_losses[key],device_list[i%len(device_list)],key) for i,key in enumerate(dict_losses.keys())])

        for key,losses in results:
            dict_losses[key] = losses
            print(losses.min(),losses.max(),losses.mean())
        for key in dict_losses.keys():
            dict_losses[key] = dict_losses[key]/dict_losses['original']
        loss_computing_time+=time.time()

        
    for key in dict_losses.keys():
        print(f"ON NODE {id_current_node}, FOR NUMBER OF COMPONENTS = {key} : \n Minimum loss : {dict_losses[key].min()}, Maximum loss: {dict_losses[key].max()}, Average loss: {dict_losses[key].mean()}\n\n",flush=True)
        
    print(f"Loading time: {loading_time}(s) \n Formatting time: {formatting_time} (s) \n Loss computing time: {loss_computing_time} (s)",flush=True)

    with open(f"losses_node_{id_current_node}.pickle", 'wb') as f:
        pickle.dump(dict_losses, f)
    
if __name__ == "__main__":
    main()

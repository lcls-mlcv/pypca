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
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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

def reconstruct_and_compute_metrics(model, projections_filename, device_list, idx_list, max_compo_list, rank, shm_list, shape, dtype):

    device=device_list[rank]
     # Access shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_list[rank].name)
    og_imgs = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    # Move original images to GPU
    og_imgs = torch.tensor(og_imgs, device=device)
    
    # Open the HDF5 files using h5py
    with h5py.File(model, 'r') as f_model, h5py.File(projections_filename, 'r') as f_proj:
        # Load the mean vector
        mu = torch.tensor(f_model['mu'][:], device=device)

        # Get dataset shapes for V and U
        V_shape = f_model['V'].shape

        # Sort the list of max components and ensure uniqueness
        max_compo_list = sorted(set(max_compo_list))
        num_images = len(idx_list)  # Number of images to reconstruct at once
        # Initialize the dictionary to hold metrics
        psnr_scores_dict, ssim_scores_dict = {max_compo: [] for max_compo in max_compo_list}, {max_compo: [] for max_compo in max_compo_list}
        
        # Iterate over each range of components in max_compo_list
        start_compo = 0
        rec_img_current = torch.zeros((num_images, V_shape[0], V_shape[1]), device=device)
        for max_compo in max_compo_list:
            for compo in range(start_compo, max_compo, 50):
                end = min(compo + 50, max_compo)

                # Process all indices at the same time for the current range of components
                for r in range(V_shape[0]):
                    # Load chunks of U and V from the HDF5 file in parallel across all idx_list
                    U_chunk = torch.tensor(f_proj['projected_images'][r, idx_list, compo:end], device=device)
                    V_chunk = torch.tensor(f_model['V'][r, :, compo:end], device=device)
                    # Update the reconstructed image for this component range
                    rec_img_current[:,r,:] += torch.matmul(U_chunk, V_chunk.T)

                    # Clean GPU memory for U_chunk and V_chunk
                    del U_chunk, V_chunk
                    torch.cuda.empty_cache()

            # Add the mean vector and save the current reconstruction state
            rec_img_dummy = torch.stack([rec_img_current[:, r, :] + mu[r] for r in range(V_shape[0])], dim=1)
            start_compo = max_compo
            
            torch.cuda.empty_cache()
            if rank == 0:
                print(f"Reconstructed images with {max_compo} number of components", flush=True)

            rec_img_dummy = rec_img_dummy.reshape(og_imgs.shape)

            psnr_scores, ssim_scores = [], []
            for img_orig, img_comp in zip(og_imgs, rec_img_dummy):
                # Ensure images are in range [0, 1] if needed
                data_range = img_orig.max() - img_orig.min()
    
                # PSNR computation
                mse = torch.mean((img_orig - img_comp) ** 2)
                psnr_val = 20 * torch.log10(data_range / torch.sqrt(mse)).item()
                psnr_scores.append(psnr_val)
    
                # SSIM computation (use skimage for SSIM as PyTorch lacks built-in support)
                img_orig_cpu = img_orig.detach().cpu().numpy()
                img_comp_cpu = img_comp.detach().cpu().numpy()
                ssim_val = ssim(img_orig_cpu, img_comp_cpu, multichannel=True, data_range=data_range.item())
                ssim_scores.append(ssim_val)

            # Store scores in dictionaries
            psnr_scores_dict[max_compo] = psnr_scores
            ssim_scores_dict[max_compo] = ssim_scores
            
            if rank == 0:
                print(f"Computed metric with {max_compo} number of components", flush=True)
                
    return psnr_scores_dict, ssim_scores_dict

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
    parser.add_argument("--projections_filename", help="Path to the projections file",required=True, type=str)
    return parser.parse_args()

def main():
    # Parse input parameters
    params = parse_input()
    exp, init_run, det_type = params.exp, params.run, params.det_type
    start_offset = params.start_offset if params.start_offset is not None else 0
    filename = params.model
    num_gpus, num_runs = params.num_gpus, params.num_runs
    num_images = json.loads(params.num_images)
    num_images_to_add = sum(num_images)
    loading_batch_size = params.loading_batch_size
    projections_filename = params.projections_filename

    compo_list=[1,2,3,5,10,50,100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000]
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    all_psnr_scores,all_ssim_scores = {},{}
    for num_compo in compo_list:
        all_psnr_scores[num_compo]=[]
        all_ssim_scores[num_compo]=[]
    last_batch = False
    
    loading_time=0
    formatting_time=0
    loss_computing_time=0
    reconstructing_time=0
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
                current_loading_batch = np.split(current_loading_batch, num_gpus, axis=0)
                shape, dtype = current_loading_batch[0].shape, current_loading_batch[0].dtype
            
                # Create shared memory for batches
                shm_list = create_shared_images(current_loading_batch)
                print("Images split and on shared memory",flush=True)

                current_loading_batch = []
                gc.collect()
                
                device_list = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(num_gpus)]
                formatting_time+=time.time()
                reconstructing_time-=time.time()
                # Reconstruct images and compute metrics
                results = pool.starmap(reconstruct_and_compute_metrics,[(filename, projections_filename, device_list, range(event+int(loading_batch_size/num_gpus)*rank,event+int(loading_batch_size/num_gpus)*(rank+1)),compo_list,rank, shm_list, shape, dtype) for rank in range(num_gpus)])
                formatting_time-=time.time()
                for rank in range(num_gpus):
                    for num_compo in compo_list:
                        all_psnr_scores[num_compo].append(results[rank][0][num_compo])
                        all_ssim_scores[num_compo].append(results[rank][1][num_compo])
                formatting_time+=time.time()
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
        for num_compo in compo_list:
            all_psnr_scores[num_compo]=np.concatenate(all_psnr_scores[num_compo],axis=0)
            all_ssim_scores[num_compo]=np.concatenate(all_ssim_scores[num_compo],axis=0)
        formatting_time+=time.time()
        print(f" Loading time: {loading_time}(s) \n Formatting time: {formatting_time} (s) \n Reconstructing time: {reconstructing_time} (s) \n Loss computing time: {loss_computing_time} (s)",flush=True)
        with open(f"psnr.pickle", 'wb') as f:
            pickle.dump(all_psnr_scores, f)
        with open(f"ssim.pickle", 'wb') as f:
            pickle.dump(all_ssim_scores, f) 
        
if __name__ == "__main__":
    main()

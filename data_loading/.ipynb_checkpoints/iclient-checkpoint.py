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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from processing.fit import remove_file_with_timeout
from processing.fit import iPCA_Pytorch_without_Psana

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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(self.server_address)
            request_data = json.dumps({
                'exp': exp, 'run': run, 'access_mode': access_mode,
                'detector_name': detector_name, 'event': event, 'mode': 'calib',
            })
            sock.sendall(request_data.encode('utf-8'))

            response_data = sock.recv(4096).decode('utf-8')
            response_json = json.loads(response_data)

            shm_name = response_json['name']
            shape = response_json['shape']
            dtype = np.dtype(response_json['dtype'])

            shm = None
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                data_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                result = np.array(data_array)
            finally:
                if shm:
                    shm.close()
                    shm.unlink()

            sock.sendall("ACK".encode('utf-8'))
            return result

def parse_input():
    """Parse command line input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", help="Experiment name.", required=True, type=str)
    parser.add_argument("-r", "--run", help="Run number.", required=True, type=int)
    parser.add_argument("-d", "--det_type", help="Detector name, e.g epix10k2M or jungfrau4M.", required=True, type=str)
    parser.add_argument("--start_offset", help="Run index of first image to be incorporated into iPCA model.", required=False, type=int)
    parser.add_argument("--num_images", help="Total number of images per run to be incorporated into model.", required=True, type=str)
    parser.add_argument("--loading_batch_size", help="Size of the batches used when loading the images on the client.", required=True, type=int)
    parser.add_argument("--num_components", help="Number of principal components to retain.", required=True, type=int)
    parser.add_argument("--batch_size", help="Batch size for iPCA algorithm.", required=True, type=int)
    parser.add_argument("--path", help="Path to the output directory.", required=False, type=str)
    parser.add_argument("--tag", help="Tag to append to the output file name.", required=False, type=str)
    parser.add_argument("--training_percentage", help="Percentage of the data to be used for training.", required=True, type=float)
    parser.add_argument("--num_gpus", help="Number of GPUs to use.", required=True, type=int)
    parser.add_argument("--num_runs", help="Number of runs to process.", required=True, type=int)
    parser.add_argument("--num_nodes", help="Number of nodes to use.", required=False, type=int)
    parser.add_argument("--id_current_node", help="ID of the current node.", required=False, type=int)

    return parser.parse_args()

def create_shared_images(images):
    shm_list = []
    for sub_imgs in images:
        chunk_size = np.prod(images[0].shape) * images[0].dtype.itemsize
        shm = shared_memory.SharedMemory(create=True, size=chunk_size)
        shm_images = np.ndarray(sub_imgs.shape, dtype=sub_imgs.dtype, buffer=shm.buf)
        np.copyto(shm_images, sub_imgs)
        shm_list.append(shm)
    return shm_list

def create_or_update_dataset(f, name, data):
    if name in f:
        del f[name]
    f.create_dataset(name, data=data)

def run_batch_process(algo_state_dict, ipca_state_dict, last_batch, rank, device, shape, dtype, shm_list, ipca_instance):
    algo_state_dict[rank] = {k: v.cuda(device) if torch.is_tensor(v) else v for k, v in algo_state_dict[rank].items()}
    return ipca_instance.run_batch(algo_state_dict, ipca_state_dict, last_batch, rank, device, shape, dtype, shm_list)

def compute_loss_process(rank, device_list, shape, dtype, shm_list, model_state_dict, batch_size, ipca_instance, loss_or_not):
    model_state_dict[rank] = {k:v for k, v in model_state_dict[rank].items()}
    model_state_dict[rank]['V'] = torch.tensor(model_state_dict[rank]['V'], device=device_list[rank])
    model_state_dict[rank]['mu'] = torch.tensor(model_state_dict[rank]['mu'], device=device_list[rank])
    return ipca_instance.compute_loss(rank, device_list, shape, dtype, shm_list, model_state_dict, batch_size, loss_or_not)

if __name__ == "__main__":
    start_time = time.time()
    params = parse_input()
    
    # Initialize parameters
    exp, init_run, det_type = params.exp, params.run, params.det_type
    start_offset = params.start_offset if params.start_offset is not None else 0
    num_components, batch_size = params.num_components, params.batch_size
    path, tag = params.path, params.tag
    training_percentage = params.training_percentage
    num_gpus = params.num_gpus
    num_runs = params.num_runs
    num_images = json.loads(params.num_images)
    num_tot_images = sum(num_images)
    num_training_images = int(num_tot_images * training_percentage)
    loading_batch_size = params.loading_batch_size
    num_nodes, id_current_node = params.num_nodes, params.id_current_node
    num_tot_gpus = num_nodes * num_gpus

    # Setup logging and multiprocessing
    logging.basicConfig(level=logging.INFO)
    mp.set_start_method('spawn', force=True)

    # Initialize iPCA instance
    filename_with_tag = f"{path}pypca_model_{tag}.h5"
    remove_file_with_timeout(filename_with_tag, True, timeout=10)
    ipca_instance = iPCA_Pytorch_without_Psana(
        exp=exp, run=init_run, det_type=det_type, num_images=num_tot_images,
        num_components=num_components, batch_size=batch_size, filename=filename_with_tag,
        training_percentage=training_percentage, num_gpus=num_gpus
    )

    # Initialize timers and state dictionaries
    l_time, t_time, f_time = 0, 0, 0
    algo_state_dict_local = ipca_instance.save_state()
    
    with mp.Manager() as manager:
        algo_state_dict = [manager.dict() for _ in range(num_gpus)]
        ipca_state_dict = [manager.dict() for _ in range(num_gpus)]
        model_state_dict = [manager.dict() for _ in range(num_gpus)]
        
        # Initialize algo_state_dict for each GPU
        for key, value in algo_state_dict_local.items():
            for rank in range(num_gpus):
                algo_state_dict[rank][key] = value.cpu().clone() if torch.is_tensor(value) else value

        with Pool(processes=num_gpus) as pool:
            fitting_start_time = time.time()
            num_images_seen = 0
            
            for run in range(init_run, init_run + num_runs):
                for event in range(start_offset, start_offset + num_images[run-init_run], loading_batch_size):
                    # Load and process images
                    beginning_time = time.time()
                    last_batch = num_images_seen + loading_batch_size >= num_training_images

                    # Prepare dataset and dataloader
                    requests_list = [(exp, run, 'idx', det_type, img) for img in range(event, min(event+loading_batch_size, num_images[run-init_run]))]
                    server_address = ('localhost', 5000)
                    dataset = IPCRemotePsanaDataset(server_address=server_address, requests_list=requests_list)
                    dataloader = DataLoader(dataset, batch_size=20, num_workers=4, prefetch_factor=None)
                    
                    # Load images
                    current_loading_batch = []
                    for batch in dataloader:
                        current_loading_batch.append(batch)
                        if num_images_seen + len(current_loading_batch) >= num_training_images:
                            last_batch = True
                            break

                    intermediate_time = time.time()
                    l_time += intermediate_time - beginning_time

                    # Process loaded images
                    current_loading_batch = np.concatenate(current_loading_batch, axis=0)
                    current_len = current_loading_batch.shape[0]
                    num_images_seen += current_len
                    logging.info(f"Loaded {event+current_len} images from run {run}.")
                    
                    # Remove NaN images
                    current_loading_batch = current_loading_batch[[i for i in range(current_len) if not np.isnan(current_loading_batch[i : i + 1]).any()]]
                    logging.info(f"Number of non-NaN images in the current batch: {current_loading_batch.shape[0]}")

                    # Split images for GPUs
                    current_loading_batch = np.split(current_loading_batch, num_tot_gpus, axis=1)
                    current_loading_batch = current_loading_batch[id_current_node*num_gpus:(id_current_node+1)*num_gpus]

                    shape = current_loading_batch[0].shape
                    dtype = current_loading_batch[0].dtype

                    # Create shared memory for batches
                    shm_list = create_shared_images(current_loading_batch)
                    device_list = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(num_gpus)]

                    intermediate_time2 = time.time()
                    t_time += intermediate_time2 - intermediate_time

                    # Run batch process
                    results = pool.starmap(run_batch_process, [
                        (algo_state_dict, ipca_state_dict, last_batch, rank, device_list, shape, dtype, shm_list, ipca_instance) 
                        for rank in range(num_gpus)
                    ])

                    if last_batch:
                        # Process final results
                        S, V, mu, total_variance = zip(*[(result['S'], result['V'], result['mu'], result['total_variance']) for result in results])
                        for rank in range(num_gpus):
                            model_state_dict[rank].update({'S': S[rank], 'V': V[rank], 'mu': mu[rank], 'total_variance': total_variance[rank]})
                        break

                    final_time = time.time()
                    f_time += final_time - intermediate_time2

                    # Memory management
                    torch.cuda.empty_cache()
                    gc.collect()

                if last_batch:
                    break

            fitting_end_time = time.time()
            logging.info(f"Time elapsed for fitting: {fitting_end_time - fitting_start_time} seconds.")
            logging.info(f"Loading time: {l_time}")
            logging.info(f"Treating time: {t_time}")
            logging.info(f"Gathering+Fitting time: {f_time}")

        # Save results
        saving_start_time = time.time()
        S, V, mu, total_variance = zip(*[(model_state_dict[rank]['S'], model_state_dict[rank]['V'], 
                                          model_state_dict[rank]['mu'], model_state_dict[rank]['total_variance']) 
                                         for rank in range(num_gpus)])

        with h5py.File(filename_with_tag, 'w') as f:
            for key, value in [('exp', exp), ('det_type', det_type), ('start_offset', start_offset),
                               ('run', run), ('num_runs', num_runs), ('S', S), ('V', V), ('mu', mu),
                               ('total_variance', total_variance), ('num_images', num_images)]:
                create_or_update_dataset(f, key, value)

        saving_end_time = time.time()
        logging.info(f'Model saved to {filename_with_tag} in {saving_end_time-saving_start_time} seconds')

    # Shut down server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(('localhost', 5000))
        sock.sendall("DONE".encode('utf-8'))

    logging.info('Server is shut down!')
    logging.info('Pipca is done!')

    end_time = time.time()
    logging.info(f"Total time elapsed: {end_time - start_time} seconds.")

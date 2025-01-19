import os
import csv
import h5py
import argparse
import math
import logging
import time
import psutil
import gc
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing import Pool, shared_memory
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.linalg import qr
import statistics
from processing.pca_on_gpu.pca_module import IncrementalPCAonGPU

class iPCA_Pytorch_without_Psana:

    """Incremental Principal Component Analysis, uses PyTorch. Can run on GPUs."""

    def __init__(
        self,
        exp,
        run,
        det_type,
        start_offset=0,
        num_images=10,
        num_components=10,
        batch_size=10,
        output_dir="",
        filename='pipca.model_h5',
        images=np.array([]),
        training_percentage=1.0,
        num_gpus=4
    ):

        self.start_offset = start_offset
        self.images = images
        self.output_dir = output_dir
        self.filename = filename
        self.shm = []
        
        self.num_images = num_images
        self.num_components = num_components
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        self.run = run
        self.exp = exp
        self.det_type = det_type
        self.start_time = None
        self.device_list = []

        self.training_percentage = training_percentage
        self.num_training_images = math.ceil(self.num_images * self.training_percentage)
        if self.num_training_images <= self.num_components:
            self.num_training_images = self.num_components

    def create_shared_images(self):
        """Create shared memory blocks for the images to be shared across the GPUs."""
        for sub_imgs in self.images:
            chunk_size = np.prod(self.images[0].shape) * self.images[0].dtype.itemsize
            shm = shared_memory.SharedMemory(create=True, size=chunk_size)
            shm_images = np.ndarray(sub_imgs.shape, dtype=sub_imgs.dtype, buffer=shm.buf)
            np.copyto(shm_images, sub_imgs)
            self.shm.append(shm)
            sub_imgs = None  # Delete the original images to free up memory

        self.images = None  # Delete the original images to free up memory

    def save_state(self):
        return self.__dict__.copy()

    def update_state(self,state_updates,device_list=None, shm_list = None):
        if state_updates is not None:
            self.__dict__.update(state_updates)
        if device_list is not None:
            self.device_list = device_list
        if shm_list is not None:
            self.shm = shm_list       

    def run_batch(self,algo_state_dict,ipca_state_dict,last_batch,rank,device_list,images_shape,images_dtype,shm_list):
        """Run the iPCA algorithm on the given data. The data is split across the available GPUs and the algorithm is run in parallel on each GPU. Does not need to be run on the entire dataset.
        Note that algo_state_dict and ipca_state_dict are shared across all GPUs and are updated in-place."""
        device = device_list[rank]
        algo_state_dict = algo_state_dict[rank]
        ipca_state_dict = ipca_state_dict[rank]
        for key, value in algo_state_dict.items():
            if torch.is_tensor(value):
                algo_state_dict[key] = value.to(device)
            else:
                algo_state_dict[key] = value

        self.device = device
        self.update_state(state_updates=algo_state_dict,device_list=device_list,shm_list = shm_list)

        ipca = IncrementalPCAonGPU(n_components = self.num_components, batch_size = self.batch_size, device = device, ipca_state_dict = ipca_state_dict)

        existing_shm = shared_memory.SharedMemory(name=self.shm[rank].name)
        images = np.ndarray(images_shape, dtype=images_dtype, buffer=existing_shm.buf)
        self.images = images

        self.num_images = self.images.shape[0]

        ipca.fit(self.images.reshape(self.num_images, -1)) ##va falloir faire gaffe au training ratio

        if not last_batch:
            existing_shm.close()
            existing_shm.unlink()
            self.shm = None
            self.images = None
            current_ipca_state_dict = ipca.save_ipca()
            current_algo_state_dict = self.save_state()
            for key, value in current_algo_state_dict.items():
                algo_state_dict[key] = value.cpu().clone() if torch.is_tensor(value) else value
            for key, value in current_ipca_state_dict.items():
                ipca_state_dict[key] = value.cpu().clone() if torch.is_tensor(value) else value
            torch.cuda.empty_cache()
            gc.collect()
            return None
    
        existing_shm.close()
        existing_shm.unlink()
        current_ipca_state_dict = ipca.save_ipca()
        current_algo_state_dict = self.save_state()
        for key, value in current_algo_state_dict.items():
            algo_state_dict[key] = value.cpu().clone() if torch.is_tensor(value) else value
        for key, value in current_ipca_state_dict.items():
            ipca_state_dict[key] = value.cpu().clone() if torch.is_tensor(value) else value

        if str(torch.device("cuda" if torch.cuda.is_available() else "cpu")).strip() == "cuda":
            S = ipca.singular_values_.cpu().detach().numpy()
            V = ipca.components_.cpu().detach().numpy().T
            mu = ipca.mean_.cpu().detach().numpy()
            total_variance = ipca.explained_variance_.cpu().detach().numpy()
        else:
            S = ipca.singular_values_
            V = ipca.components_.T
            mu = ipca.mean_
            total_variance = ipca.explained_variance_
        torch.cuda.empty_cache()
        gc.collect()
        dict_to_return = {'S':S, 'V':V, 'mu':mu, 'total_variance':total_variance}
        return dict_to_return

    def compute_loss(self,rank,device_list,images_shape,images_dtype,shm_list,model_state_dict,batch_size,loss_or_not):
        device = device_list[rank]
        model_state_dict = model_state_dict[rank]
        existing_shm = shared_memory.SharedMemory(name=shm_list[rank].name)
        images = np.ndarray(images_shape, dtype=images_dtype, buffer=existing_shm.buf)
        
        V = model_state_dict['V']
        mu = model_state_dict['mu']
        transformed_images = []

        average_losses = []
        ##
        list_norm_diff = torch.tensor([], device=device)
        list_init_norm = torch.tensor([], device=device)
        ##

        for start in range(0, images.shape[0], batch_size):
            end = min(start + batch_size, images.shape[0])
            batch_imgs = images[start:end]
            batch_imgs = torch.tensor(batch_imgs.reshape(end-start,-1), device=device)
            initial_norm = torch.norm(batch_imgs, dim=1, p = 'fro')
            transformed_batch = torch.mm((batch_imgs.clone() - mu),V)
            transformed_images.append(transformed_batch)
            if loss_or_not:
                reconstructed_batch = torch.mm(transformed_batch,V.T) + mu
                diff = batch_imgs - reconstructed_batch
                norm_batch = torch.norm(diff, dim=1, p = 'fro')
                ##
                list_norm_diff = torch.cat((list_norm_diff,norm_batch),dim=0)
                list_init_norm = torch.cat((list_init_norm,initial_norm),dim=0)
                ##
                norm_batch = norm_batch/initial_norm
                average_losses.append(torch.mean(norm_batch).cpu().detach().numpy())
        
        average_loss = np.mean(average_losses) if loss_or_not else None
        transformed_images = torch.cat(transformed_images, dim=0).cpu().detach().numpy()
        list_norm_diff = list_norm_diff.cpu().detach().numpy()
        list_init_norm = list_init_norm.cpu().detach().numpy()
        existing_shm.close()
        existing_shm.unlink()
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Memory Allocated on GPU {rank}: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB",flush=True)
        print(f"Memory Cached on GPU {rank}: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB",flush=True)
        print(f"Memory free on GPU {rank}: {torch.cuda.mem_get_info(device)[0] / 1024**3:.2f} GB",flush=True)

        return average_loss,average_losses,transformed_images,list_norm_diff,list_init_norm

    

###############################################################################################################
###############################################################################################################

def append_to_dataset(f, dataset_name, data):
    if dataset_name not in f:
        f.create_dataset(dataset_name, data=np.array(data))
    else:
        if isinstance(f[dataset_name], h5py.Dataset) and f[dataset_name].shape == ():
            # Scalar dataset, convert to array
            existing_data = np.atleast_1d(f[dataset_name][()])
        else:
            # Non-scalar dataset, use slicing
            existing_data = f[dataset_name][:]

        new_data = np.atleast_1d(np.array(data))
        data_combined = np.concatenate([existing_data, new_data])
        del f[dataset_name]
        f.create_dataset(dataset_name, data=data_combined)

def create_or_update_dataset(f, name, data):
    if name in f:
        del f[name]
    f.create_dataset(name, data=data)

def remove_file_with_timeout(filename_with_tag, overwrite=True, timeout=10):
    """
    Remove the file specified by filename_with_tag if it exists.
    
    Parameters:
        filename_with_tag (str): The name of the file to remove.
        overwrite (bool): Whether to attempt removal if the file exists (default is True).
        timeout (int): Maximum time allowed for attempting removal (default is 10 seconds).
    """
    start_time = time.time()  # Record the start time

    while overwrite and os.path.exists(filename_with_tag):
        # Check if the loop has been running for more than the timeout period
        if time.time() - start_time > timeout:
            break  # Exit the loop
            
        try:
            os.remove(filename_with_tag)
        except FileNotFoundError:
            break  # Exit the loop

def mapping_function(images, type_mapping = "id"):
    """
    Map the images to a different type.
    
    Parameters:
        images (np.array or torch.tensor): The images to map.
        type (str): The type to map to (default is "id").
        
    Returns:
        np.array or torch.tensor : The mapped images.
    """
    if isinstance(images, np.ndarray):
        if type_mapping == "id":
            return images
        elif type_mapping == "sqrt":
            return np.sign(images) * np.sqrt(np.abs(images))
        elif type_mapping == "log":
            return np.sign(images) * np.log(np.abs(images)+10**(-6))
        else:
            return images
    elif isinstance(images, torch.Tensor):
        if type_mapping == "id":
            return images
        elif type_mapping == "sqrt":
            return torch.sign(images) * torch.sqrt(torch.abs(images))
        elif type_mapping == "log":
            return torch.sign(images) * torch.log(torch.abs(images)+10**(-6))
        else:
            return images
    else:
        raise ValueError("The input type is not supported")

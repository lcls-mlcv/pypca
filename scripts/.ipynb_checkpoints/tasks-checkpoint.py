import logging
import os
import requests
import glob
import shutil
import numpy as np
import itertools
import h5py
import time
import yaml
import csv
from mpi4py import MPI
import subprocess
import socket
import json
from misc.ischeduler import JobScheduler
from misc.get_max_events import main as compute_max_events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch the URL to post progress update
update_url = os.environ.get('JID_UPDATE_COUNTERS')

def t_sne(config):
    setup = config.setup
    task = config.t_sne
    num_images = task.num_images
    num_gpus = task.num_gpus
    filename = task.filename
    num_tries = task.num_tries
    threshold = task.threshold
    num_runs = 0
    distribution_images = []
    exp = setup.exp
    run = task.run
    grid_size = task.grid_size
    det_type = setup.det_type
    copy_num_images = num_images
    guiding_panel = task.guiding_panel

    while num_images > 0:
        max_event = compute_max_events(exp, run+num_runs, det_type)
        images_for_run = min(max_event, num_images)
        distribution_images.append(images_for_run)
        num_images -= images_for_run
        num_runs += 1

    print(f"Number of runs: {num_runs}")
    num_images_str = json.dumps(distribution_images)
    print(f"Number of images: {num_images_str}")

    if task.get('loading_batch_size') is not None:
        loading_batch_size = task.loading_batch_size
    else:
        loading_batch_size = 2000

    server_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../pypca/data_loading/iserver.py")
    client_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../pypca/data_loading/t_snes.py")

    command = "which python; ulimit -n 4096;"
    command += f"python {server_path} & echo 'Server is running'"
    command += f"; echo 'Number of images: {num_images}'"
    command += "; sleep 10"
    command += ";conda deactivate; echo 'Server environment deactivated'"
    command += "; conda activate /sdf/group/lcls/ds/tools/conda_envs/py3.11-nopsana-torch-rapids; which python; echo 'Client environment activated'"
    command += f"; python {client_path} --filename {filename} --num_images {num_images_str} --loading_batch_size {loading_batch_size} --num_tries {num_tries} --threshold {threshold} --num_runs {num_runs} --grid_size {grid_size} --guiding_panel {guiding_panel}"

    js = JobScheduler(os.path.join(".", f't_snes_{copy_num_images}.sh'),queue = 'ampere', ncores=  1, jobname=f't_snes_{copy_num_images}',logdir='/sdf/home/n/nathfrn/pypca/scripts/logs',account='lcls',mem = '200G',num_gpus = num_gpus) ##
    js.write_header()
    js.write_main(f"{command}\n", dependencies=['psana'],find_python_path=False)
    js.clean_up()
    js.submit()
    print('All done!')

def create_pypca(config, num_nodes = 1, id_current_node = 0):
    setup = config.setup
    task = config.create_pypca_multinodes
    exp = setup.exp
    run = task.run
    det_type = setup.det_type
    start_offset = task.start_offset
    num_images = task.num_images
    num_tot_images = num_images
    distribution_images = [] 

    ## Computes number of runs and number of images per run
    num_runs = 0
    while num_images > 0:
        max_event = compute_max_events(exp, run+num_runs, det_type)
        images_for_run = min(max_event, num_images)
        distribution_images.append(images_for_run)
        num_images -= images_for_run
        num_runs += 1
    ##

    print(f"Number of runs: {num_runs}")
    num_images_str = json.dumps(distribution_images)
    num_components = task.num_components
    batch_size = task.batch_size
    path = task.path
    tag = task.tag

    if num_nodes > 1:
        tag = f"{tag}_node_{id_current_node}"

    num_gpus = task.num_gpus
    training_percentage = task.training_percentage
    smoothing_function = task.smoothing_function
    compute_loss = task.compute_loss
    comm = MPI.COMM_WORLD
    ncores = comm.Get_size()
    compute_projected_images = task.compute_projected_images

    if task.get('loading_batch_size') is not None:
        loading_batch_size = task.loading_batch_size
    else:
        loading_batch_size = 2000

    server_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../data_loading/iserver.py")
    client_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../data_loading/iclient.py")

    command = "which python; ulimit -n 4096;"
    command += f"python {server_path} & echo 'Server is running'"
    command += f"; echo 'Number of images: {num_tot_images}'; echo 'Number of events to collect per run: {num_images_str}'"
    command += "; sleep 10"
    command += ";conda deactivate; echo 'Server environment deactivated'"
    command += "; conda activate /sdf/group/lcls/ds/tools/conda_envs/py3.11-nopsana-torch-rapids; which python; echo 'Client environment activated'"
    command += f"; python {client_path} -e {exp} -r {run} -d {det_type} --start_offset {start_offset} --num_images '{num_images_str}' --loading_batch_size {loading_batch_size} --num_components {num_components} --batch_size {batch_size} --path {path} --tag {tag} --training_percentage {training_percentage} --smoothing_function {smoothing_function} --num_gpus {num_gpus} --compute_loss {compute_loss} --num_runs {num_runs} --compute_projected_images {compute_projected_images} --num_nodes {num_nodes} --id_current_node {id_current_node}"

    js = JobScheduler(os.path.join(".", f'create_pypca_{num_components}_{num_tot_images}_{batch_size}_node_{id_current_node}.sh'),queue = 'ampere', ncores=  1, jobname=f'create_pypca_{num_components}_{num_tot_images}_{batch_size}_node_{id_current_node}',logdir='/sdf/home/n/nathfrn/pypca/scripts/logs',account='lcls',mem = '200G',num_gpus = num_gpus)
    js.write_header()
    js.write_main(f"{command}\n", dependencies=['psana'],find_python_path=False)
    js.clean_up()
    js.submit()
    print('All done!')

def create_pypca_multinodes(config):
    num_nodes = config.create_pypca_multinodes.num_nodes
    if num_nodes ==1:
        create_pypca(config)
    else:
        import multiprocessing
        from misc.clean_pypca import clean_pypca
        algo_start_time = time.time()
        with multiprocessing.Pool(processes=num_nodes) as pool:
            args = [(config, num_nodes, node) for node in range(num_nodes)]
            pool.starmap(create_pypca, args)
        algo_end_time = time.time()
        print(f"Algorithm time: {algo_end_time - algo_start_time}")
        
        clean_pypca(config.create_pypca_multinodes.path, config.create_pypca_multinodes.tag, num_nodes)

        print('All nodes done!')

def update_pypca(config,num_nodes = 1, id_current_node = 0):

    setup = config.setup
    task = config.update_pypca_multinodes
    exp = setup.exp
    run = task.run
    det_type = setup.det_type
    start_offset = task.start_offset
    num_images = task.num_images
    num_tot_images = num_images
    lower_bound = task.lower_bound
    upper_bound = task.upper_bound

    distribution_images = [] 
    ##
    num_runs = 0
    while num_images > 0:
        max_event = compute_max_events(exp, run+num_runs, det_type)
        images_for_run = min(max_event, num_images)
        distribution_images.append(images_for_run) #-1 enlevé là
        num_images -= images_for_run
        num_runs += 1
    ##
    print(f"Number of runs: {num_runs}")
    num_images_str = json.dumps(distribution_images)
    batch_size = task.batch_size
    num_gpus = task.num_gpus
    model = task.model

    comm = MPI.COMM_WORLD
    ncores = comm.Get_size()

    if task.get('loading_batch_size') is not None:
        loading_batch_size = task.loading_batch_size
    else:
        loading_batch_size = 2000

    server_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../data_loading/iserver.py")
    client_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../data_loading/update_pipca.py")

    command = "which python; ulimit -n 4096;"
    command += f"python {server_path} & echo 'Server is running'"
    command += f"; echo 'Number of images: {num_tot_images}'; echo 'Number of events to collect per run: {num_images_str}'"
    command += "; sleep 10"
    command += ";conda deactivate; echo 'Server environment deactivated'"
    command += "; conda activate /sdf/group/lcls/ds/tools/conda_envs/py3.11-nopsana-torch-rapids; which python; echo 'Client environment activated'"
    command += f"; python {client_path} -e {exp} -r {run} -d {det_type} --start_offset {start_offset} --num_images '{num_images_str}' --loading_batch_size {loading_batch_size} --batch_size {batch_size} --num_runs {num_runs} --lower_bound {lower_bound} --upper_bound {upper_bound} --model {model} --num_gpus {num_gpus} --num_nodes {num_nodes} --id_current_node {id_current_node}"

    js = JobScheduler(os.path.join(".", f'update_pypca_{num_tot_images}_{batch_size}_node_{id_current_node}.sh'),queue = 'ampere', ncores=  1, jobname=f'update_pypca_{num_tot_images}_{batch_size}_node_{id_current_node}',logdir='/sdf/home/n/nathfrn/pypca/scripts/logs',account='lcls',mem = '200G',num_gpus = num_gpus)
    js.write_header()
    js.write_main(f"{command}\n", dependencies=['psana'],find_python_path=False)
    js.clean_up()
    js.submit()
    print('All done!')

def update_pypca_multinodes(config):
    num_nodes = config.update_pypca_multinodes.num_nodes
    if num_nodes ==1:
        update_pypca(config)
    else:
        import multiprocessing
        from misc.clean_pypca import clean_pypca
        algo_start_time = time.time()
        with multiprocessing.Pool(processes=num_nodes) as pool:
            args = [(config, num_nodes, node) for node in range(num_nodes)]
            pool.starmap(update_pypca, args)
        algo_end_time = time.time()
        print(f"Algorithm time: {algo_end_time - algo_start_time}")
        
        model_path = os.path.dirname(config.reduce_pypca_multinodes.model)
        tag = config.reduce_pypca_multinodes.model.split('/')[-1]
        clean_pypca(model_path, tag, num_nodes,mode='update')

    print('All nodes done!')

def reduce_pypca(config,num_nodes = 1, id_current_node = 0):

    setup = config.setup
    task = config.reduce_pypca_multinodes
    exp = setup.exp
    run = task.run
    det_type = setup.det_type
    start_offset = task.start_offset
    num_images = task.num_images
    num_tot_images = num_images

    distribution_images = [] 
    ##
    num_runs = 0
    while num_images > 0:
        max_event = compute_max_events(exp, run+num_runs, det_type)
        images_for_run = min(max_event, num_images)
        distribution_images.append(images_for_run)
        num_images -= images_for_run
        num_runs += 1
    ##
    print(f"Number of runs: {num_runs}")
    num_images_str = json.dumps(distribution_images)
    batch_size = task.batch_size
    num_gpus = task.num_gpus
    model = task.model

    comm = MPI.COMM_WORLD
    ncores = comm.Get_size()

    if task.get('loading_batch_size') is not None:
        loading_batch_size = task.loading_batch_size
    else:
        loading_batch_size = 2000

    server_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../data_loading/iserver.py")
    client_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../data_loading/pypca_reducer.py")

    command = "which python; ulimit -n 8000;"
    command += f"python {server_path} & echo 'Server is running'"
    command += f"; echo 'Number of images: {num_tot_images}'; echo 'Number of events to collect per run: {num_images_str}'"
    command += "; sleep 10"
    command += ";conda deactivate; echo 'Server environment deactivated'"
    command += "; conda activate /sdf/group/lcls/ds/tools/conda_envs/py3.11-nopsana-torch-rapids; which python; echo 'Client environment activated'; conda list"
    command += f"; python {client_path} -e {exp} -r {run} -d {det_type} --start_offset {start_offset} --num_images '{num_images_str}' --loading_batch_size {loading_batch_size} --batch_size {batch_size} --num_runs {num_runs} --model {model} --num_gpus {num_gpus} --num_nodes {num_nodes} --id_current_node {id_current_node}"

    js = JobScheduler(os.path.join(".", f'reduce_pypca_{num_tot_images}_{batch_size}_node_{id_current_node}.sh'),queue = 'ampere', ncores=  1, jobname=f'reduce_pypca_{num_tot_images}_{batch_size}_node_{id_current_node}',logdir='/sdf/home/n/nathfrn/pypca/scripts/logs',account='lcls',mem = '200G',num_gpus = num_gpus)
    js.write_header()
    js.write_main(f"{command}\n", dependencies=['psana'],find_python_path=False)
    js.clean_up()
    js.submit()
    print('All done!')

def reduce_pypca_multinodes(config):
    num_nodes = config.reduce_pypca_multinodes.num_nodes
    if num_nodes ==1:
        reduce_pypca(config)
    else:
        import multiprocessing
        from misc.clean_pypca import clean_pypca
        algo_start_time = time.time()
        with multiprocessing.Pool(processes=num_nodes) as pool:
            args = [(config, num_nodes, node) for node in range(num_nodes)]
            pool.starmap(reduce_pypca, args)
        algo_end_time = time.time()
        print(f"Algorithm time: {algo_end_time - algo_start_time}")
        
        model_path = os.path.dirname(config.reduce_pypca_multinodes.model)
        tag = f"projected_images_{config.setup.exp}_start_run_{config.reduce_pypca_multinodes.run}_num_images_{config.reduce_pypca_multinodes.num_images}"
        clean_pypca(model_path, tag, num_nodes,mode='reduce')

    print('All nodes done!')

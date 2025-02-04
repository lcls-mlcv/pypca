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

def dim_reduc(config):
    setup = config.setup
    task = config.dim_reduc
    num_images = task.num_images
    num_gpus = task.num_gpus
    filename = task.filename
    num_tries = task.num_tries
    threshold = task.threshold
    num_runs = 0
    distribution_images = []
    exp = setup.exp
    run = task.start_run
    grid_size = task.grid_size
    det_type = setup.det_type
    guiding_panel = task.guiding_panel
    type_of_embedding = task.type_of_embedding
    log_dir = setup.log_dir
    first_rank = task.first_rank
    max_run = task.max_run
        
    while num_images > 0:
        max_event = compute_max_events(exp, run+num_runs, det_type)
        images_for_run = min(max_event, num_images)
        distribution_images.append(images_for_run)
        num_images -= images_for_run
        num_runs += 1

    if num_images == -1: #includes all images of the run
        num_runs = max_run - run + 1
        num_images=0
        for r in range(run,max_run+1):
            images_for_run=compute_max_events(exp,r,det_type)
            num_images+=images_for_run
            distribution_images.append(images_for_run)

    print(f"Number of runs: {num_runs}")
    num_images_str = json.dumps(distribution_images)
    print(f"Number of images: {num_images_str}")

    if task.get('loading_batch_size') is not None:
        loading_batch_size = task.loading_batch_size
    else:
        loading_batch_size = 2000

    server_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../data_loading/iserver.py")
    client_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../visuals/dim_reduc.py")

    command = "which python; ulimit -n 4096;"
    command += f"python {server_path} & echo 'Server is running'"
    command += f"; echo 'Number of images: {num_images}'"
    command += "; sleep 10"
    command += ";conda deactivate; echo 'Server environment deactivated'"
    command += "; conda activate /sdf/group/lcls/ds/tools/conda_envs/py3.11-nopsana-torch-rapids; which python; echo 'Client environment activated'"
    command += f"; python {client_path} --filename {filename} --num_images {num_images_str} --loading_batch_size {loading_batch_size} --num_tries {num_tries} --threshold {threshold} --num_runs {num_runs} --grid_size {grid_size} --guiding_panel {guiding_panel} --type_of_embedding {type_of_embedding} --first_rank {first_rank}"

    js = JobScheduler(os.path.join(".", f'dim_reduc_{task.num_images}.sh'),queue = 'ampere', ncores=  1, jobname=f'dim_reduc_{task.num_images}',logdir=log_dir,account=config.setup.account,mem = '200G',num_gpus = num_gpus) ## ACCOUNT PREEMPTABLE
    js.write_header()
    js.write_main(f"{command}\n", dependencies=['psana'],find_python_path=False)
    js.clean_up()
    js.submit()
    print('All done!')

def fit_node(config,num_nodes = 1, id_current_node = 0):

    setup = config.setup
    task = config.fit
    exp = setup.exp
    run = task.start_run
    det_type = setup.det_type
    start_offset = task.start_offset
    num_images = task.num_images
    num_tot_images = num_images
    lower_bound = task.lower_bound
    upper_bound = task.upper_bound
    log_dir = setup.log_dir
    distribution_images = []
    num_components = task.num_components
    max_run = task.max_run
    ##
    num_runs = 0
    while num_images > 0 and (max_run-run+1)!=num_runs:
        max_event = compute_max_events(exp, run+num_runs, det_type)
        images_for_run = min(max_event, num_images)
        distribution_images.append(images_for_run)
        num_images -= images_for_run
        num_runs += 1
    ##

    if num_images == -1: #includes all images of the run
        num_runs = max_run - run + 1
        num_images=0
        for r in range(run,max_run+1):
            images_for_run=compute_max_events(exp,r,det_type)
            num_images+=images_for_run
            distribution_images.append(images_for_run)
            
    print(f"Number of runs: {num_runs}")
    
    num_images_str = json.dumps(distribution_images)
    batch_size = task.batch_size
    num_gpus = task.num_gpus
    model_path = task.model_path

    comm = MPI.COMM_WORLD
    ncores = comm.Get_size()

    if task.get('loading_batch_size') is not None:
        loading_batch_size = task.loading_batch_size
    else:
        loading_batch_size = 2000

    server_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../data_loading/iserver.py")
    client_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../processing/fit.py")

    command = "which python; ulimit -n 8000;"
    command += f"python {server_path} & echo 'Server is running';"
    if id_current_node==0:
        command += " ../monitor.sh &"
    command += f" echo 'Number of images: {num_tot_images}'; echo 'Number of events to collect per run: {num_images_str}'; echo 'Total number of events to collect: {num_images}'"
    command += "; sleep 10"
    command += "; conda deactivate; echo 'Server environment deactivated'"
    command += "; conda activate /sdf/group/lcls/ds/tools/conda_envs/py3.11-nopsana-torch-rapids; which python; echo 'Client environment activated'"
    command += f"; python {client_path} -e {exp} -r {run} -d {det_type} --start_offset {start_offset} --num_images '{num_images_str}' --loading_batch_size {loading_batch_size} --batch_size {batch_size} --num_runs {num_runs} --lower_bound {lower_bound} --upper_bound {upper_bound} --model {model_path} --num_gpus {num_gpus} --num_nodes {num_nodes} --id_current_node {id_current_node} --num_components {num_components}"

    js = JobScheduler(os.path.join(".", f'fit_{run}_{max_run}_{num_tot_images}_{batch_size}_node_{id_current_node}.sh'),queue = 'ampere', ncores=  1, jobname=f'fit_{run}_{max_run}_{num_tot_images}_{batch_size}_node_{id_current_node}',logdir=log_dir,account=config.setup.account,mem = '200G',num_gpus = num_gpus)  ##ACCOUNT PREEMPTABLE
    js.write_header()
    js.write_main(f"{command}\n", dependencies=['psana'],find_python_path=False)
    js.clean_up()
    js.submit()
    print('All done!')

def fit(config):
    task = config.fit
    num_nodes = task.num_nodes
    from misc.merge_pypcas import merge_pypcas
    if num_nodes ==1:
        fit_node(config)
    else:
        import multiprocessing
        algo_start_time = time.time()
        with multiprocessing.Pool(processes=num_nodes) as pool:
            args = [(config, num_nodes, node) for node in range(num_nodes)]
            pool.starmap(fit_node, args)
        algo_end_time = time.time()
        print(f"Algorithm time: {algo_end_time - algo_start_time}")

    if not os.path.isdir(task.model_path):
        model_path = os.path.dirname(task.model_path)
        tag = task.model_path.split('/')[-1]
        overwrite=False
    else:
        model_path = task.model_path
        overwrite=True
        tag = f"{task.start_run}_{task.max_run}_{task.num_images}_{task.num_components}_{task.batch_size}"
    merge_pypcas(model_path, tag, num_nodes,mode='fit',overwrite=overwrite)

    print('All nodes done!')

def transform_node(config,num_nodes = 1, id_current_node = 0):
    setup = config.setup
    task = config.transform
    exp = setup.exp
    run = task.start_run
    det_type = setup.det_type
    start_offset = task.start_offset
    num_images = task.num_images
    num_tot_images = num_images
    log_dir = setup.log_dir
    distribution_images = [] 
    max_run = task.max_run
    ##
    num_runs = 0
    while num_images > 0:
        max_event = compute_max_events(exp, run+num_runs, det_type)
        images_for_run = min(max_event, num_images)
        distribution_images.append(images_for_run)
        num_images -= images_for_run
        num_runs += 1

    if num_images == -1: #includes all images of the run
        num_runs = max_run - run + 1
        num_images=0
        for r in range(run,max_run+1):
            images_for_run=compute_max_events(exp,r,det_type)
            num_images+=images_for_run
            distribution_images.append(images_for_run)
            
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
    client_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../processing/transform.py")

    command = "which python; ulimit -n 8000;"
    command += f"python {server_path} & echo 'Server is running';"
    if id_current_node==0:
        command += " ../monitor.sh &"
    command += f" echo 'Number of images: {num_tot_images}'; echo 'Number of events to collect per run: {num_images_str}'; echo 'Total number of events to collect: {num_images}'"
    command += "; sleep 10"
    command += "; conda deactivate; echo 'Server environment deactivated'"
    command += "; conda activate /sdf/group/lcls/ds/tools/conda_envs/py3.11-nopsana-torch-rapids; which python; echo 'Client environment activated'"
    command += f"; python {client_path} -e {exp} -r {run} -d {det_type} --start_offset {start_offset} --num_images '{num_images_str}' --loading_batch_size {loading_batch_size} --batch_size {batch_size} --num_runs {num_runs} --model {model} --num_gpus {num_gpus} --num_nodes {num_nodes} --id_current_node {id_current_node}"

    js = JobScheduler(os.path.join(".", f'transform_{run}_{max_run}_{num_tot_images}_{batch_size}_node_{id_current_node}.sh'),queue = 'ampere', ncores=  1, jobname=f'transform_{run}_{max_run}_{num_tot_images}_{batch_size}_node_{id_current_node}',logdir=log_dir,account=config.setup.account,mem = '200G',num_gpus = num_gpus) ##ACCOUNT PREEMPTABLE
    js.write_header()
    js.write_main(f"{command}\n", dependencies=['psana'],find_python_path=False)
    js.clean_up()
    js.submit()
    print('All done!')

def transform(config):
    from misc.merge_pypcas import merge_pypcas
    task = config.transform
    num_nodes = task.num_nodes
    time1=time.time()
    num_images = task.num_images
    if num_images == -1: #includes all images of the run
        num_runs = task.max_run - task.start_run + 1
        num_images=0
        for r in range(task.start_run,task.max_run+1):
            images_for_run=compute_max_events(config.setup.exp,r,config.setup.det_type)
            num_images+=images_for_run
    
    with h5py.File(task.model) as f:
        num_components = np.array(f['S']).shape[1]
    if num_nodes ==1:
        transform_node(config)
    else:
        import multiprocessing
        algo_start_time = time.time()
        with multiprocessing.Pool(processes=num_nodes) as pool:
            args = [(config, num_nodes, node) for node in range(num_nodes)]
            pool.starmap(transform_node, args)
        algo_end_time = time.time()
        print(f"Algorithm time: {algo_end_time - algo_start_time}")

    model_path = os.path.dirname(task.model)
    tag = f"projections_{task.start_run}_{task.max_run}_{num_images}_{num_components}"
    merge_pypcas(model_path, tag, num_nodes,mode='transform')

    print('All nodes done!')

def fit_transform(config):
    start_time = time.time()
    fit(config)
    int_time = time.time()
    print(f"Fitting done in {int_time-start_time} (s)",flush=True)
    transform(config)
    end_time = time.time()
    print(f"Transforming done in {end_time-int_time} (s)",flush=True)
    print(f"Total time : {end_time-start_time} (s)",flush=True)

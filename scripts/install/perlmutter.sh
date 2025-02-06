#!/bin/bash

echo "Installing PyPCA server environment (CCTBX/psana)"
sbatch perlmutter_server.slurm

echo "Installing PyPCA client environment (pytorch/rapids)"
sbatch perlmutter_client.slurm
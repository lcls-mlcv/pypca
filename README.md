### Pypca

## Set-up

# Set the bashrc

Change the path in the bashrc (adapt it to your home directory):

export PATH=/sdf/home/n/nathfrn/pypca/scripts:$PATH
export PYTHONPATH=/sdf/home/n/nathfrn/pypca

# Navigate the files

cd pypca/scripts/slurm_outputs

# Configurate the .yaml files

In pypca/scripts/yamls, create a .yaml file for each experiment. Adapt the current "mfxp23120.yaml".

## Launch a job

../elog_submit.sh -c ../yamls/exp_name.yaml -t job_name -a lcls:exp_name -q milano -n 1

Current jobs are : fit, transform, dim_reduc

Tested experiments : mfxp23120, mfxx49820


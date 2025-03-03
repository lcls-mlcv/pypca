
# PyPCA Documentation

PyPCA is an incremental and parallelized implementation of Principal Component Analysis (PCA), designed specifically for large-scale experimental data processing within the LCLS (Linac Coherent Light Source) ecosystem.

---

## Description

PyPCA enables dimensionality reduction on large-scale data using a distributed approach. The algorithm is optimized to run on computing clusters with GPU support, allowing for the processing of large-scale experimental data.

---

## Installation

### Environment Setup

1. Clone the repository and configure environment variables in your `.bashrc` file:

    ```bash
    # Adapt these paths to your home directory or wherever you cloned the repository
    export PATH=/path/to/your/folder/pypca/scripts:$PATH
    export PYTHONPATH=/path/to/your/folder/pypca
    ```

2. Reload your `.bashrc` or open a new terminal.

---

## Experiment Configuration

Each experiment requires a customized YAML configuration file.

1. Navigate to the YAML files folder:
    ```bash
    cd pypca/scripts/yamls
    ```

2. Create a YAML file for your experiment based on existing examples (like `mfxx49820.yaml`).

---

## Usage

### Launching a Job

To submit a job to the Slurm resource manager:

```bash
cd pypca/scripts/slurm_outputs
../elog_submit.sh -c ../yamls/exp_name.yaml -t job_name -a lcls:exp_name -q milano -n 1
```

Where:
- `exp_name.yaml`: YAML configuration file for your experiment
- `job_name`: type of task to execute (see below)
- `lcls:exp_name`: account and experiment name
- `milano`: queue name
- `n`: number of nodes

### Available Task Types

- **fit**: Trains a PCA model on the data
- **transform**: Applies a trained PCA model to project the data
- **dim_reduc**: Performs dimensionality reduction (e.g., UMAP) on transformed data
- **fit_transform**: Combines training and projection steps
- **compute_loss**: Calculates reconstruction loss/error
- **compute_psnr_ssim**: Calculates PSNR and SSIM metrics to evaluate reconstruction quality

### Tested Experiments

- `mfxp23120`
- `mfxx49820`
- `mfxl1008021`

---

## Configuration Parameters

### General Configuration

- **exp**: Experiment identifier
- **det_type**: Detector type used
- **root_dir**: Root directory of the PyPCA installation
- **log_dir**: Directory for log files
- **account**: Account used for resource allocation

### Common Parameters for Multiple Tasks

- **start_run** / **max_run**: Initial and final run numbers to process
- **num_images**: Number of images to process (-1 for all)
- **batch_size**: Processing batch size
- **loading_batch_size**: Data loading batch size
- **num_gpus**: Number of GPUs to use --> number of "panels" or "subimages" created.
- **num_nodes**: Number of compute nodes

### Training-Specific Parameters (fit)

- **num_components**: Number of principal components to extract
- **model_path**: Path to save the trained model
- **lower_bound** / **upper_bound**: Limits for data selection

### Transformation and Analysis Parameters

- **model**: Path to the trained model to use
- **projections_filename**: Output file for projections

### Dimensionality Reduction Parameters
 
- **grid_size**: Grid size for visualization
- **type_of_embedding**: Reduction algorithm (e.g., 'umap')
- **threshold**: Score threshold for dimensionality reduction algorithm
- **num_tries**: Max number of times the algorithm is run. Stops prematurely is threshold is reached, keeps the best score if not.
- **guiding_panel**: Which "panel" will be used for dimensionality reduction
- **first_rank**: Default 0 but can be increased in case the first PCA components do not want to be taken into account.

#### NB: File tagging

- **pypca_models**: the tagging will be the following : `pypca_model_{start_run}_{max_run}_{num_images}_{num_components}_{batch_size}.h5`. Replace the curly-braced placeholders (`{}`) with their corresponding values
- **projections**: the tagging will be the following : `projections_{start_run}_{max_run}_{num_images}_{num_components}.h5`. Replace the curly-braced placeholders (`{}`) with their corresponding values
  
---

## Examples

### Train a PCA model on the `mfxx49820` experiment

```bash
../elog_submit.sh -c ../yamls/mfxx49820.yaml -t fit -a lcls:mfxx49820 -q milano -n 8
```

### Transform data with an existing model

```bash
../elog_submit.sh -c ../yamls/mfxx49820.yaml -t transform -a lcls:mfxx49820 -q milano -n 8
```

import h5py
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import make_interp_spline
from scipy.ndimage import uniform_filter1d

def draw_plots(plot_components=None, smoothing_window_size=50,metric='psnr'):
    with open(f"{metric}.pickle", 'rb') as f:
            data = pickle.load(f)
        
    data_first_plot = {}
    if plot_components is None:
        plot_components=list(data.keys())
        
    for key in plot_components:
        if key in data:
            data_first_plot[key] = data[key]

    fig = plt.figure(figsize=(16, 12))

    # Determine min and max components for normalization
    min_components = min(data_first_plot.keys())
    max_components = max(data_first_plot.keys())

    # Use a colormap
    cmap = plt.cm.viridis

    max_loss = float('-inf')
    min_loss = float('inf')

    # Top plot: Loss evolution vs Image index
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    for components, losses in data_first_plot.items():
        # Normalize the number of components to the range [0, 1]
        norm = (components - min_components) / (max_components - min_components)
        color = cmap(norm)  # Get color from the colormap

        losses = losses.flatten()
        smoothed_losses = uniform_filter1d(losses, size=smoothing_window_size)  # Apply smoothing
        max_loss= max(max_loss, np.max(smoothed_losses))
        min_loss= min(min_loss, np.min(smoothed_losses))
        if components == max(list(data.keys())):
            ax1.plot(range(len(smoothed_losses)), smoothed_losses, color='red', alpha=0.9, linewidth=3, label='Max rank computed') 
        else:
            ax1.plot(range(len(smoothed_losses)), smoothed_losses, color=color, alpha=0.7, linewidth=2)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_components, vmax=max_components))
    cbar = fig.colorbar(sm, ax=ax1)
    cbar.set_label('Number of components', rotation=270, labelpad=15)

    ax1.set_xlabel('Image index')
    ax1.set_ylabel(metric)
    ax1.set_title(f'{metric} evolution vs Image index')
    ax1.grid(True, alpha=0.3)
    if metric=='ssim':
        max_loss=1/1.1
    ax1.set_ylim(min_loss*0.9, max_loss * 1.1)

    # Bottom-left plot: Loss distribution vs number of components used
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    box_data = [losses.flatten() for losses in data.values()]
    ax2.boxplot(box_data, labels=data.keys())
    ax2.set_xlabel('Number of components')
    ax2.set_ylabel(f'{metric} distribution')
    ax2.set_title(f'{metric} distribution vs Number of Components')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right') ## TEST
    ax2.grid(True, alpha=0.3)

    # Bottom-right plot: Mean Loss vs Number of Components
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    mean_losses = {key: np.mean(data[key]) for key in data}
    ax3.plot(list(mean_losses.keys()), list(mean_losses.values()), marker='o', color='blue', linestyle='-', linewidth=2)
    
    ax3.set_xlabel('Number of components')
    ax3.set_ylabel(f'Mean {metric}')
    ax3.set_title(f'Mean {metric} vs Number of Components')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'analyse_{metric}.png', dpi=300, bbox_inches='tight')

import h5py
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import make_interp_spline
from scipy.ndimage import uniform_filter1d

def merge_losses(num_nodes,path_to_files='./'):
    dict_fused_losses={}
    for node_id in range(num_nodes):
        with open(f"{path_to_files}losses_node_{node_id}.pickle", 'rb') as f:
            loaded_data = pickle.load(f)
            
        for key in loaded_data.keys():
            if key not in dict_fused_losses:
                dict_fused_losses[key]= [[] for _ in range(num_nodes)]
            dict_fused_losses[key][node_id].append(loaded_data[key])

    for key in dict_fused_losses.keys():
        loss_key=None
        for loss_node in dict_fused_losses[key]:
            if loss_key is not None:
                loss_key+=np.array(loss_node)**2
            else:
                loss_key=np.array(loss_node)**2
        dict_fused_losses[key]=np.sqrt(loss_key)

    for key in dict_fused_losses.keys():
        dict_fused_losses[key] = dict_fused_losses[key]/dict_fused_losses['original']
        print(f"For number of components: {key}, min loss: {dict_fused_losses[key].min()},max loss: {dict_fused_losses[key].max()},average loss: {dict_fused_losses[key].mean()}",flush=True)

    del dict_fused_losses['original']
    return dict_fused_losses

def draw_plots(data, plot_components=None, smoothing_window_size=50):
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
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss evolution vs Image index')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(min_loss*0.9, max_loss * 1.1)

    # Bottom-left plot: Loss distribution vs number of components used
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    box_data = [losses.flatten() for losses in data.values()]
    ax2.boxplot(box_data, labels=data.keys())
    ax2.set_xlabel('Number of components')
    ax2.set_ylabel('Loss distribution')
    ax2.set_title('Loss distribution vs Number of Components')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right') ##TEST
    ax2.grid(True, alpha=0.3)

    # Bottom-right plot: Mean Loss vs Number of Components
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    mean_losses = {key: np.mean(data[key]) for key in data}
    ax3.plot(list(mean_losses.keys()), list(mean_losses.values()), marker='o', color='blue', linestyle='-', linewidth=2)
    
    ax3.set_xlabel('Number of components')
    ax3.set_ylabel('Mean Loss')
    ax3.set_title('Mean Loss vs Number of Components')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reorganized_analyse_loss.png', dpi=300, bbox_inches='tight')

def clean(num_nodes):
    for node_id in range(num_nodes):
        f = f"{path_to_files}losses_node_{node_id}.pickle"
        if os.path.exists(f):  
            os.remove(f)
            print(f"File deleted : {f}")
        else:
            print(f"File not found : {f}")
    
def main(num_nodes,path,delete_or_not,plot_components,smoothing_window_size):
    time1=time.time()
    data=merge_losses(num_nodes,path)
    time2=time.time()
    draw_plots(data,plot_components,smoothing_window_size)
    time3=time.time()
    if delete_or_not:
        clean(num_nodes)
    print(f"Merged losses from nodes in {time2-time1} (s)",flush=True)
    print(f"Drew plots in {time3-time2} (s)",flush=True)

        
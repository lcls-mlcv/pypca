import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import random
import h5py
import pandas as pd
import pickle
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from mpl_toolkits.axes_grid1 import ImageGrid
from data_loading.ipsana import retrieve_pixel_index_map, assemble_image_stack_batch, PsanaInterface
import time

def binning_indices_with_centroids(embedding, grid_size=50):
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    x_bin_size = (x_max - x_min) / grid_size
    y_bin_size = (y_max - y_min) / grid_size

    bins = {}
    bin_centers = []
    binned_indices = []

    for index, (x, y) in enumerate(embedding):
        x_bin = int((x - x_min) / x_bin_size)
        y_bin = int((y - y_min) / y_bin_size)

        x_bin = min(x_bin, grid_size - 1)
        y_bin = min(y_bin, grid_size - 1)

        bin_key = (x_bin, y_bin)

        if bin_key not in bins:
            bins[bin_key] = []
        
        bins[bin_key].append(index)

        x_center = x_min + (x_bin + 0.5) * x_bin_size
        y_center = y_min + (y_bin + 0.5) * y_bin_size

        bin_centers.append((x_center, y_center))
        binned_indices.append(index)

    return bins, np.array(bin_centers), binned_indices, x_min, y_min, x_max, y_max

def plot_t_sne_scatters(filename, eps=0.1, min_samples=10, num_panels=[0], grid_size=50, save_figures=False, num_compo=1, guiding_panel=-1):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    global_embedding = np.array(data["embeddings"])
    panel_embeddings = np.array(data["embeddings_rank"])
    title = 'Projections'
    S = np.array(data["S"])
    num_gpus = len(S)
    nb_panels = min(len(num_panels), num_gpus)

    n_rows = int(np.ceil(np.sqrt(nb_panels + 1)))
    n_cols = int(np.ceil((nb_panels + 1) / n_rows))

    fig = make_subplots(rows=n_rows, cols=n_cols, 
                        subplot_titles=["Global Projection"] + [f"Panel {i+1}" for i in range(nb_panels)])

    global_clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(global_embedding)
    global_df = pd.DataFrame({
        'Dimension1': global_embedding[:, 0],
        'Dimension2': global_embedding[:, 1],
        'Index': range(len(global_embedding)),
        'Cluster': global_clustering.labels_
    })

    global_scatter = go.Scatter(
        x=global_df['Dimension1'],
        y=global_df['Dimension2'],
        mode='markers',
        marker=dict(color=global_df['Cluster'], colorscale='Viridis', showscale=False),
        text=global_df['Index'],
        hoverinfo='text',
        showlegend=False
    )

    fig.add_trace(global_scatter, row=1, col=1)

    for i in range(nb_panels):
        row = (i + 1) // n_cols + 1
        col = (i + 1) % n_cols + 1

        embedding = panel_embeddings[num_panels[i]]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embedding)
        
        df = pd.DataFrame({
            'Dimension1': embedding[:, 0],
            'Dimension2': embedding[:, 1],
            'Index': range(len(embedding)),
            'Cluster': clustering.labels_
        })

        scatter = go.Scatter(
            x=df['Dimension1'],
            y=df['Dimension2'],
            mode='markers',
            marker=dict(color=df['Cluster'], colorscale='Viridis', showscale=False),
            text=df['Index'],
            hoverinfo='text',
            showlegend=False
        )

        fig.add_trace(scatter, row=row, col=col)

    fig.update_layout(height=500*n_rows, width=500*n_cols, title_text=title)
    fig.update_xaxes(title_text="Dimension 1")
    fig.update_yaxes(title_text="Dimension 2")

    if save_figures:
        fig.write_html(f"./visuals/plots/projections_in_2D_{num_compo}.html")
    else:
        fig.show()

    if guiding_panel==-1:
        embedding_for_binning=global_embedding
    else:
        embedding_for_binning=panel_embeddings[guiding_panel]
        
    _, binned_centers, binned_indices, x_min, y_min, x_max, y_max = binning_indices_with_centroids(embedding_for_binning, grid_size=grid_size)
    binned_df = pd.DataFrame(binned_centers, columns=['Binned_Dimension1', 'Binned_Dimension2'])
    binned_df['Index'] = binned_indices

    binned_scatter = go.Scatter(
        x=binned_df['Binned_Dimension1'],
        y=binned_df['Binned_Dimension2'],
        mode='markers',
        marker=dict(color='rgba(0,0,255,0.5)', size=8),
        text=binned_df['Index'],
        hoverinfo='text',
        showlegend=False,
        name='Binned'
    )

    binned_fig = go.Figure(data=binned_scatter)
    binned_fig.update_layout(title='Binned Embedding', xaxis_title='Dimension 1', yaxis_title='Dimension 2', height=500, width=500)
    
    if save_figures:
        binned_fig.write_html(f"./visuals/plots/binned_embedding_{num_compo}.html")
    else:
        binned_fig.show()

    return x_min, y_min, x_max, y_max

def create_average_img(proj_binned, V, mu):
    img_binned = {}
    V = np.array(V)
    mu = np.array(mu)
    count=0
    total_im = len(proj_binned.keys())
    start_time = time.time()
    first_rank = None
    for key, proj in proj_binned.items():
        if (count+1)%50==0:
            print(f"{count+1}/{total_im} images treated", flush=True)
        count+=1
            
        if len(proj) == 0:
            img_binned[key] = None
        else:
            avg_img = []
            if not first_rank:
                first_rank = V.shape[2]-np.array(proj)[0].shape[0]
                
            for rank in range(V.shape[0]):
                panel = np.dot(proj[rank], V[rank,:,first_rank:].T) + mu[rank]
                avg_img.append(panel)
            img_binned[key] = np.array(avg_img)

    print("Reconstructing time (s):",time.time()-start_time)
    return img_binned

def averaged_imgs(model_filename, filename, downsample_factor=10, coords=(0,0,1,1),save_figures=False, pickle_path='img_binned.pkl'):
    img_binned = {}
    weights = {}
    start_time=time.time()
    with h5py.File(filename, "r") as hdf:
        group = hdf['proj_binned']
        weights_group = hdf['weights']
        for dataset_name in group.keys():
            img_binned[dataset_name] = group[dataset_name][()]
            weights[dataset_name] = weights_group[dataset_name][()]

    print("Gathered projectors and weights",flush=True)
    
    with h5py.File(model_filename, 'r') as f:
        metadata = f['metadata']
        exp = str(np.asarray(metadata.get('exp')))[2:-1]
        run = int(np.asarray(metadata.get('run')))
        det_type = str(np.asarray(metadata.get('det_type')))[2:-1]
        V = f['V']
        num_compo = V.shape[2]
        mu = f['mu'][:]
        img_binned = create_average_img(img_binned, V, mu)
        mu = np.array(mu)
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(img_binned, f)
        pickle.dump(weights, f)
        pickle.dump(mu, f)  ########################################################################TEST
    
    int_time1=time.time()
    print(f"Average images created and stored in {int_time1-start_time} (s)", flush=True)
    
    psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
    a, b, c = psi.det.shape()
    pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))
    
    keys = list(img_binned.keys())
    grid_size = int(np.sqrt(len(keys)))

    bin_data = {}
    for key in keys:
        if img_binned[key] is not None:
            img = img_binned[key]
            img = img.reshape((a, b, c))
            img = assemble_image_stack_batch(img, pixel_index_map)
            bin_data[key] = img
        else:
            bin_data[key] = None
    
    binned_centers = []
    for key in keys:
        if img_binned[key] is not None:
            x, y = map(float, key.split("_"))
            binned_centers.append((x, y))
    
    binned_centers = np.array(binned_centers)
    
    if binned_centers.size > 0:
        binned_df = pd.DataFrame(binned_centers, columns=['Binned_Dimension1', 'Binned_Dimension2'])
    else:
        binned_df = pd.DataFrame(columns=['Binned_Dimension1', 'Binned_Dimension2'])

    binned_scatter = go.Scatter(
        x=binned_df['Binned_Dimension1'],
        y=binned_df['Binned_Dimension2'],
        mode='markers',
        marker=dict(color='rgba(0,0,255,0.5)', size=8),
        showlegend=False,
        name='Binned'
    )

    fig = go.Figure(data=binned_scatter)
    fig.update_layout(title='Binned Embedding', xaxis_title='Dimension 1', yaxis_title='Dimension 2', height=500, width=500)
    
    if save_figures:
        fig.write_html(f"./visuals/plots/binned_embedding_check_{num_compo}.html")
    else:
        fig.show()
    
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(grid_size, grid_size),
                    axes_pad=0.1,
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.05)

    start_time = time.time()
    empty_counter=0
    x_min, y_min, x_max, y_max = coords
    for idx, key in enumerate(keys):
        if idx % 50 == 0:
            print(f"Processing bin {idx+1}/{len(keys)}")
    
        x, y = map(float, key.split('_'))
        
        grid_x = int((x - x_min) / (x_max - x_min) * (grid_size))
        #grid_y = int((y - y_min) / (y_max - y_min) * (grid_size))
        grid_y = int((y_max - y) / (y_max - y_min) * (grid_size))
        
        if bin_data[key] is None:
            empty_counter += 1
            continue
        else:
            downsampled_img = bin_data[key][::downsample_factor, ::downsample_factor]
            i=grid_y*grid_size+grid_x
            im = grid[i].imshow(downsampled_img, cmap='viridis', 
                                             vmin=np.percentile(bin_data[key], 5), 
                                             vmax=np.percentile(bin_data[key], 95))
            grid[i].axis('off')
            
    print(f"All bins processed, {empty_counter} empty bins",flush=True)
    
    plt.colorbar(im, cax=grid.cbar_axes[0])
    plt.tight_layout()
    plt.show()
    plt.close()
    print("Time taken to plot grid (s):",time.time()-start_time,flush=True)

def cluster_averaged_imgs(model_filename, filename, num_clusters=3, dbscan=None, coords=(0,0,1,1), downsample_factor=10, pickle_path=None,guiding_panel=0):
    start_time=time.time()

    if pickle_path is None:
        img_binned = {}
        weights = {}
        with h5py.File(filename, "r") as hdf:
            group = hdf['proj_binned']
            weights_group = hdf['weights']
            for dataset_name in group.keys():
                img_binned[dataset_name] = group[dataset_name][()]
                weights[dataset_name] = weights_group[dataset_name][()]
                
        with h5py.File(model_filename, 'r') as f:
            metadata = f['metadata']
            exp = str(np.asarray(metadata.get('exp')))[2:-1]
            run = int(np.asarray(metadata.get('run')))
            det_type = str(np.asarray(metadata.get('det_type')))[2:-1]
            V = f['V']
            mu = f['mu'][:]
            img_binned = create_average_img(img_binned, V, mu)
    else:
        with h5py.File(model_filename, 'r') as f:
            metadata = f['metadata']
            exp = str(np.asarray(metadata.get('exp')))[2:-1]
            run = int(np.asarray(metadata.get('run')))
            det_type = str(np.asarray(metadata.get('det_type')))[2:-1]
        with open(pickle_path, 'rb') as f:
            img_binned = pickle.load(f)
            weights = pickle.load(f)
            mu = pickle.load(f) ##########################################################################################"
    
    int_time1=time.time()
    print(f"Average images and weights created/gathered in {int_time1-start_time} (s)", flush=True)
    grid_size = int(np.sqrt(len(img_binned.keys())))
            
    keys = [key for key in img_binned.keys() if img_binned[key] is not None]
    coordinates = np.array([list(map(float, key.split('_'))) for key in keys])
    images = np.array([img_binned[key].flatten() for key in keys])
    weights = np.array([weights[key] for key in keys])
    scaler_coord = StandardScaler()
    
    coordinates_scaled = scaler_coord.fit_transform(coordinates)

    if dbscan: # Clustering DBSCAN
        eps,min_samples = dbscan
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(coordinates_scaled)
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Number of clusters found: {num_clusters}")
    else:
        # Clustering K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates_scaled)

    int_time2=time.time()
    print(f"Clustering done in {int_time2 - int_time1} (s)!",flush=True)
    
    # Compute statistics cluster by cluster
    cluster_stats = {}
    for i in range(num_clusters):
        cluster_images = images[cluster_labels == i]
        weights_cluster = weights[cluster_labels == i]
        for idx in range(len(weights_cluster)):
            cluster_images[idx] = cluster_images[idx] * weights_cluster[idx]
        cluster_coords = coordinates[cluster_labels == i]
        mean_image = np.sum(cluster_images, axis=0) / np.sum(weights_cluster) #####################################################
        var_image = np.sqrt(
            np.sum(
                weights_cluster[:, np.newaxis] * (images[cluster_labels == i] - mean_image)**2,
                axis=0
            ) / np.sum(weights_cluster)
        )
        cluster_stats[i] = {
            'mean_image': mean_image,
            'var_image': var_image
        }
        
    psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
    a, b, c = psi.det.shape()
    pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))

    for i in range(num_clusters):
        cluster_stats[i]['mean_image'] = cluster_stats[i]['mean_image'].reshape((a, b, c))
        cluster_stats[i]['var_image'] = cluster_stats[i]['var_image'].reshape((a, b, c))
        cluster_stats[i]['mean_image'] = assemble_image_stack_batch(cluster_stats[i]['mean_image'], pixel_index_map)
        cluster_stats[i]['var_image'] = assemble_image_stack_batch(cluster_stats[i]['var_image'], pixel_index_map)
    int_time3=time.time()
    print(f"Reconstructing mean and variance images done in {int_time3-int_time2} (s)!",flush=True)
    n_cols = 4  
    n_rows = max(int(np.ceil(num_clusters / 2)),2)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    fig.suptitle("Cluster Mean and Variance Images", fontsize=16)
    
    for i in range(num_clusters):
        row = i // 2
        col = (i % 2) * 2
    
        im_mean = axs[row, col].imshow(cluster_stats[i]['mean_image'], cmap='viridis',vmin=np.percentile(cluster_stats[i]['mean_image'], 5),vmax=np.percentile(cluster_stats[i]['mean_image'], 95))
        axs[row, col].set_title(f'Cluster {i} - Mean')
        fig.colorbar(im_mean, ax=axs[row, col])
    
        im_var = axs[row, col+1].imshow(cluster_stats[i]['var_image'], cmap='plasma',vmin=np.percentile(cluster_stats[i]['var_image'], 5),vmax=np.percentile(cluster_stats[i]['var_image'], 95))
        axs[row, col+1].set_title(f'Cluster {i} - Variance')
        fig.colorbar(im_var, ax=axs[row, col+1])
    
        cluster_size = np.sum(weights[cluster_labels == i])
        coord_text = f'Size: {cluster_size} or {cluster_size/np.sum(weights)*100:.2f}%'
        axs[row, col].text(0.5, -0.1, coord_text, transform=axs[row, col].transAxes, ha='center')
    
    plt.tight_layout()
    plt.show()
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
    fig_scatter.suptitle("Cluster Assignments", fontsize=16)
    
    scatter = ax_scatter.scatter(coordinates[:, 0], coordinates[:, 1], c=cluster_labels, cmap='viridis')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster')
    
    ax_scatter.set_xlabel('X Coordinate')
    ax_scatter.set_ylabel('Y Coordinate')
    ax_scatter.set_title('Cluster Assignments for Each Index')

    plt.tight_layout()
    plt.show()

    #HARD CODED PANEL VISUALISATION
    print("Warning. This has been hard-coded for mfxp23120 and pypca with 16 GPUs.",flush=True)
    panel_img = []
    for panel in range(a):
        if panel == guiding_panel:
            panel_img.append(np.ones((b,c)))
        else:
            panel_img.append(np.zeros((b,c)))
    panel_img = np.array(panel_img).reshape((a,b,c))
    panel_img = assemble_image_stack_batch(panel_img, pixel_index_map)

    plt.tight_layout()
    plt.title('Guiding panel correspondance')
    plt.imshow(panel_img)

def random_walk_animation(bin_data_path, steps=50, save_path="random_walk_animation", interval=500, fps=2, fade_frames=5, grid_size=50):
    # Load bin data
    bin_data = np.load(bin_data_path, allow_pickle=True).item()
    keys = list(bin_data.keys())

    # Helper function to find the closest valid position
    def find_closest_valid_position(x, y, direction, valid_positions, keys):
        dx, dy = direction
        new_x, new_y = x, y
        step = 1.0
        max_steps = 100
        
        for _ in range(max_steps):
            new_x += dx * step
            new_y += dy * step
            
            for nearby_key in keys:
                nearby_x, nearby_y = map(float, nearby_key.split("_"))
                if (abs(nearby_x - new_x) < step and 
                    abs(nearby_y - new_y) < step and 
                    valid_positions[nearby_key]):
                    return keys.index(nearby_key)
        
        return None

    # Create dictionary marking valid (non-blank) positions
    valid_positions = {key: bin_data[key] is not None for key in keys}
    
    # Generate path
    valid_keys = [key for key in keys if valid_positions[key]]
    current_key = random.choice(valid_keys)
    path = [keys.index(current_key)]
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for _ in range(steps):
        x, y = map(float, keys[path[-1]].split("_"))
        valid_neighbors = []
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            key = f"{new_x}_{new_y}"
            if key in keys and valid_positions[key]:
                valid_neighbors.append(keys.index(key))
        
        if valid_neighbors:
            next_idx = random.choice(valid_neighbors)
            path.append(next_idx)
        else:
            random_direction = random.choice(directions)
            closest_idx = find_closest_valid_position(x, y, random_direction, valid_positions, keys)
            if closest_idx is not None:
                path.append(closest_idx)
            else:
                path.append(path[-1])
    
    # Create figure with two side-by-side subplots
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    ax_det = fig.add_subplot(gs[0])
    ax_pos = fig.add_subplot(gs[1])
    
    def update(frame):
        ax_det.clear()
        ax_pos.clear()
        
        main_frame = frame // (fade_frames + 1)
        sub_frame = frame % (fade_frames + 1)
        
        # Display detector image with fade effect
        current_idx = path[min(main_frame, len(path)-1)]
        current_key = keys[current_idx]
        img = bin_data[current_key]
        
        if main_frame < len(path) - 1 and sub_frame > 0:
            next_idx = path[main_frame + 1]
            next_key = keys[next_idx]
            next_img = bin_data[next_key]
            
            if img is not None and next_img is not None:
                alpha = sub_frame / (fade_frames + 1)
                img = (1 - alpha) * img + alpha * next_img
        
        if img is not None:
            masked_img = np.ma.masked_where(np.isnan(img), img)
            ax_det.imshow(masked_img, cmap='viridis')
        
        # Update position visualization
        x_coords, y_coords = zip(*[map(float, key.split("_")) for key in keys])
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        ax_pos.set_xlim(x_min, x_max)
        ax_pos.set_ylim(y_min, y_max)
        
        # Mark all valid positions
        valid_x = [float(key.split("_")[0]) for key in keys if valid_positions[key]]
        valid_y = [float(key.split("_")[1]) for key in keys if valid_positions[key]]
        ax_pos.scatter(valid_x, valid_y, c='lightgray', alpha=0.5)
        
        # Mark visited positions
        visited_x = [float(keys[idx].split("_")[0]) for idx in path[:main_frame+1]]
        visited_y = [float(keys[idx].split("_")[1]) for idx in path[:main_frame+1]]
        ax_pos.scatter(visited_x, visited_y, c='blue', alpha=0.7)
        
        # Mark current position
        current_x, current_y = map(float, keys[path[min(main_frame, len(path)-1)]].split("_"))
        ax_pos.scatter([current_x], [current_y], c='red', s=100)
        
        # Set proper axis limits and remove ticks
        ax_det.set_axis_off()
        ax_pos.set_axis_off()
        
        return ax_det.artists + ax_pos.artists
    
    # Create and save animation
    total_frames = (len(path) - 1) * (fade_frames + 1) + 1
    ani = animation.FuncAnimation(fig, update, 
                                  frames=total_frames, 
                                  interval=interval, 
                                  blit=True)
    
    save_path += f'_{fps}fps.gif'
    writer = animation.PillowWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()

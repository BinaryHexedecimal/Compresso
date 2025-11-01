from PIL import Image
import io
#import matplotlib
#matplotlib.use("Agg")  # Use non-GUI backend. Vigtigt!!!!
import base64
import random
import numpy as np
import plotly.graph_objects as go
import networkx as nx


def get_images(dataset_tensor_by_label, N: int, 
               sequential_offset: int, random_mode: bool):
    #shape: (num_samples, C, H, W)
    num_samples = dataset_tensor_by_label.shape[0]
    
    if num_samples == 0:
        return []
    elif num_samples < N:
        N = num_samples
    if random_mode:
        # Random selection
        selected_indices = random.sample(range(num_samples), min(N, num_samples))
        new_sequential_offset = sequential_offset
    else:
        # Sequential selection
        start_idx = sequential_offset
        end_idx = start_idx + N
        
        # Wrap around if needed
        selected_indices = list(range(start_idx, min(end_idx, num_samples)))
        if end_idx > num_samples:
            selected_indices += list(range(0, end_idx - num_samples))
        
        # Update offset for next call
        new_sequential_offset = end_idx % num_samples
    
    # Fetch images
    images = [dataset_tensor_by_label[idx] for idx in selected_indices]
    
    return new_sequential_offset, images




def tensor_to_image_bytes(tensor):
    arr = tensor.cpu().numpy()  # (C,H,W)
    
    # Handle grayscale vs RGB automatically
    if arr.shape[0] == 1:  # grayscale
        arr = arr.squeeze(0)  # H,W
        mode = "L"
    else:  # RGB
        arr = np.transpose(arr, (1, 2, 0))  # H,W,C
        mode = "RGB"

    # Convert float [0,1] or int [0,255] to uint8
    if arr.dtype == np.float32 or arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    img = Image.fromarray(arr, mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()



def transform_images_for_frontend(images):
    images_base64 = []
    for img in images:
        arr = (img.numpy().transpose(1,2,0) * 255).astype(np.uint8)
        if arr.shape[-1] == 1:  # grayscale
            arr = arr.squeeze(-1)
        pil_img = Image.fromarray(arr)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        img_str = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
        images_base64.append(img_str)

    return images_base64



def draw_graph(G, c: int):
    weights = np.array([data.get("weight", 1.0) for _, _, data in G.edges(data=True)])

    # Apply log compression (add 1 to avoid log(0))
    log_w = np.log1p(weights)  # log(1 + w)
    log_min, log_max = log_w.min(), log_w.max()

    # Normalize to [0, 1]
    w_norm = (log_w - log_min) / (log_max - log_min + 1e-9)

    # Invert for spring layout
    inv_weights = 1.0  - w_norm
    
    # Assign back
    for (u, v, data), inv_w in zip(G.edges(data=True), inv_weights):
        data["inv_weight"] = inv_w

    pos = nx.spring_layout(G, weight="inv_weight", seed=42)
    nx.set_node_attributes(G, pos, 'pos')

    # ----------------- Draw edges -----------------
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(width=0.5, color='#888'),  # uniform width
            hoverinfo='none',
            customdata=[[u, v]],
            name='edge'
        )
        edge_traces.append(trace)

    # ----------------- Draw nodes -----------------
    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_adjacencies = [len(adj) for _, adj in G.adjacency()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='none',
        customdata=list(G.nodes()),
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_adjacencies,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left'
            ),
            line_width=2
        )
    )

    # ----------------- Highlight (center) nodes -----------------
    highlight_x, highlight_y = [], []
    for i in range(c):
        x, y = pos[i]
        highlight_x.append(x)
        highlight_y.append(y)

    highlight_trace = go.Scatter(
        x=highlight_x, y=highlight_y,
        mode='markers',
        customdata=list(range(c)),
        marker=dict(
            symbol='x',
            size=10,
            color='red',
            line=dict(width=1, color='darkred')
        ),
        hoverinfo='none',
        name='Highlighted Nodes'
    )

    # ----------------- Build final figure -----------------
    fig = go.Figure(
        data=edge_traces + [node_trace, highlight_trace],
        layout=go.Layout(
            title=None,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    text="Python code: <a href='https://plotly.com/python/network-graphs/'>Plotly Network Graph</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ]
        )
    )
    fig.update_layout(showlegend=False, hovermode='closest')

    return fig

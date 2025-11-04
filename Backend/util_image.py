from PIL import Image
import io
import base64
import random
import numpy as np
import plotly.graph_objects as go
import networkx as nx


# Select N images either randomly or sequentially; updates offset for next call.
def get_images(dataset_tensor_by_label, N: int, sequential_offset: int, random_mode: bool):
    num_samples = dataset_tensor_by_label.shape[0]

    if num_samples == 0:
        return []
    elif num_samples < N:
        N = num_samples

    if random_mode:
        selected_indices = random.sample(range(num_samples), min(N, num_samples))
        new_sequential_offset = sequential_offset
    else:
        start_idx = sequential_offset
        end_idx = start_idx + N
        selected_indices = list(range(start_idx, min(end_idx, num_samples)))
        if end_idx > num_samples:
            selected_indices += list(range(0, end_idx - num_samples))
        new_sequential_offset = end_idx % num_samples

    images = [dataset_tensor_by_label[idx] for idx in selected_indices]
    return new_sequential_offset, images


# Convert a torch tensor (C,H,W) to raw PNG bytes.
def tensor_to_image_bytes(tensor):
    arr = tensor.cpu().numpy()
    if arr.shape[0] == 1:     # grayscale
        arr, mode = arr.squeeze(0), "L"
    else:                    # RGB
        arr, mode = np.transpose(arr, (1, 2, 0)), "RGB"

    if arr.dtype == np.float32 or arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    img = Image.fromarray(arr, mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# Convert images (tensors) to base64-encoded PNG strings for frontend preview.
def transform_images_for_frontend(images):
    images_base64 = []
    for img in images:
        arr = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        if arr.shape[-1] == 1:  # grayscale
            arr = arr.squeeze(-1)
        pil_img = Image.fromarray(arr)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        img_str = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
        images_base64.append(img_str)
    return images_base64


# Draw a NetworkX graph in Plotly, highlighting the first c nodes.
def draw_graph(G, c: int):
    weights = np.array([data.get("weight", 1.0) for _, _, data in G.edges(data=True)])

    # Compress weights with log scaling and normalize to [0,1]
    log_w = np.log1p(weights)
    log_min, log_max = log_w.min(), log_w.max()
    w_norm = (log_w - log_min) / (log_max - log_min + 1e-9)

    # Invert weights for layout (close = strong attraction)
    inv_weights = 1.0 - w_norm
    for (u, v, data), inv_w in zip(G.edges(data=True), inv_weights):
        data["inv_weight"] = inv_w

    pos = nx.spring_layout(G, weight="inv_weight", seed=42)
    nx.set_node_attributes(G, pos, 'pos')

    # Edges
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines', line=dict(width=0.5, color='#888'),
            hoverinfo='none', customdata=[[u, v]], name='edge'
        ))

    # Nodes
    node_x, node_y = zip(*[pos[n] for n in G.nodes()])
    node_adjacencies = [len(adj) for _, adj in G.adjacency()]
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='none',
        customdata=list(G.nodes()),
        marker=dict(
            showscale=True, colorscale='YlGnBu', reversescale=True,
            color=node_adjacencies, size=10, line_width=2,
            colorbar=dict(thickness=15, title='Node Connections', xanchor='left')
        )
    )

    # Highlight first c nodes (centers)
    highlight_x, highlight_y = zip(*[pos[i] for i in range(c)]) if c > 0 else ([], [])
    highlight_trace = go.Scatter(
        x=highlight_x, y=highlight_y, mode='markers', customdata=list(range(c)),
        marker=dict(symbol='x', size=10, color='red', line=dict(width=1, color='darkred')),
        hoverinfo='none', name='Highlighted Nodes'
    )

    # Build final figure
    fig = go.Figure(
        data=edge_traces + [node_trace, highlight_trace],
        layout=go.Layout(
            title=None, showlegend=False, hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    text="Python code: <a href='https://plotly.com/python/network-graphs/'>Plotly Network Graph</a>",
                    showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002
                )
            ]
        )
    )
    fig.update_layout(showlegend=False, hovermode='closest')
    return fig

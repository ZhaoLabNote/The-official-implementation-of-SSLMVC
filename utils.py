import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def sinkhorn(Q, nmb_iters):
    '''Sinkhorn algorithm'''
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= (sum_Q + 1e-8)

        device = Q.device
        r = torch.ones(Q.shape[0], device=device) / Q.shape[0]
        c = torch.ones(Q.shape[1], device=device) / (1 * Q.shape[1])

        for _ in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            u = r / (u + 1e-8)
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / (torch.sum(Q, dim=0) + 1e-8)).unsqueeze(0)

        return (Q / (torch.sum(Q, dim=0, keepdim=True) + 1e-8)).t().float()

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    if inp_tensor.isinf().any():
        m = inp_tensor[torch.isfinite(inp_tensor)].max() if torch.isfinite(inp_tensor).any() else torch.tensor(0.0, device=inp_tensor.device)
        inp_tensor = torch.where(torch.isinf(inp_tensor), m, inp_tensor)
    return inp_tensor

def process_output(output1, output2):
    epsilon = 0.3
    n_ite = 10 
    q1 = output1 / epsilon
    q2 = output2 / epsilon
    q1 = torch.exp(q1).t()
    q2 = torch.exp(q2).t()
    q1 = sinkhorn(q1, n_ite)
    q2 = sinkhorn(q2, n_ite)
    return q1, q2

def t_distribution(distances, df=1.0):

    distances = np.maximum(distances, np.finfo(np.float32).eps)
    
    q = 1.0 / (1.0 + distances / df)
    q = q / np.sum(q, axis=1, keepdims=True)
    
    return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    p = (weight.T / weight.sum(1)).T
    return p

def update_cluster_centers(features, soft_assignments, centers, alpha=0.9):
    n_clusters = centers.shape[0]
    new_centers = np.zeros_like(centers)
    
    for j in range(n_clusters):
        weighted_sum = np.sum(features * soft_assignments[:, j:j+1], axis=0)
        weight_sum = np.sum(soft_assignments[:, j])
        if weight_sum > 0:
            new_centers[j] = weighted_sum / weight_sum
    
    new_centers = alpha * centers + (1 - alpha) * new_centers
    
    return new_centers

def generate_pseudo_labels(model, device, dataset, view, class_num, update_interval=5, pca_dim=None, alpha=0.5, current_epoch=0):
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )
    
    Z_views_list = [[] for _ in range(view)]
    collected_gate_activations_per_view = [[] for _ in range(view)]

    with torch.no_grad():
        for batch_idx, (xs, _, _) in enumerate(dataloader):
            for v_idx in range(view):
                xs[v_idx] = xs[v_idx].to(device)
            all_zs, _, batch_gate_activations = model.forward_plot(xs)
            current_batch_zs_views = all_zs[:view]

            for v_idx in range(view):
                clean_tensor = torch.nan_to_num(current_batch_zs_views[v_idx], nan=0.0, posinf=1e6, neginf=-1e6)
                Z_views_list[v_idx].append(clean_tensor.cpu().numpy())
                if batch_gate_activations and len(batch_gate_activations) == view:
                    collected_gate_activations_per_view[v_idx].append(batch_gate_activations[v_idx].item())
    
    for v_idx in range(view):
        Z_views_list[v_idx] = np.vstack(Z_views_list[v_idx])
        Z_views_list[v_idx] = np.nan_to_num(Z_views_list[v_idx], nan=0.0, posinf=1e6, neginf=-1e6)
    
    avg_gate_activations = [np.mean(collected_gate_activations_per_view[v_idx]) if collected_gate_activations_per_view[v_idx] else 0 for v_idx in range(view)]
    
    sum_activations = np.sum(avg_gate_activations)
    if sum_activations > 0:
        view_weights = np.array(avg_gate_activations) / sum_activations
    else:
        print("Warning: All gate activations are zero or empty. Using uniform view weights.")
        view_weights = np.ones(view) / view
    
    print(f"Calculated view weights based on gate activations: {view_weights}")

    if pca_dim is not None:
        for v_idx in range(view):
            if Z_views_list[v_idx].shape[1] > 30: 
                pca = PCA(n_components=0.95) 
                Z_views_list[v_idx] = pca.fit_transform(Z_views_list[v_idx])
                Z_views_list[v_idx] = np.nan_to_num(Z_views_list[v_idx], nan=0.0, posinf=1e6, neginf=-1e6)
                print(f"View {v_idx} reduced to {Z_views_list[v_idx].shape[1]} dimensions")
    
    Z_global_list = []
    for v_idx in range(view):
        weighted_features = Z_views_list[v_idx] * view_weights[v_idx]
        Z_global_list.append(weighted_features)
    Z_global = np.concatenate(Z_global_list, axis=1)
    Z_global = np.nan_to_num(Z_global, nan=0.0, posinf=1e6, neginf=-1e6)
    
    N = Z_global.shape[0]
    
    kmeans = KMeans(n_clusters=class_num, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(Z_global)
    centers = kmeans.cluster_centers_
    
    max_iter = 10
    tol = 1e-4
    center_shift = float('inf')
    
    for i in range(max_iter):
        if center_shift < tol:
            break

        distances = np.zeros((N, class_num))
        for j in range(class_num):
            distances[:, j] = np.sum(np.square(Z_global - centers[j]), axis=1)
        
        q = t_distribution(distances)
        
        old_centers = centers.copy()
        
        centers = update_cluster_centers(Z_global, q, centers, alpha=0.9)
        
        center_shift = np.linalg.norm(centers - old_centers)
    
    distances = np.zeros((N, class_num))
    for j in range(class_num):
        distances[:, j] = np.sum(np.square(Z_global - centers[j]), axis=1)
    
    Q = t_distribution(distances, df=1.0)
    
    P = target_distribution(Q)
    
    model.train()
    return P, view_weights

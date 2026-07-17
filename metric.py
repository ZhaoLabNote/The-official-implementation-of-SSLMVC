from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def valid(model, device, dataset, view, data_size, class_num, eval_h=False):
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )

    q = np.zeros((data_size, class_num))
    h = np.zeros((data_size, view))
    q_fusion = np.zeros((data_size, class_num))
    targets = np.zeros(data_size).astype(np.int64)
    start = 0
    
    with torch.no_grad():
        for batch_idx, (xs, target, _) in enumerate(dataloader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            
            target_np = target.cpu().detach().numpy()
            if len(target_np.shape) > 1 and target_np.shape[1] == 1:
                target_np = target_np.squeeze(1)
                
            targets[start:start + target.shape[0]] = target_np
            
            qs, _ = model.forward_cluster(xs)
            
            for v in range(view):
                q[start:start + target.shape[0], :] = q[start:start + target.shape[0], :] + qs[v].cpu().detach().numpy()
            
            q_fusion[start:start + target.shape[0], :] = qs[-1].cpu().detach().numpy()
                
            start += target.shape[0]

    pred = np.argmax(q_fusion, axis=1)
    acc_fusion = cluster_acc(targets, pred)
    nmi_fusion = v_measure_score(targets, pred)
    ari_fusion = adjusted_rand_score(targets, pred)
    pur_fusion = purity(targets, pred)
    
    model.train()
    
    return acc_fusion, nmi_fusion, ari_fusion, pur_fusion

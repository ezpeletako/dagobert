"""Standalone DAG utilities decoupled from replicator class."""
import numpy as np
from typing import List, Set


def compute_bic_score(data: np.ndarray, adjacency: np.ndarray) -> float:
    """Independent BIC scorer usable anywhere."""
    n_samples, n_vars = data.shape
    W = adjacency
    
    log_likelihood = 0.0
    
    for j in range(n_vars):
        y = data[:, j]
        parents = np.where(W[j, :] > 0)[0]
        
        if len(parents) == 0:
            mse = np.var(y)
        else:
            X_p = data[:, parents]
            try:
                beta, _, _, _ = np.linalg.lstsq(X_p, y, rcond=None)
                mse = np.mean((y - X_p @ beta) ** 2)
            except np.linalg.LinAlgError:
                mse = np.var(y)
        
        if mse > 1e-8:
            log_likelihood += -0.5 * n_samples * (np.log(2 * np.pi * mse) + 1)
        else:
            log_likelihood += -0.5 * n_samples * (np.log(2 * np.pi * 1e-8) + 1)
    
    n_edges = int(W.sum())
    bic = log_likelihood - 0.5 * (n_edges + n_vars) * np.log(n_samples)
    return float(bic)


def validate_dag_structure(topo_order: np.ndarray, adj_lower: np.ndarray) -> bool:
    """Sanity checks before instantiation."""
    n = len(topo_order)
    
    if adj_lower.shape != (n, n):
        return False
    
    if not np.all(np.isin(topo_order, np.arange(n))):
        return False
    
    if len(set(topo_order)) != n:
        return False  # Duplicate entries
    
    if not np.allclose(adj_lower, np.tril(adj_lower, k=-1)):
        return False  # Upper triangle should be zero
    
    return True


def extract_markov_boundary(dag: np.ndarray) -> List[int]:
    """Find neighbors (parents + children) for each node."""
    n = dag.shape[0]
    boundary = {}
    
    for node in range(n):
        parents = np.where(dag[node, :] > 0)[0].tolist()
        children = np.where(dag[:, node] > 0)[0].tolist()
        boundary[node] = list(set(parents + children))
    
    return boundary
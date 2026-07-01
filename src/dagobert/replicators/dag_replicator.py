"""Directed Acyclic Graph replicator for causal structure learning."""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.stats import chi2

from .replicator import Replicator


def _has_path(adj: np.ndarray, src: int, dst: int) -> bool:
    """DFS check if path exists from src → dst."""
    visited = set()
    stack = [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node in visited:
            continue
        visited.add(node)
        # Edges go forward in adjacency; scan row
        for neighbor in np.where(adj[node] > 0)[0]:
            if neighbor not in visited:
                stack.append(neighbor)
    return False


def _is_acyclic_direct(adj: np.ndarray) -> bool:
    """Direct cycle check via DFS (fallback safety)."""
    n = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    color = np.full(n, WHITE, dtype=np.int8)
    
    def dfs(u):
        color[u] = GRAY
        for v in range(n):
            if adj[u, v] > 0:
                if color[v] == GRAY:
                    return False  # Back-edge found
                if color[v] == WHITE and not dfs(v):
                    return False
        color[u] = BLACK
        return True
    
    for i in range(n):
        if color[i] == WHITE and not dfs(i):
            return False
    return True


@dataclass(frozen=True)
class DagReplicator(Replicator):
    """Immutability-safe DAG representation with guaranteed acyclicity.
    
    Encoding uses topological order + lower-triangle adjacency:
    - `topo_order`: permutation π where all edges flow π(i) → π(j) with i < j
    - `adj_lower`: binary mask below diagonal representing edges in sorted order
    
    This guarantees no cycles can ever form during mutation/crossover.
    """
    
    n_nodes: int
    topo_order: np.ndarray          # Shape: (n_nodes,), values in [0..n-1]
    adj_lower: np.ndarray           # Shape: (n_nodes, n_nodes), lower triangle only
    data: np.ndarray                # Shape: (n_samples, n_nodes) observation data
    
    def _build_full_adj(self) -> np.ndarray:
        """Convert compact rep to full adjacency matrix for scoring."""
        adj = np.zeros((self.n_nodes, self.n_nodes))
        
        for j in range(self.n_nodes):
            node_j = self.topo_order[j]
            for i in range(j):  # Only lower triangle respects topo order
                if self.adj_lower[j, i]:
                    node_i = self.topo_order[i]
                    adj[node_j, node_i] = 1.0
        
        return adj
    
    def express(self) -> np.ndarray:
        """Return the full adjacency matrix (phenotype)."""
        return self._build_full_adj()
    
    def evaluate(self) -> float:
        """Compute BIC score for DAG on observational data."""
        W = self.express()
        X = self.data
        
        # Linear Gaussian SEM: X = XW^T + ε
        # BIC = −logL + k·log(N) where k = non-zero edges
        
        n_samples, n_vars = X.shape
        
        # Fit regression coefficients: θ = (X^T X)^{-1} X^T X W ? 
        # Simplified: residual variance per variable
        log_likelihood = 0.0
        
        for j in range(n_vars):
            y = X[:, j]
            # Predictors are variables pointing TO j (parents)
            parents = np.where(W[j, :] > 0)[0]
            
            if len(parents) == 0:
                # No incoming edges: fit intercept only
                residuals = y - y.mean()
                mse = np.var(residuals)
            else:
                # Multiple linear regression
                X_parents = X[:, parents]
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X_parents, y, rcond=None)
                    preds = X_parents @ coeffs
                    residuals = y - preds
                    mse = np.mean(residuals ** 2)
                except np.linalg.LinAlgError:
                    # Singular matrix fallback
                    mse = np.var(y)
            
            # Log-likelihood contribution: −N/2 · log(2πσ²) − N/(2σ²) · Σε²
            if mse > 1e-8:
                log_likelihood += -0.5 * n_samples * (np.log(2 * np.pi * mse) + 1)
            else:
                # Numerical stability floor
                log_likelihood += -0.5 * n_samples * (np.log(2 * np.pi * 1e-8) + 1)
        
        # Complexity penalty: degrees of freedom per edge
        n_edges = int(W.sum())
        k_params = n_edges + n_vars  # edge weights + noise variances
        
        bic_penalty = 0.5 * k_params * np.log(n_samples)
        
        return float(log_likelihood - bic_penalty)
    
    def mutate(
        self,
        rng: np.random.Generator,
        p_edge_flip: float = 0.05,
        p_topo_swap: float = 0.1,
    ):
        """Apply stochastic edge and ordering mutations."""
        # Copy arrays (immutable dataclass means fresh instance needed)
        new_adj = self.adj_lower.copy()
        new_order = self.topo_order.copy()
        
        # --- Edge flipping (primary variation) ---
        # Scan lower triangle, flip bits with probability p_edge_flip
        mask = rng.random(new_adj.shape) < p_edge_flip
        new_adj[mask] ^= 1  # XOR toggles 0↔1
        
        # --- Occasional topology swap (structural exploration) ---
        if rng.random() < p_topo_swap and self.n_nodes >= 2:
            # Choose two random adjacent positions in topo order
            pos1 = rng.integers(0, self.n_nodes - 1)
            pos2 = pos1 + 1  # Swap adjacent positions
            
            # Swapping may violate edge constraints—recompute adj_lower
            temp_order = new_order.copy()
            temp_order[pos1], temp_order[pos2] = temp_order[pos2], temp_order[pos1]
            
            # Reconstruct valid lower-triangle under new order
            rebuild_mask = rng.random(self.n_nodes, self.n_nodes) < 0.5
            new_order = temp_order
            new_adj = rebuild_mask.astype(np.float64) & (~np.eye(self.n_nodes, dtype=bool))
            new_adj = np.tril(new_adj, k=-1)  # Keep only lower triangle
        
        return DagReplicator(
            n_nodes=self.n_nodes,
            topo_order=new_order,
            adj_lower=new_adj,
            data=self.data  # Shared reference OK (assumed immutable)
        )
    
    @classmethod
    def crossover(
        cls,
        parent_a: 'DagReplicator',
        parent_b: 'DagReplicator',
        rng: np.random.Generator,
        edge_inherit_rate: float = 0.7,
        order_blend_weight: float = 0.5,
    ):
        """Merge structural information from two parent DAGs.
        
        Uses position-based ordering voting and probabilistic edge inheritance.
        Offspring is GUARANTEED acyclic because edges respect final topology.
        """
        assert type(parent_a) == type(parent_b), "Parents must be same type"
        assert parent_a.n_nodes == parent_b.n_nodes, "Must have same node count"
        
        n = parent_a.n_nodes
        
        # === STEP 1: Hybridize topology orders ===
        # For each node, get its position ranking in both parents
        order_votes = []
        for node in range(n):
            pos_a = list(parent_a.topo_order).index(node)
            pos_b = list(parent_b.topo_order).index(node)
            
            # Higher rank = later position = more descendants
            rank_a = (n - 1 - pos_a) * (order_blend_weight if rng.random() < 0.5 else 1 - order_blend_weight)
            rank_b = (n - 1 - pos_b) * ((1 - order_blend_weight) if rng.random() < 0.5 else order_blend_weight)
            
            order_votes.append((node, rank_a + rank_b))
        
        # Sort by composite rank (descending = later in topological sequence)
        order_votes.sort(key=lambda x: x[1], reverse=True)
        child_order = np.array([v[0] for v in order_votes])
        
        # === STEP 2: Merge edge sets respecting child topology ===
        # Need to map parent adjacency to child ordering first
        child_adj = np.zeros((n, n))
        
        for j in range(n):
            node_j = child_order[j]
            for i in range(j):  # Only lower triangle positions
                node_i = child_order[i]
                
                # Query parent edge existence
                has_edge_a = parent_a._edge_exists(node_i, node_j)
                has_edge_b = parent_b._edge_exists(node_i, node_j)
                
                # Probability-weighted inheritance
                prob_from_a = (1.0 if has_edge_a else 0.0) * edge_inherit_rate
                prob_from_b = (1.0 if has_edge_b else 0.0) * (1.0 - edge_inherit_rate)
                
                inherited = rng.random() < (prob_from_a + prob_from_b)
                child_adj[j, i] = 1.0 if inherited else 0.0
        
        return cls(
            n_nodes=n,
            topo_order=child_order,
            adj_lower=child_adj,
            data=parent_a.data
        )
    
    def _edge_exists(self, source: int, target: int) -> bool:
        """Check if directed edge source → target exists in this DAG."""
        adj = self.express()
        return adj[target, source] > 0.5  # Threshold for float comparisons
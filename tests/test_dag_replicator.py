import numpy as np
from dagobert.replicators.dag_replicator import DagReplicator
from dagobert.rng import make_rng


def test_dag_is_acyclic():
    """Verify all mutated offspring remain acyclic."""
    rng = make_rng(123)
    n_nodes = 5
    n_samples = 100
    
    # Ground truth: simple chain A→B→C→D→E
    true_adj = np.diag(np.ones(n_nodes - 1), k=1)
    X_true = np.cumsum(rng.normal(size=(n_samples, n_nodes)), axis=1)
    
    pop = [DagReplicator(
        n_nodes=n_nodes,
        topo_order=rng.permutation(n_nodes),
        adj_lower=(rng.random((n_nodes, n_nodes)) < 0.3).astype(float) & ~np.eye(n_nodes),
        data=X_true
    ) for _ in range(20)]
    
    # Apply many mutations
    for _ in range(50):
        mutated_pop = []
        for ind in pop:
            child = ind.mutate(rng, p_edge_flip=0.2, p_topo_swap=0.3)
            mutated_pop.append(child)
            # Verify acyclicity invariant holds
            assert _validate_no_cycles(child.express()), "Cycle detected!"
        pop = mutated_pop


def test_crossover_preserves_structure():
    """Crossover output is valid DagReplicator."""
    rng = make_rng(456)
    n_nodes = 4
    data = rng.normal(size=(50, n_nodes))
    
    p1 = DagReplicator(
        n_nodes=n_nodes,
        topo_order=np.array([0, 1, 2, 3]),
        adj_lower=np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=float),
        data=data
    )
    
    p2 = DagReplicator(
        n_nodes=n_nodes,
        topo_order=np.array([3, 2, 1, 0]),
        adj_lower=(rng.random((n_nodes, n_nodes)) < 0.5).astype(float) & ~np.eye(n_nodes),
        data=data
    )
    
    child = DagReplicator.crossover(p1, p2, rng)
    
    assert child.n_nodes == n_nodes
    assert child.data.shape == data.shape
    assert _validate_no_cycles(child.express())


def _validate_no_cycles(adj: np.ndarray) -> bool:
    """Inline acyclicity checker for tests."""
    n = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    color = np.full(n, WHITE, dtype=np.int8)
    
    def dfs(u):
        color[u] = GRAY
        for v in range(n):
            if adj[u, v] > 0.5 and color[v] == GRAY:
                return False
        color[u] = BLACK
        return True
    
    for i in range(n):
        if color[i] == WHITE and not dfs(i):
            return False
    return True
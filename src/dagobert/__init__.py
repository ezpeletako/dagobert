"""DAGObert: Evolutionary DAG discovery framework."""

from .population import evolve
from .rng import make_rng
from .evaluators import default_evaluate, CachedEvaluator
from .selectors import tournament_selection, roulette_selection
from .variators import mutate_only, crossover_or_mutate
from .replicators.dag_replicator import DagReplicator


__version__ = "0.1.0"
__all__ = [
    'evolve', 'make_rng', 'default_evaluate',
    'tournament_selection', 'roulette_selection',
    'mutate_only', 'crossover_or_mutate', 'DagReplicator',
]
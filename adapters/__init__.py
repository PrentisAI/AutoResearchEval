"""Source adapters: each lowers one source into the unified IR (CLAUDE.md §8).

Submodules are imported lazily by callers (``from adapters import custodian_logs``)
so that adapters requiring heavy optional deps (aiida_walker, atomate2_taskdoc,
mp_api_tasks, mlflow_wandb) do not break ``import adapters`` when those extras
are absent. Each adapter that needs an external package guards the import and
anchors its version in a module header (§1.7, §11).
"""

__all__ = [
    "aiida_walker",
    "atomate2_taskdoc",
    "custodian_logs",
    "mp_api_tasks",
    "mlflow_wandb",
]

"""Source adapters: each lowers one source into the unified IR.

  openalex      — OpenAlex metadata + automatic bronze/silver/golden paper tiering
  paper_corpus  — a MinerU-parsed PDF corpus, read as section-sliced text

Submodules are imported on demand (``from adapters.openalex import OpenAlexClient``)
so an adapter needing a heavy optional dependency cannot break ``import adapters``.
"""

__all__ = ["openalex", "paper_corpus"]

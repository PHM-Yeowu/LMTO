import os

DATA = os.environ.get("LMTO_DATA", "data")                          # OpenTO and OpenTO-NL as downloaded
DERIVED = os.environ.get("LMTO_DERIVED", f"{DATA}/derived")         # scripts/prepare_data.py writes here
RESULTS = os.environ.get("LMTO_RESULTS", "results")
PYEDGE = os.environ.get("LMTO_PYEDGE", "OptimizeAnyTopology")       # clone of the OAT repository, which ships pyEDGE

# Generator epochs 1 to 4 over all training problems, each from the weights of the one before with a new order of the
# problems and a new cosine schedule; the corrector starts from the fourth epoch and makes one pass over one of its
# drafts per training problem
CONFIGS = {f"generator-e{e}": {"model": "Qwen/Qwen2.5-0.5B", "seed": e - 1, "lr": 2e-4 if e == 1 else 1e-4,
                               "lr_new": 1e-3, "budget": 24000, "corrupt": 0.3,
                               "init": f"generator-e{e - 1}" if e > 1 else None} for e in (1, 2, 3, 4)}
CONFIGS["corrector"] = {"model": "Qwen/Qwen2.5-0.5B", "corrector": True, "seed": 0, "lr": 1e-4, "lr_new": 1e-3,
                        "budget": 36000, "init": "generator-e4", "gray_weight": 1.0, "vol_weight": 200.0,
                        "workers": 32}

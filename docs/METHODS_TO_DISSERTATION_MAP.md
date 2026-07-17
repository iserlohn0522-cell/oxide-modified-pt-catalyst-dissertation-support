# Methods-to-Dissertation Map

| Dissertation area | Public support |
|---|---|
| Chapter 4 selected endpoint-energy comparison | `data/ch4_dft/cases/`, `selected_endpoint_energies.csv`, `selected_pair_differences.csv` |
| Chapter 4 protocol distinctions | `data/ch4_dft/protocols.csv`, `docs/ch4_dft/README.md` |
| Figures 4.4 and 4.6 numerical reconstruction | `data/ch4_dft/figure4_*.csv`, `scripts/ch4_dft_figures/` |
| Chapter 5 sliding-window detector preparation | `src/pt_tem_cv/dataset.py`, `tiling.py`, `augment.py` |
| Detector screening/training/evaluation | `src/pt_tem_cv/screening.py`, `training.py`, `evaluation.py` |
| Detector-box-prompted mask interface | `docs/ch5_ml_tem/README.md`, synthetic example schema; SAM 3 checkpoint excluded |
| Substrate footprint consensus | `src/pt_tem_cv/substrate.py`, production consensus config, three release assets |
| Human-reviewed scale conversion | `src/pt_tem_cv/scale.py`, scale verification schema |
| Projected particle descriptors | `src/pt_tem_cv/morphometry.py` |
| Final aggregate Chapter 5 results | `data/ch5_ml_tem/aggregate_results.csv` |

The map identifies public inspection points; it does not imply that private raw data are part of the repository.

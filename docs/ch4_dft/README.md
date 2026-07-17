# Chapter 4 DFT figure support

This directory documents the plot-level values used to reproduce dissertation Figures 4.4 and 4.6. The public files are intentionally limited to the reported constrained-separation energy differences and the method/evidence labels needed to interpret those figures.

## Included files

- `data/ch4_dft/figure4_4_constrained_separation.csv`: the selected Zr-JC/Zr-JC*/Nb-WP comparison in Figure 4.4.
- `data/ch4_dft/figure4_6_extension.csv`: the oxide-fragment extension values in Figure 4.6.
- `scripts/ch4_dft_figures/render_figure4_4_constrained_separation.py`
- `scripts/ch4_dft_figures/render_figure4_6_extension.py`

In both data files, `delta_e_sep_2A_eV` is the within-pair difference `E(displaced) - E(retained)` after the prescribed 2 Å displacement and support relaxation. `Zr-JC` is the primary Zr junction-contact endpoint, `Zr-JC*` is the second lower-energy displaced basin from the same retained endpoint and displacement, and `Nb-WP` is the Nb weak-contact comparator.

The Zr, Nb, Ti, and W rows use the fixed-geometry PBE-D3(BJ), C9-off comparison family. `TaOx*` was obtained under a different relaxation protocol and is included only as a qualitative extension. CeOx was not evaluated quantitatively because it requires an explicit DFT+U and charge-state protocol.

Full coordinate and calculation archives are outside this public figure-support package.

## Reproduce the figures

From the repository root:

```text
python scripts/ch4_dft_figures/render_figure4_4_constrained_separation.py --output-dir path/to/output
python scripts/ch4_dft_figures/render_figure4_6_extension.py --output-dir path/to/output
```

If `--output-dir` is omitted, the scripts write to a temporary-system directory and print the resulting paths.

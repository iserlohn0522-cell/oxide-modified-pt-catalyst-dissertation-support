# Chapter 4 DFT Support

This directory documents the selected endpoint calculations and plot-level values used in dissertation Chapter 4. It is a curated calculation subset, not a complete cluster archive.

## Included files

- `data/ch4_dft/figure4_4_constrained_separation.csv`: selected Zr-JC/Zr-JC*/Nb-WP values used in Figure 4.4.
- `data/ch4_dft/figure4_6_extension.csv`: oxide-fragment extension values used in Figure 4.6.
- `data/ch4_dft/selected_endpoint_energies.csv`: accepted endpoint energies underlying the reported differences.
- `data/ch4_dft/selected_pair_differences.csv`: pair definitions and `E(displaced) - E(retained)` values.
- `data/ch4_dft/protocols.csv`: numerical settings that distinguish the principal, control, and periodic-extension protocols.
- `data/ch4_dft/cases/`: curated five-case packages with accepted retained/displaced inputs, coordinates, declared output excerpts, results, and source/public hashes.
- `scripts/ch4_dft_figures/`: the two public plotting scripts and a small CP2K final-energy extractor.

In the figure tables, `delta_e_sep_2A_eV` is the within-pair difference `E(displaced) - E(retained)` after the prescribed 2 Å displacement and support relaxation. `Zr-JC` is the primary Zr junction-contact endpoint, `Zr-JC*` is the second lower-energy displaced basin from the same retained endpoint and displacement, and `Nb-WP` is the Nb weak-contact comparator.

## Protocol boundaries

Zr-JC, Zr-JC*, Nb-WP, and TiOx use the 600/50 Ry, NGRIDS 5, nonperiodic fixed-geometry PBE-D3(BJ), C9-off protocol. The Zr-OS and Nb-JC controls instead use a matched 400/50 Ry, NGRIDS 4, periodic protocol and are interpreted against each other. The WOx extension retains a periodic cell at 600/50 Ry and is not protocol-identical to the nonperiodic principal comparison. `TaOx*` was obtained under a different relaxation protocol and is included only as a qualitative extension. CeOx was not evaluated quantitatively because it requires an explicit DFT+U and charge-state protocol.

The 400 Ry periodic controls and the 600 Ry nonperiodic principal calculations differ in more than the plane-wave cutoff: `NGRIDS`, boundary conditions, electrostatics, cell construction, and SCF settings also differ. Their raw absolute energies must not be compared across protocol families. Only within-pair energy differences from a common protocol are reported, and the two control cases form their own matched comparison.

The selected input coordinates allow the released fixed-geometry calculations to be inspected or rerun with CP2K 2023.1. Scheduler files, wavefunctions, projected-density files, full outputs, failed trials, and complete calculation archives remain outside this public package. `extract_cp2k_output_excerpt.py` documents how each path-free evidence excerpt was made, and `extract_cp2k_final_energy.py` documents how final energies were read. The accepted numerical results are supplied directly in the CSV tables and endpoint `result.json` files.

## Reproduce the figures

From the repository root:

```text
python scripts/ch4_dft_figures/render_figure4_4_constrained_separation.py --output-dir path/to/output
python scripts/ch4_dft_figures/render_figure4_6_extension.py --output-dir path/to/output
```

If `--output-dir` is omitted, the scripts write to a system temporary directory and print the resulting paths.

# DFT Protocol Classes

`M1_600Ry_nonperiodic` contains the principal Zr-JC, Zr-JC*, Nb-WP, and TiOx fixed-geometry PBE-D3(BJ) endpoint energies at 600/50 Ry, NGRIDS 5, and nonperiodic Martyna–Tuckerman electrostatics.

`M2_400Ry_periodic` contains only the Zr-OS and Nb-JC matched controls at 400/50 Ry, NGRIDS 4, and periodic electrostatics. The retained geometries originated from earlier closed-shell unconstrained optimizations, whereas the reported retained energies are later LSD fixed-geometry single points. Only the displaced-endpoint optimizations fixed the Pt6 atoms.

`M3_600Ry_periodic` contains the WOx periodic-boundary repair endpoint energies at 600/50 Ry and NGRIDS 5. It is retained as a periodic extension and is not relabeled as nonperiodic-protocol matched.

`M4_method_distinct` identifies the qualitative TaOx* value from a different relaxation protocol. CeOx remains unquantified.

The tabular settings are in `data/ch4_dft/protocols.csv`. Absolute energies are never combined across protocol classes.

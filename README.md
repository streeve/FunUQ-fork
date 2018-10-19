# FunUQ
## Functional Uncertainty Quantification for Molecular Dynamics with Python 
## Sam Reeve

This code enables FunUQ for MD with LAMMPS MD. More examples are under current investigation. If you use this code, please cite the following: 

Reeve, S. T. & Strachan, A. Error correction in multi-fidelity molecular dynamics simulations using functional uncertainty quantification. J. Comput. Phys. 334, 207–220 (2017).

Reeve, S. T. & Strachan, Quantifying uncertainties originating from interatomic potentials in molecular dynamics (Submitted MSMSE 2018).



Try an example through nanoHUB: https://nanohub.org/resources/funuq
The notebooks included in "examples/" can be used to run new simulations, calculate functional derivatives, make corrections, and plot results (without download or install on nanoHUB)

Examples currently included:
 * Correction between Lennard-Jones and sine-modifed LJ for energy and pressure (NVT)
 * Correction between Morse and exponential-6 for energy and pressure (NVT)
 * Correction between Morse and	exponential-6 for energy and volume (NPT)
 * Comparison of bruteforce and perturbative (all-atom or RDF) functional derivatives


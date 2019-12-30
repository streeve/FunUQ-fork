# FunUQ
## Functional Uncertainty Quantification with Python 
## Sam Reeve

This code enables FunUQ for MD with LAMMPS. More examples are under current investigation. If you use this code, please cite the FunUQ literature:

Reeve, S. T. & Strachan, A. Functional uncertainty quantification for isobaric molecular dynamics simulations and defect formation energies. Modelling Simul. Mater. Sci. Eng. 27, 044002 (2019). https://doi.org/10.1088/1361-651X/ab16fa

Reeve, S. T. & Strachan, A. Error correction in multi-fidelity molecular dynamics simulations using functional uncertainty quantification. J. Comput. Phys. 334, 207–220 (2017). https://doi.org/10.1016/j.jcp.2016.12.039

Strachan, A., Mahadevan, S., Hombal, V. & Sun, L. Functional derivatives for uncertainty quantification and error estimation and reduction via optimal high-fidelity simulations. Modelling Simul. Mater. Sci. Eng. 21, 065009 (2013). https://doi.org/10.1088/0965-0393/21/6/065009


Try an example through nanoHUB: https://nanohub.org/resources/funuq


Examples currently included:
 * One case from Reeve & Strachan J. Comput. Phys. 2017: correction between Lennard-Jones and a sine-modified LJ ("Sine 1")
 * Cases from Reeve & Strachan MSMSE 2019: correction between Morse and exponential-6
     * NVT (canonical ensemble)
     * NPT (isothermal-isobaric ensemble)


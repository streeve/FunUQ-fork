
### FunUQ run file ###
### v0.3; Sam Reeve

from funuq import *

morse_dict = {
    'units': ['eV/atom', 'GPa'],
    'description': 'Morse liquid (similar case to JCompPhys paper, 2017)',
    'correctpot_name': 'buck'
}

# Initialize class
# Input potential function, list of quantities of interest, 
#     template/data directory, run directory, optional input dict (above)
morse = FunUQ_perturb_coord('morse', ['PotEng', 'Press'],
                            'morse_liquid/', '/scratch/halstead/s/sreeve/funuq/morse_liquid/',
                            input_dict=morse_dict)

# Run simulations
#morse.run_lammps() 

# Calculate functional derivative
#morse.rerun_gauss()
morse.prepare_FD()
morse.calc_FD()
for x in range(2):
    morse.write_FD(x)
    morse.plot_FD(x)

# Calculate discrepancy for exponential-6 (Buckingham)
#morse.discrep('exp6')

# Get corrections for properties
#morse.correct()


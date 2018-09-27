
### FunUQ run file ###
### v0.4; Sam Reeve

#from funuq import *
from FunUQ import *
import numpy as np

# Run new simulations? 
run = False
run_verify = False
run_perturb = False
run_bruteforce = False

# Define directories
startdir = 'morse_fluct/'
rundir = '/scratch/halstead/s/sreeve/FunUQ/active/morse_liquid/'

# Define/create potentials
Pot_main = Potential('morse', paramdir=startdir, create=True, N=8000, rhi=8.0, cut=6.)
Pot_correct = Potential('exp6', paramdir=startdir, create=True, N=8000, rhi=8.0, cut=6.)

# Define QoI
QoI_list = ['PotEng', 'Press', 'Volume', 'HeatCapacityVol'] #, 'Compressibility', 'ThermalExpansion']
Nqoi = len(QoI_list)
QoI_dict = {'description': 'Test liquid',
            'Ncopies': 2,
            'units': ['eV/atom', 'GPa', 'nm**3', 'eV/atom/K'] #, '1/GPa', '1/K'],
            #'overwrite': True
}

QoI = QuantitiesOfInterest(QoI_list, Pot_main, 
                           startdir, rundir, 'main', 'metal',
                           input_dict=QoI_dict)
QoI_correct = QuantitiesOfInterest(QoI_list, Pot_correct,
                                   startdir, rundir, 'exp6', 'metal',
                                   input_dict=QoI_dict)
# Run simulations (if needed)
if run:
    QoI.run_lammps()
if run_verify:
    QoI_correct.run_lammps()
if run or run_verify:
    exit()

QoI.extract_lammps()
QoI_correct.extract_lammps()

# Define functional derivative inputs
FD_dict = {'alist': [-1e-6, -2e-6, 1e-6, 2e-6], #coord
           #'alist': [-1e-12, -2e-12, 1e-12, 2e-12],#allatom

           # Bruteforce
           #'alist': [-0.02, -0.01, 0.01, 0.02],
           #'rmin': 1.5,
           #'rlist': np.linspace(1.5, 6., 30),
           }
#FuncDer = FuncDer_bruteforce(QoI, Pot_main,
#                             input_dict=FD_dict)
#FuncDer = FuncDer_perturb_allatom(QoI, Pot_main,
#                                   input_dict=FD_dict)
FuncDer = FuncDer_perturb_coord(QoI, Pot_main, ensemble='nvt',
                                input_dict=FD_dict)


# Run simulations for functional derivative (if needed)
if run_bruteforce:
    FuncDer.run_lammps()
    exit() 
if run_perturb and FuncDer.method == 'perturbative_allatom':
    FuncDer.rerun_gauss()
    exit()

# Calculate functional derivative
FuncDer.prepare_FD()
FuncDer.calc_FD()
for x in range(Nqoi-1,Nqoi):
    FuncDer.write_FD(x)
    FuncDer.plot_FD(x)
    

# Prepare correction
Correct = FunUQ(Pot_main, Pot_correct, QoI.Q_names, QoI.Qavg, QoI_correct.Qavg, Q_units=QoI.units, FD=FuncDer.funcder, R=FuncDer.rlist)

# Read FD instead of recomputing
#Correct = FunUQ(Pot_main, Pot_correct, QoI.Q_names, QoI.Qavg, QoI_correct.Qavg)
#Correct.read_FD(QoI.resultsdir, FuncDer.FDnames)

# Calculate functional discrepancy
Correct.discrepancy()
#Correct.plot_discrep()

# Get corrections for properties
Correct.correct()
#for x in range(Nqoi):
#    Correct.plot_funcerr(x)
#    Correct.plot_correction(x)


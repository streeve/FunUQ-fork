
### FunUQ run file ###
### v0.4; Sam Reeve

#from funuq import *
from FunUQ_MD import *


# Run new simulations? 
run = False
run_verify = False
run_perturb = False
run_bruteforce = False
################################################################### Scale QoI values?

# Main inputs
startdir = 'morse_npt_fluct/'
rundir = '/scratch/halstead/s/sreeve/funuq/npt_morse_liquid/'

# Potentials
Pot_main = Potential('morse', paramdir=startdir)
Pot_correct = Potential('exp6', paramdir=startdir)

# QoI
QoI_list = ['PotEng', 'Press', 'Volume', 'HeatCapacityPress', 'Compressibility', 'ThermalExpansion']
QoI_dict = {'description': 'Test liquid',
            'Ncopies': 2,
            'units': ['eV/atom', 'GPa', 'nm**3', 'eV/atom/K', '1/GPa', '1/K'],
            #'overwrite': True
}

QoI = QuantitiesOfInterest(QoI_list, Pot_main, 
                           startdir, rundir, 'main',
                           input_dict=QoI_dict)
QoI_correct = QuantitiesOfInterest(QoI_list, Pot_correct,
                                   startdir, rundir, 'exp6',
                                   input_dict=QoI_dict)

if run:
    QoI.run_lammps()
if run_verify:
    QoI_correct.run_lammps()
if run or run_verify:
    exit()

QoI.extract_lammps()
QoI_correct.extract_lammps()

print QoI.Qavg, QoI_correct.Qavg

FD_dict = {'alist': [-1e-12, -2e-12, 1e-12, 2e-12],
#'rN': 60
           #'rlist': np.linspace(0.0998333333333333, 6.09983333333333334, 30)
           
           # Bruteforce
#'rmin': 1.5,
#'rlist': np.linspace(1.5, 6., 30),
           }
#FuncDer = FuncDer_bruteforce(QoI, Pot_main,
#                             input_dict=FD_dict)
#FuncDer = FuncDer_perturb_allatom(QoI, Pot_main,
#                                   input_dict=FD_dict)
FuncDer = FuncDer_perturb_coord(QoI, Pot_main, ensemble='npt',
                                input_dict=FD_dict)


# Calculate functional derivative
if run_bruteforce:
    FuncDer.run_lammps()
    exit() 

if run_perturb and FuncDer.method == 'perturb_allatom':
    FuncDer.rerun_gauss()
    exit()

FuncDer.prepare_FD()
FuncDer.calc_FD()
for x in range(len(QoI_list)):
    FuncDer.write_FD(x)
    FuncDer.plot_FD(x)


Correct = FunUQ(Pot_main, Pot_correct, QoI.Q_names, QoI.Qavg, QoI_correct.Qavg, FD=FuncDer.funcder, R=FuncDer.rlist)
#Correct = FunUQ(Pot_main, Pot_correct, QoI.Q_names, QoI.Qavg, QoI_correct.Qavg)

# Get corrections for properties
#Correct.read_FD(QoI.resultsdir, FuncDer.FDnames)

Correct.discrepancy()

# Get corrections for properties
Correct.correct()
#Correct.plot_correct()

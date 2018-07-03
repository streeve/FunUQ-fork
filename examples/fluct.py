
### FunUQ run file ###
### v0.4; Sam Reeve

#from funuq import *
from FunUQ_MD import *


# Run new simulations? 
run = False
run_verify = False
run_perturb = False
################################################################### Scale QoI values?

# Main inputs
startdir = 'morse_fluct_shifted/'
rundir = '/scratch/halstead/s/sreeve/funuq/morse_liquid/'

# Define potentials
Pot_main = Potential('morse', paramdir=startdir)
Pot_correct = Potential('exp6', paramdir=startdir)

# Calculate QoI
QoI_list = ['PotEng', 'Press', 'HeatCapacityVol']
QoI_dict = {'description': 'Test liquid',
            'Ncopies': 2,
            'units': ['eV/atom', 'GPa', 'eV/atom/K'],
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

FD_dict = {'alist': [-1e-8, -2e-8, 1e-8, 2e-8],
           #'rlist': np.linspace(0.0998333333333333, 6.09983333333333334, 30)

           #'rmin': 1.5,
           #'rlist': np.linspace(1.5, 6., 30),
           }
#FuncDer = FuncDer_bruteforce(QoI, Pot_main,
#                             input_dict=FD_dict)
#FuncDer = FuncDer_perturb_allatom(QoI, Pot_main,
#                                   input_dict=FD_dict)
FuncDer = FuncDer_perturb_coord(QoI, Pot_main,
                                 input_dict=FD_dict)


### Calculate functional derivative
if run and FuncDer.method == 'bruteforce':
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


### Calculate correction

Correct = FunUQ(Pot_main, Pot_correct, QoI.Q_names, QoI.Qavg, QoI_correct.Qavg, FD=FuncDer.funcder, R=FuncDer.rlist)

# Don't recalculate FD
#Correct = FunUQ(Pot_main, Pot_correct, QoI.Q_names, QoI.Qavg, QoI_correct.Qavg)
#Correct.read_FD(QoI.resultsdir, FuncDer.FDnames)

Correct.discrepancy()
Correct.plot_discrep() 

# Get corrections for properties
Correct.correct()
#Correct.plot_correct()

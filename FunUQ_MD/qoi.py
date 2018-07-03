# FunUQ v0.4, 2018; Sam Reeve; Strachan Research Group
# https://github.rcac.purdue.edu/StrachanGroup

# import general
import sys, os, subprocess, shutil, numpy as np
from random import random; from glob import glob
from matplotlib import pyplot as plt

# import local functions
from utils import *
from parsetools import *



class QuantitiesOfInterest(object): 
    '''
    Class for running LAMMPS and extracting thermodynamic results
    '''
    def __init__(self, Qlist, potential, initdir, maindir, purpose, input_dict=None):
        '''
        Defaults for base class; to be modified by user through dictionary as needed
        '''

        self.overwrite = False
        #self.run = False

        # Quantity of interest
        self.Q_names = list(Qlist)
        requiredQ = ['PotEng'] #, 'Press', 'Volume']
        for rQ in requiredQ:
            if rQ not in self.Q_names:
                self.Q_names += [rQ]

        # Distinguish direct thermodynamic properties,
        # fluctuation properties, and unrecognized properties
        self.Q_thermo = ['']*len(self.Q_names)
        for qc, q in enumerate(self.Q_names):
            thermo = is_thermo(q)
            self.Q_thermo[qc] = thermo
            if not thermo:
                fluct = is_fluct(q)
                if not fluct:
                    raise FUQexception("{} is not a supported quantity of interest."
                                       .format(q))


        self.pot = potential 

        self.units = []
        #self.get_units()

        # Files and directories
        self.Ncopies = 5
        self.intemplate = 'in.template'
        self.subtemplate = 'submit.template'
        self.infile = 'in.lammps'
        self.subfile = 'run.pbs'
        self.logfile = 'log.lammps'
        self.initdir = initdir
        self.maindir = maindir
        self.FDname = 'out.funcder'
        
        #self.parampath = os.path.join(initdir, self.paramfile)
        self.inpath = os.path.join(self.initdir, self.intemplate)
        self.subpath = os.path.join(self.initdir, self.subtemplate)

        # Other
        self.description = ''
        self.name = self.pot.potname
        self.rerun_folder = 'rerun_'
        self.copy_folder = 'copy_'
        self.copy_start = 0


        # User overrides defaults
        if input_dict != None:
            for key, val in input_dict.iteritems():
                try: 
                    setattr(self, key, val)
                except:
                    raise KeyError("{} is not a valid input parameter.".format(key))
        
        # Get user templates
        self.intxt = read_template(self.inpath)
        self.subtxt = read_template(self.subpath)

        # Create/find simulation directories
        self.resultsdir = os.path.join(self.maindir, 'results/')
        self.rundir = os.path.join(self.maindir, '{}_runs/'.format(purpose))

        for d in [self.maindir, self.rundir, self.resultsdir]:
            try:
                os.mkdir(d)
            except: pass

        # Create potential table
        #if self.createpot:
        self.pot.create(self.rundir)


    def get_units(self):
        for q in self.Q_names:
            if q == 'PotEng':
                self.units.append('eV/atom')
            elif q == 'Press':
                self.units.append('GPa')
            else:
                raise FUQexception("{} is not a supported quantity of interest."
                                   "The class needs to be updated: see 'PotEng' as an example."
                                   .format(q))


    def run_lammps(self):
        '''
        Run unmodified potential (multiple copies)
        Main perturbative or verification second potential

        Args: potential class object
        '''
        if self.pot == None:
            replace_in = {'SEED':'0'}
            rundir = self.rundir
        else:
            replace_in = {'TABLECOEFF':self.pot.paircoeff, 'TABLESTYLE':self.pot.pairstyle, 'SEED':'0'}
            replace_sub = {'NAME': self.name, 'INFILE': self.infile}
            rundir = self.pot.potdir
        copy_files(self.initdir, rundir)

        maxcopy = 0
        if not self.overwrite:
            existing = glob(os.path.join(rundir, 'copy_*'))
            for x in existing:
                nextcopy = float(x.split(self.copy_folder)[-1])
                if nextcopy > maxcopy:
                    maxcopy = nextcopy 

        for copy in range(maxcopy, maxcopy+self.Ncopies):
            cdir = os.path.join(rundir, self.copy_folder+str(copy))
            replace_in['SEED'] = str(int(random()*100000))
            replace_sub['NAME'] = '{}_{}'.format(self.name, copy)
            intxt = replace_template(self.intxt, replace_in)
            subtxt = replace_template(self.subtxt, replace_sub)
            submit_lammps(self.subfile, subtxt,
                          self.infile, intxt, cdir)


    def extract_lammps(self, log='log.lammps'):
        '''
        Read lammps log thermo data; location from potential class input
        Used for base properties and perturbative calculations
        '''
        for copy0, copy in enumerate(range(self.copy_start, self.copy_start + self.Ncopies)):
            logfile = os.path.join(self.rundir, self.copy_folder+str(copy), log) 
            cols, thermo, Natoms = read_thermo(logfile)

            # Only extract these values once
            if not copy0:
                self.Natoms = Natoms
                self.Ntimes = np.shape(thermo)[0]
                #self.times = np.array(sample(range(startsteps, totalsteps), self.Ntimes))

                #self.Q = np.zeros([self.Ntimes, self.Ncopies, len(self.Q_names)])
                self.Q = np.zeros([self.Ntimes, self.Ncopies, 1,1,1, len(self.Q_names)])
                self.Qavg = np.zeros([len(self.Q_names)])
                self.Qstd = np.zeros([len(self.Q_names)])

                # Find names of thermo/fluctuation properties
                Q_thermonames = []
                #Q_fluctnames = []
                for q,qthermo in zip(self.Q_names, self.Q_thermo):
                    if qthermo:
                        Q_thermonames += [q]
                    #else: self.Q_fluctnames += [q]

                # Get columns for thermo properties
                Q_cols = find_columns(cols, Q_thermonames) #self.Q_names)
                self.Q_cols = ['X']*len(self.Q_names)
                for qc, q in enumerate(self.Q_names):
                    if q in Q_thermonames:
                        self.Q_cols[qc] = Q_cols[Q_thermonames.index(q)]

                # Get other necessary properties
                self.V = thermo[0, find_columns(cols, ['Volume'])[0]]
                self.T = thermo[0, find_columns(cols, ['Temp'])[0]]
                self.beta = 1./8.617e-5/self.T


            #for qc, (q,thermoq) in enumerate(zip(self.Q_cols, self.Q_thermo)):
            for qc, (q,qn) in enumerate(zip(self.Q_cols, self.Q_names)):
                if qn == 'PotEng':
                    self.PEcol = qc
                if qn == 'Press':
                    self.Pcol = qc
                if qn == 'Volume':
                    self.Vcol = qc

                    
                if self.Q_thermo[qc]:
                    self.Q[:, copy0, 0,0,0, qc] = thermo[:,q]
                    #self.Q[:,copy0,qc] = thermo[:,q]
                elif qn == 'HeatCapacityVol':
                    self.Q[:, copy0, 0,0,0, qc] = thermo[:,self.Q_cols[self.PEcol]]**2
                elif qn == 'HeatCapacityPress':
                    self.Q[:, copy0, 0,0,0, qc] = (thermo[:,self.Q_cols[self.PEcol]]
                                                   + (thermo[:,self.Q_cols[self.Pcol]]*
                                                      thermo[:,self.Q_cols[self.Vcol]]))**2
                elif qn == 'Compressibility':
                    self.Q[:, copy0, 0,0,0, qc] = thermo[:,self.Q_cols[self.Vcol]]**2
                elif qn == 'ThermalExpansion':
                    self.Q[:, copy0, 0,0,0, qc] = (thermo[:,self.Q_cols[self.Vcol]]**2)
                                                   #(thermo[:,self.Q_cols[self.PEcol]]
                                                   # + (thermo[:,self.Q_cols[self.Pcol]]*
                                                   #    thermo[:,self.Q_cols[self.Vcol]])))

                # TODO: if there are issues, fill with NaN to ignore
                # Works okay while jobs running IF the first copy stays ahead
        

        self.Qavg = np.mean(self.Q, axis=(0,1,2,3,4))
        self.Qstd = np.std(self.Q, axis=(0,1,2,3,4))

        self.get_conversions()
        for qc, q in enumerate(self.Q_names):
            '''
            if self.Q_thermo[qc]:
                Q = self.Q[:,:,0,0,0,qc]
                self.Qavg[qc] = np.mean(Q)
                self.Qstd[qc] = np.std(Q)
            '''
            if q == 'HeatCapacityVol':
                self.Qavg[qc] = fluctuation(self.beta/self.T, self.Qavg[qc], self.Qavg[self.PEcol])
                #self.beta/(self.T)*(self.Qavg[qc] - self.Qavg[self.PEcol]**2)
            elif q == 'HeatCapacityPress':
                self.Qavg[qc] = fluctuation(self.beta/self.T, self.Qavg[qc], self.Qavg[self.PEcol] + self.Qavg[self.Pcol]*self.Qavg[self.Vcol])
            elif q == 'Compressibility':
                self.Qavg[qc] = fluctuation(self.beta/self.V, self.Qavg[qc], self.Qavg[self.Vcol])
            elif q == 'ThermalExpansion':
                self.Qavg[qc] = fluctuation(self.beta/self.T/self.V, self.Qavg[qc], self.Qavg[self.PEcol] + self.Qavg[self.Pcol]*self.Qavg[self.Vcol])

            self.Qavg[qc] = self.Qavg[qc]*self.conversions[qc]


    def get_conversions(self):
        self.conversions = [1.]*len(self.Q_names)

        for qc, q in enumerate(self.Q_names):
            if q == 'PotEng':
                self.conversions[qc] = 1./self.Natoms
            elif q == 'Press':
                self.conversions[qc] = 0.0001
            elif q == 'Volume':
                self.conversions[qc] = 0.001
            elif 'HeatCapacity' in q:
                self.conversions[qc] = 1./self.Natoms #*8.617e-5
            elif q == 'Compressibility':
                self.conversions[qc] = 1e4
            elif q == 'ThermalExpansion':
                self.conversions[qc] = 1.

    def write_something(self, qc):
        txt = ("FunUQ v0.4, 2018; Sam Reeve\n"
               "Functional derivative for {} in {} with {} at {}K with {} atoms; "
               "Description: {}\n"
               .format(self.Q_names[qc], self.units[qc], self.pot.potname, self.T, self.Natoms, self.description))
        for r, fd in zip(self.rlist, self.funcder[:,qc]):
            txt += "{} {}\n".format(r, fd)

        name = '{}_{}'.format(self.Q_names[qc], self.FDname)
        with open(os.path.join(self.resultsdir, name), 'w') as f:
            f.write(txt)


    def plot_something(self, qc):
        plt.figure()
        plt.plot(self.rlist, self.funcder[:,qc], color='navy')
        plt.scatter(self.rlist, self.funcder[:,qc], color='navy')
        plt.xlabel("Position ($\AA$)", fontsize='large')
        plt.xlim([self.rmin, self.rmax-0.2])
        x_range = np.where((self.rlist > self.rmin) & (self.rlist < self.rmax))
        ymin = np.min(self.funcder[x_range,qc])
        ymin += -ymin*0.1
        ymax = np.max(self.funcder[x_range,qc])
        ymax += ymax*0.1
        plt.ylim([ymin, ymax])
        plt.ylabel("Functional Derivative ({}/$\AA$)".format(self.units[qc]), fontsize='large')
        plt.show()

    
def fluctuation(pre, square_avg, avg):

    return pre*(square_avg - avg**2)

# Other fluctuation properties

# FunUQ v0.4, 2018; Sam Reeve; Strachan Research Group
# https://github.rcac.purdue.edu/StrachanGroup

# import general
import sys, os, subprocess, shutil, numpy as np
from random import random; from glob import glob
from matplotlib import pyplot as plt
from copy import deepcopy

# import local functions
from .utils import is_thermo, is_fluct, copy_files, read_template, replace_template, submit_lammps, FUQerror
from .parsetools import read_thermo, find_columns



class QuantitiesOfInterest(object): 
    '''
    Class for running LAMMPS and extracting thermodynamic results
    '''
    def __init__(self, Qlist, potential, initdir, maindir, purpose, unittype, input_dict=None):
        '''
        Defaults for base class; to be modified by user through dictionary as needed
        '''

        self.overwrite = False
        #self.run = False

        # Quantity of interest
        self.Q_names = list(Qlist)
        requiredQ = ['PotEng', 'Press', 'Volume']
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
                    raise FUQerror("{} is not a supported quantity of interest."
                                   .format(q))


        self.pot = potential 

        self.units = []
        self.unittype = unittype
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

        self.create_pot = True

        # User overrides defaults
        if input_dict != None:
            for key, val in input_dict.items():
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

        # COPY potential table
        #if self.create_pot:
        #    self.pot.create(self.rundir)
        #else:
        #    self.pot.copy(self.initdir, self.rundir)
        self.pot.copy(self.initdir, self.rundir)


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
                nextcopy = int(x.split(self.copy_folder)[-1])
                if nextcopy > maxcopy:
                    maxcopy = nextcopy 
            if maxcopy > 0:
                maxcopy += 1
        elif self.copy_start > 0:
            maxcopy = self.copy_start
        # TODO can't start at fixed number and not overwrite

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
                for q,qthermo in zip(self.Q_names, self.Q_thermo):
                    if qthermo:
                        Q_thermonames += [q]

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
                    self.Q[:, copy0, 0,0,0, qc] = deepcopy(thermo[:,self.Q_cols[self.PEcol]])**2
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
        print (self.Qavg)


        for qc, q in enumerate(self.Q_names):

            if q == 'HeatCapacityVol':
                self.Qavg[qc] = fluctuation(self.beta/self.T, self.Qavg[qc], self.Qavg[self.PEcol])
            elif q == 'HeatCapacityPress':
                self.Qavg[qc] = fluctuation(self.beta/self.T, self.Qavg[qc], self.Qavg[self.PEcol] + self.Qavg[self.Pcol]*self.Qavg[self.Vcol])
            elif q == 'Compressibility':
                self.Qavg[qc] = fluctuation(self.beta/self.V, self.Qavg[qc], self.Qavg[self.Vcol])
            elif q == 'ThermalExpansion':
                self.Qavg[qc] = fluctuation(self.beta/self.T/self.V, self.Qavg[qc], self.Qavg[self.PEcol] + self.Qavg[self.Pcol]*self.Qavg[self.Vcol])

        # ONLY CONVERT after all fluctuations calculated
        self.get_conversions()
        for qc, q in enumerate(self.Q_names):
            self.Qavg[qc] = self.Qavg[qc]*self.conversions[qc]


    def get_conversions(self):
        self.conversions = [1.]*len(self.Q_names)

        for qc, q in enumerate(self.Q_names):
            if q == 'PotEng' or q == 'E_vdwl':
                if self.unittype == 'metal':
                    self.conversions[qc] = 1./self.Natoms
                elif self.unittype == 'real':
                    self.conversions[qc] = 1. #0.0433644/self.Natoms
            elif q == 'Press':
                if self.unittype == 'metal':
                    self.conversions[qc] = 0.0001
                elif self.unittype == 'real':
                    self.conversions[qc] = 0.000101325
            elif q == 'Volume':
                self.conversions[qc] = 0.001
            elif 'HeatCapacity' in q:
                self.conversions[qc] = 1./self.Natoms/8.617e-5
            elif q == 'Compressibility':
                self.conversions[qc] = 1e4
            elif q == 'ThermalExpansion':
                self.conversions[qc] = 1.

    
def fluctuation(pre, avg_ofthe_square, avg):

    return pre*(avg_ofthe_square - avg**2)


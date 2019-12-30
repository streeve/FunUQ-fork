# FunUQ v0.1, 2018; Sam Reeve; Strachan Research Group
# https://github.rcac.purdue.edu/StrachanGroup

# import general
import sys, os, subprocess, shutil, numpy as np
from random import random, sample; from glob import glob
from matplotlib import pyplot as plt
from copy import deepcopy

# import local functions
from .utils import is_thermo, is_fluct, copy_files, read_file, replace_template, submit_lammps, FUQerror
from .parsetools import read_thermo, find_columns



class QuantitiesOfInterest(object): 
    '''
    Class for running LAMMPS and extracting thermodynamic results
    '''
    def __init__(self, Qlist, Potential, maindir, init, run, 
                 **kwargs):
        '''
        Defaults for base class

        INPUTS:
        Required:
            Qlist - List of quantities of interest
            Potential - FunUQ potential object
            maindir - Full path to calculations
            init - Directory name for LAMMPS and potential files
            run - Directory name for LAMMPS simulations and FunUQ results

        Optional:
            description - String describing calculations
        '''

        self.overwrite = False

        # Quantity of interest
        self.Q_names = list(Qlist)
        self.ensemble = kwargs.get('ensemble', 'nvt')
        if self.ensemble == 'nvt':
            requiredQ = ['PotEng']
        elif self.ensemble == 'npt':
            requiredQ = ['PotEng', 'Volume']
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
                if fluct:
                    print("WARNING: fluctionation properties still in development")
                if not fluct:
                    raise FUQerror("{} is not a supported quantity of interest."
                                   .format(q))


        self.pot = Potential 

        self.units = kwargs.get('units', ['']*len(self.Q_names))
        self.unittype = kwargs.get('unittype', 'metal')
        self.get_units() # This is only for energy (from unittype)

        # Files and directories
        self.Ntimes = kwargs.get('Ntimes')
        self.Ncopies = kwargs.get('Ncopies', 1)
        self.sample = kwargs.get('sample', True)
        self.Nmax = None 

        self.init = init
        self.run = run
        self.maindir = maindir
        self.intemplate = kwargs.get('intemplate', 'in.template')
        self.subtemplate = kwargs.get('subtemplate', 'submit.template')
        self.infile = kwargs.get('infile', 'in.lammps')
        self.subfile = kwargs.get('subfile', 'run.pbs')
        self.logfile = kwargs.get('logfile', 'log.lammps')
        self.FDname = kwargs.get('FDname', 'out.funcder')
        
        #self.parampath = os.path.join(initdir, self.paramfile)
        self.initdir = os.path.join(self.maindir, self.init)
        self.inpath = os.path.join(self.initdir, self.intemplate)
        self.subpath = os.path.join(self.initdir, self.subtemplate)

        # Other
        self.description = kwargs.get('description', '')
        self.name = kwargs.get('name', self.pot.potname)
        self.rerun_folder = kwargs.get('rerun_folder', 'rerun_')
        self.copy_folder = kwargs.get('copy_folder', 'copy_')
        self.copy_start = kwargs.get('copy_start', 0)

        self.create_pot = kwargs.get('create_pot', True)

        # User overrides defaults
        '''
        if input_dict != None:
            for key, val in input_dict.items():
                try: 
                    setattr(self, key, val)
                except:
                    raise KeyError("{} is not a valid input parameter.".format(key))
        '''
        # Get user templates
        self.intxt = read_file(self.inpath)
        self.mode = kwargs.get('mode')
        if self.mode == 'PBS':
            self.subtxt = read_file(self.subpath)

        # Create/find simulation directories
        self.resultsdir = os.path.join(self.maindir, 'results/')
        self.rundir = os.path.join(self.maindir, '{}_runs/'.format(self.run))

        for d in [self.rundir, self.resultsdir]:
            try:
                os.mkdir(d)
            except: pass

        # COPY potential table
        #if self.create_pot:
        #    self.pot.create(self.rundir)
        #else:
        #    self.pot.copy(self.initdir, self.rundir)

        #self.pot.copy(self.initdir, self.rundir)
            
        self.Qavg = [0]*len(self.Q_names)


    def __str__(self):
        out = '\t'.join(self.Q_names) + '\n'
        for avg in self.Qavg:
            out += '{:.3f}\t'.format(avg)
        out += '\n'

        return out


    # This is mostly for plotting
    def get_units(self):
        if self.unittype == 'metal':
            self.PE_units = ' (eV)'
        elif self.unittype == 'real':
            self.PE_units = ' (kcal/mol)'
        else:
            self.PE_units == ''


    def run_lammps(self, mode='PBS'):
        '''
        Run unmodified potential (multiple copies)
        Main perturbative or verification second potential
        '''
        replace_in = {'SEED':'0', 'TABLECOEFF':'', 'TABLESTYLE':'',
                      'RUNDIR':'', 'TEMP':'0', 'POTFILE':''}

        # TODO: this is confusing; it's either local or it's in one dir
        if mode == 'nanoHUB_submit':
            self.pot.paircoeff = self.pot.paircoeff.replace(self.pot.paramdir, '.')
            replace_in['RUNDIR'] = '.'
            potpath=self.pot.potpath
            initdir=self.pot.paramdir,
        else:
            replace_in['RUNDIR'] = self.pot.paramdir
            potpath = None
            initdir = None

        replace_in['TABLECOEFF'] = self.pot.paircoeff
        replace_in['TABLESTYLE'] = self.pot.pairstyle
        replace_in['POTFILE'] = self.pot.potfile

        replace_sub = {'NAME': self.name, 'INFILE': self.infile}
        
        copy_list = np.arange(self.copy_start, self.Ncopies, dtype='int')
        copy_rundir = os.path.join(self.rundir, self.copy_folder+'{}')

        submit_lammps(replace_in, self.infile, self.intxt,
                      copy_rundir, copy_list, self.overwrite,
                      potpath=potpath, initdir=initdir,
                      replace_sub=replace_sub, mode=mode)


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
                self.Nmax = np.shape(thermo)[0]
                if self.Ntimes is None:
                    self.Ntimes = self.Nmax

                self.times = np.zeros([self.Ntimes, self.Ncopies], dtype='int')
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
                self.P = (thermo[0, find_columns(cols, ['Press'])[0]]
                          *0.0001/160.21766208) # bar -> GPa -> eV/A**3
                self.T = thermo[0, find_columns(cols, ['Temp'])[0]]
                self.beta = 1./8.617e-5/self.T

            # Next copy may have more or fewer steps finished
            # Return reduced/padded array
            thermo = fix_arr(self.Ntimes, thermo)

            # Randomly sample for convergence plots
            if self.sample:
                self.times[:,copy0] = np.array(sample(range(self.Nmax), self.Ntimes))
            else:
                self.times[:,copy0] = np.arange(self.Ntimes)

            for qc, (q,qn) in enumerate(zip(self.Q_cols, self.Q_names)):
                if qn == 'PotEng':
                    self.PEcol = qc
                if qn == 'Press':
                    self.Pcol = qc
                if qn == 'Volume':
                    self.Vcol = qc

                if self.Q_thermo[qc]:
                    self.Q[:, copy0, 0,0,0, qc] = thermo[self.times[:,copy0],q]
                    
                    
                elif qn == 'HeatCapacityVol':
                    self.Q[:, copy0, 0,0,0, qc] = thermo[self.times,self.Q_cols[self.PEcol]]**2
                elif qn == 'HeatCapacityPress':
                    self.Q[:, copy0, 0,0,0, qc] = (thermo[self.times,self.Q_cols[self.PEcol]]
                                                   + (thermo[self.times,self.Q_cols[self.Pcol]]*
                                                      thermo[self.times,self.Q_cols[self.Vcol]]))**2
                elif qn == 'Compressibility':
                    self.Q[:, copy0, 0,0,0, qc] = thermo[self.times,self.Q_cols[self.Vcol]]**2
                elif qn == 'ThermalExpansion':
                    self.Q[:, copy0, 0,0,0, qc] = (thermo[self.times,self.Q_cols[self.Vcol]]**2)
                                                   #(thermo[:,self.Q_cols[self.PEcol]]
                                                   # + (thermo[:,self.Q_cols[self.Pcol]]*
                                                   #    thermo[:,self.Q_cols[self.Vcol]])))

                # TODO: if there are issues, fill with NaN to ignore
                # Works okay while jobs running IF the first copy stays ahead


        self.Qavg = np.nanmean(self.Q, axis=(0,1,2,3,4))
        self.Qstd = np.nanstd(self.Q, axis=(0,1,2,3,4))


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



def fix_arr(Nmax, arr):
    (Ncurr, col) = np.shape(arr)

    ### Need all the thermo data for sampling
    #if Ncurr > Nmax:
    #    arr = arr[:Nmax,:]
    if Ncurr < Nmax:
        Nblank = Nmax - Ncurr
        arr = np.pad(arr, [(0, Nblank), (0, 0)], 'constant', constant_values=np.nan)
    # else return without modifying

    return arr


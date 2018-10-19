# FunUQ v0.1, 2018; Sam Reeve; Strachan Research Group
# https://github.rcac.purdue.edu/StrachanGroup

# import general
import os, numpy as np, shutil
from glob import glob
from random import random
from operator import methodcaller
from copy import deepcopy
from time import sleep
from matplotlib import pyplot as plt

from .qoi import fluctuation, fix_arr
from .parsetools import read_thermo
from .utils import copy_files, write_file, read_file, replace_template, submit_lammps, FUQerror

class FuncDer(object): 
    '''
    Base class for Functional Derivatives for FUNctional Uncertainty Quantification

    NOT intended to be used directly
    '''
    def __init__(self, QoI, potential, ensemble='nvt', input_dict=None):
        '''
        Defaults for base class; to be modified by user through dictionary as needed
        '''

        self.QoI = QoI
        self.pot = potential 
        self.ensemble = ensemble

        # Debugging
        self.showA = True

        # Perturbations
        self.rdfbins = 2000
        self.rdffile = "funcder_1_{}.rdf".format(self.rdfbins)
        self.rmin = self.pot.rmin #0.001
        self.rmax = self.pot.rmax #6.
        self.rN = 100
        self.rlist = np.linspace(self.rmin, self.rmax, self.rN)
        self.alist = [-0.0008, -0.0004, 0.0004, 0.0008]
        #self.alist = [-0.000375, -0.00075, 0.00075, 0.000375]
        self.clist = [0.1]
        self.Ncopies = self.QoI.Ncopies
        self.copy_start = self.QoI.copy_start

        # Either set with bruteforce subclass or by user
        self.intemplate = None
        self.subtemplate = None

        self.method = 'none'
        self.FDnames = []

        # User overrides defaults
        if input_dict != None:
            for key, val in input_dict.items():
                try: 
                    setattr(self, key, val)
                except:
                    raise KeyError("{} is not a valid input parameter.".format(key))

        self.newshape = (self.QoI.Ntimes*self.Ncopies, len(self.rlist), len(self.alist),
                         len(self.clist), len(self.QoI.Q_names))
        self.Qprime = np.zeros([self.QoI.Ntimes, self.Ncopies, len(self.rlist), len(self.alist),
                         len(self.clist), len(self.QoI.Q_names)])


    def get_names(self):
        self.FDnames = ['']*len(self.QoI.Q_names)
        for qc, q in enumerate(self.QoI.Q_names):
            self.FDnames[qc] = '{}_{}.funcder'.format(q, self.method)


    def funcder_fluct(self, q, qc):
        if q == 'HeatCapacityVol':
            self.Qperturbed[:,:,:,qc] = fluctuation(self.QoI.beta/self.QoI.T, self.Qperturbed[:,:,:,qc],
                                                    self.Qperturbed[:,:,:,self.QoI.PEcol])
        elif q == 'HeatCapacityPress':
            self.Qperturbed[:,:,:,qc] = fluctuation(self.QoI.beta/self.QoI.T, self.Qperturbed[:,:,:,qc],
                                                    self.Qperturbed[:,:,:,self.QoI.PEcol]+
                                                    self.Qperturbed[:,:,:,self.QoI.Pcol]*self.Qperturbed[:,:,:,self.QoI.Vcol])
        elif q == 'Compressibility':
            self.Qperturbed[:,:,:,qc] = fluctuation(self.QoI.beta/self.QoI.V, self.Qperturbed[:,:,:,qc],
                                                    self.Qperturbed[:,:,:,self.QoI.Vcol])
        elif q == 'ThermalExpansion':
            self.Qperturbed[:,:,:,qc] = fluctuation(self.QoI.beta/self.QoI.V, self.Qperturbed[:,:,:,qc],
                                                    self.Qperturbed[:,:,:,self.QoI.PEcol]+
                                                    self.Qperturbed[:,:,:,self.QoI.Pcol]*self.Qperturbed[:,:,:,self.QoI.Vcol])


    def write_FD(self, qc):
        txt = ("FunUQ v0.1, 2018; Sam Reeve\n"
               "Functional derivative for {} in {} with {} at {}K with {} atoms; "
               "Description: {}\n"
               .format(self.QoI.Q_names[qc], self.QoI.units[qc], self.pot.potname, 
                       self.QoI.T, self.QoI.Natoms, self.QoI.description))
        for r, fd in zip(self.rlist, self.funcder[:,qc]):
            txt += "{} {}\n".format(r, fd)

        self.FDnames[qc] = '{}_{}.funcder'.format(self.QoI.Q_names[qc], self.method)
        with open(os.path.join(self.QoI.resultsdir, self.FDnames[qc]), 'w') as f:
            f.write(txt)


    def plot_FD(self, qc, ax=None, label=None, color='navy'):
        if label == None:
            label = self.method

        if ax == None:
            fig1, ax = plt.subplots(1,1)

        if np.isnan(np.sum(self.funcder[:,qc])):
            ax.scatter(self.rlist, self.funcder[:,qc], color=color, label=label)
        else:
            ax.plot(self.rlist, self.funcder[:,qc], color=color, linewidth=4, label=label)
        ax.set_xlabel("Position ($\AA$)", fontsize='large')
        ax.set_xlim([self.rmin, self.rmax-0.5])
        x_range = np.where((self.rlist > self.rmin) & (self.rlist < self.rmax-0.5))
        FDplot =self.funcder[x_range,qc]
        FDplot = FDplot[~np.isnan(FDplot)]

        ymin = np.min(FDplot)
        ymin += -ymin*0.1
        ymax = np.max(FDplot)
        ymax += ymax*0.1
        ax.set_ylim([ymin, ymax])
        ax.set_ylabel("Functional Derivative ({}/$\AA$)".format(self.QoI.units[qc]), fontsize='large')
        #ax.set_xlim([1.5, 6.5])
        ax.legend(loc='best')
        plt.show()
        return ax


    def plot_perturb(self, qc, r0=None, c0=None, ax=None, label=None, color1='navy', color2='red'):
        if r0 == None:
            r0c = int(len(self.rlist)/2)
        else:
            diff = np.abs(np.array(self.rlist) - r0)
            r0c = np.where(diff == np.min(diff))[0]

        if c0 == None:
            c0c = int(len(self.clist)/2)
        else:
            diff = np.array(self.clist) - c0
            c0c = np.where(diff== np.min(diff))[0]

        if label == None:
            label = self.rlist[r0c]

        if ax == None:
            fig1, ax = plt.subplots(1,1)

        '''
        ax1 = self.pot.plot()
        r = self.rlist[r0c]
        c = self.clist[c0c]
        for a in self.alist:
            self.pot.create_table(r0=r, a0=a, c0=c,
                                  only_gauss=True,
                                  write=False)
            self.pot.plot(ax=ax1)
        self.pot.create_table() # Undo changes above
        '''

        ax.set_xlabel("Perturbation height{}".format(self.QoI.PE_units), fontsize='large')
        ax.set_ylabel("Potential energy{}".format(self.QoI.PE_units), fontsize='large')
        ax.set_xlim([np.min(self.alist), np.max(self.alist)])
        ax.set_ylim([np.min(self.Qperturbed[r0c, :, c0c, qc]), np.max(self.Qperturbed[r0c, :, c0c, qc])])
        ax.plot(self.alist, self.fit[0, r0c, c0c, qc]*np.array(self.alist) + self.fit[1, r0c, c0c, qc], 
                c='red', linewidth=4, zorder=0, label=label)
        ax.scatter(self.alist, self.Qperturbed[r0c, :, c0c, qc], c=color1, s=40, zorder=1)
        ax.scatter([0.0], [self.QoI.Qavg[qc]], c=color2, s=40, zorder=2)
        ax.legend(loc='best')
        plt.show()
        return ax 

    
    def calc_FD(self):
        #self.alist = np.append(self.alist, 0)

        self.funcder = np.zeros([len(self.rlist), len(self.QoI.Q_names)])
        self.fit = np.zeros([2, len(self.rlist), len(self.clist), len(self.QoI.Q_names)])
        for qc, q in enumerate(self.QoI.Q_names):
            # Add energy without perturbation
            '''
            shape = np.shape(self.Qperturbed)
            QoI_avg = np.empty([shape[0], 1, shape[2]])
            QoI_avg.fill(self.QoIavg1[qc])
            self.Qperturbed = np.concatenate((self.Qperturbed, QoI_avg), axis=1)
            '''

            # Take numerical derivatives to calculate functional derivative
            for c0c, c0 in enumerate(self.clist):
                A = np.array([self.alist, np.ones(len(self.alist))])

                for r0c, r0 in enumerate(self.rlist):
                    Qslope = self.Qperturbed[r0c, :, c0c, qc]
                    #Qlsope = Qslope[~np.isnan(Qslope)]
                    #print(Qslope, A)
                    self.fit[:, r0c, c0c, qc] = np.linalg.lstsq(A.T, Qslope, rcond=None)[0]
                    self.funcder[r0c, qc] = self.fit[0, r0c, c0c, qc]


            self.funcder[:,qc] = self.funcder[:,qc]*self.QoI.conversions[qc]

        print("Calculated functional derivatives\n")



class FuncDer_bruteforce(FuncDer):
    '''
    Subclass for bruteforce FuncDer: running direct simulations with perturbed potentials
    '''
    def __init__(self, *args, **kwargs):
        super(FuncDer_bruteforce, self).__init__(*args, **kwargs)
        
        self.rundir = os.path.join(self.QoI.maindir, 'bruteforce_runs/')
        try:
            os.mkdir(self.rundir)
        except: pass

        if self.intemplate == None:
            self.intemplate = 'bruteforce.template'
        if self.subtemplate == None:
            self.subtemplate = self.QoI.subtemplate #'bruteforce.pbs'
        self.inpath = os.path.join(self.QoI.initdir, self.intemplate)
        self.subpath = os.path.join(self.QoI.initdir, self.subtemplate)
        self.intxt = read_file(self.inpath)
        self.subtxt = read_file(self.subpath)


        self.method = 'bruteforce'
        self.get_names()
        

    # TODO replace with call to sub_lmp
    def run_lammps(self, mode='PBS'):
    
        for c0 in self.clist:
            cdir = os.path.join(self.rundir, 'c{}'.format(c0))
            try:
                os.mkdir(cdir)
            except: pass

            for r0 in self.rlist: 
                rdir = os.path.join(cdir, 'r{:.2f}'.format(r0))
                try:
                    os.mkdir(rdir)
                except: pass

                for a0 in self.alist:
                    adir = os.path.join(rdir, 'a{}'.format(a0))
                    try:
                        os.mkdir(adir)
                    except: pass

                    out = '{}_{}_{}_{}'.format(self.pot.potname, r0, a0, c0)
                    potfile = '{}.table'.format(out)
                    potpath = os.path.join(adir, potfile)
                    self.pot.create_table(r0=r0, a0=a0, c0=c0, keep=False, 
                                          savepath=potpath)
                    
                    replace_in = {'SEED':'0', 'TABLESTYLE':self.pot.pairstyle}
                    if mode == 'nanoHUB_submit':
                        paircoeff = self.pot.paircoeff.replace(self.pot.paramdir, '.')
                        replace_in['RUNDIR'] = '.'
                    else:
                        paircoeff = self.pot.paircoeff.replace(self.pot.paramdir, adir)
                        replace_in['RUNDIR'] = self.pot.paramdir

                    replace_in['TABLECOEFF'] = paircoeff
                    replace_sub = {'NAME': out, 'INFILE': self.QoI.infile}
                    replace_in[self.pot.potfile] = potfile # Only for bruteforce

                    #copy_files(self.QoI.initdir, self.rundir)
                    #shutil.copy(potfile, adir)

                    for copy in range(self.copy_start, self.copy_start + self.Ncopies):
                        copydir = os.path.join(adir, self.QoI.copy_folder+str(copy))
                        replace_in['SEED'] = str(int(random()*100000))
                        #replace_sub['NAME'] = '{}_{}'.format(self.pot,copy)
                        intxt = replace_template(self.intxt, replace_in)
                        if 'PBS' in mode:
                            subtxt = replace_template(self.subtxt, replace_sub)
                        elif 'submit' in mode:
                            subtxt = self.QoI.submit_paths(intxt, potpath=potpath)
                        else: subtxt = ''

                        submit_lammps(copydir, subfile=self.QoI.subfile, subtxt=subtxt,
                                      infile=self.QoI.infile, intxt=intxt, mode=mode)
                    sleep(1)


    # Replace with call to extrct_lmp
    def extract_lammps(self):
        # Find size of first file. This does not need to match QoI
        logfile = glob(os.path.join(self.rundir, 'c*', 'r*', 'a*', self.QoI.copy_folder+'*', self.QoI.logfile))[-1]
        Ntimes,Ncol = np.shape(read_thermo(logfile, returnval='thermo'))

        Qperturbed = np.zeros([Ntimes, self.Ncopies, len(self.rlist),
                               len(self.alist), len(self.clist), len(self.QoI.Q_names)])
        for c0c, c0 in enumerate(self.clist):
            for r0c, r0 in enumerate(self.rlist):
                for a0c, a0 in enumerate(self.alist):
                    for copy0, copy in enumerate(range(self.copy_start, self.copy_start + self.Ncopies)):
                        
                        logfile = glob(os.path.join(self.rundir,
                                                    'c{}'.format(c0), 'r{:.2f}*'.format(r0), 'a{}'.format(a0),
                                                    self.QoI.copy_folder+str(copy), self.QoI.logfile))[0]
                        thermo = read_thermo(logfile, returnval='thermo')
                        if np.shape(thermo)[0] != Ntimes:
                            thermo = fix_arr(Ntimes, thermo)
                        #except IndexError:
                        #    thermo = np.full([Ntimes, Ncol], np.nan)

                        for qc, (q, qn) in enumerate(zip(self.QoI.Q_cols, self.QoI.Q_names)):
                            if self.QoI.Q_thermo[qc]:
                                Qperturbed[:, copy0, r0c, a0c, c0c, qc] = thermo[:,q]

                            elif qn == 'HeatCapacityVol':
                                Qperturbed[:, copy0, r0c, a0c, c0c, qc] = thermo[:,self.QoI.Q_cols[self.QoI.PEcol]]**2
                            elif qn == 'HeatCapacityPress':
                                Qperturbed[:, copy0, r0c, a0c, c0c, qc] = (thermo[:,self.QoI.Q_cols[self.QoI.PEcol]]
                                                                           + thermo[:,self.QoI.Q_cols[self.QoI.Pcol]]*
                                                                           thermo[:,self.QoI.Q_cols[self.QoI.Vcol]])**2
                            elif qn == 'Compressibility':
                                Qperturbed[:, copy0, r0c, a0c, c0c, qc] = thermo[:,self.QoI.Q_cols[self.QoI.Vcol]]**2
                            elif qn == 'ThermalExpansion':
                                Qperturbed[:, copy0, r0c, a0c, c0c, qc] = thermo[:,self.QoI.Q_cols[self.QoI.Vcol]]**2

                print("Extracted thermodynamic data for position {}".format(r0))

        self.Qperturbed = np.nanmean(Qperturbed, axis=(0,1))
        #print(self.Qperturbed)

        for qc, q in enumerate(self.QoI.Q_names): 
            #if self.Q_thermo[qc]:
                #tmp = Qperturbed.reshape(self.newshape)
                #self.Qperturbed[ = np.nanmean(Qperturbed, axis=(0,1))

            if not self.QoI.Q_thermo[qc]:
                self.funcder_fluct(q, qc)
            
            
    def resubmit_lammps(self, mode='PBS', strict=False):

        for c0c, c0 in enumerate(self.clist):
            for r0c, r0 in enumerate(self.rlist):
                for a0c, a0 in enumerate(self.alist):
                    for copy0, copy in enumerate(range(self.copy_start, self.copy_start + self.Ncopies)):
                        run = False
                        adir = os.path.join(self.rundir,
                                            'c{}'.format(c0), 'r{:.2f}*'.format(r0), 'a{}'.format(a0))
                        copydir = os.path.join(adir, self.QoI.copy_folder+str(copy))
                        log = glob(os.path.join(copydir, self.QoI.logfile))
                        # If file is missing, we need to rerun
                        if not log:
                            run = True
                        # Only rerun partially finished file if "strict"
                        elif strict:
                            try:
                                thermo = read_thermo(log[0], returnval='thermo')
                            except: 
                                run = True
                                                
                        if run:
                            intxt = read_file(os.path.join(copydir, self.QoI.infile))
                            if mode == 'PBS':
                                subtxt = read_file(os.path.join(copydir, self.QoI.subfile))
                            elif 'submit' in mode:
                                potfile = '{}_{}_{}_{}.table'.format(self.pot.potname, r0, a0, c0)
                                potpath = glob(os.path.join(adir, potfile))[0]
                                subtxt = self.QoI.submit_paths(intxt, potpath=potpath)
                            
                            submit_lammps(glob(copydir)[0], subfile=self.QoI.subfile, subtxt=subtxt,
                                          infile=self.QoI.infile, intxt=intxt, mode=mode)
                            sleep(1)


    def prepare_FD(self):
        self.extract_lammps()
        


class FuncDer_perturbative(FuncDer):
    '''
    Subclass for perturbative FuncDer: running ONLY unperturbed simulation and using exponential weighting

    NOT intended to be used directly
    '''
    def __init__(self, *args, **kwargs):

        super(FuncDer_perturbative, self).__init__(*args, **kwargs)

        self.rdf_R = np.linspace(self.rmin, self.rmax, self.rdfbins)


    def exp_weighting(self):
        # Get Q for weighting 
        Hprime = self.Qprime[:,:,:,:,:,self.QoI.PEcol]
        Hprime_avg = np.nanmean(Hprime, axis=(0,1))
        #Qprime_avg = np.nanmean(self.Qprime, axis=(0,1))
        if self.ensemble == 'npt':
            Vprime = self.Qprime[:,:,:,:,:,self.QoI.Vcol]
            Vprime_avg = np.nanmean(Vprime, axis=(0,1))
            #Pprime = self.Qprime[:,:,:,:,:,self.QoI.Pcol]
            #Pprime_avg = np.nanmean(Pprime, axis=(0,1))

        # Add unperturbed QoI 
        Qprime = self.Qprime + self.QoI.Q # eV**2 + eV**2

        #print(np.shape(Hprime_avg))
        #print(Hprime_avg[0,0,0], self.QoI.beta*(Hprime-Hprime_avg)[0,0,0])
        # Average the exponential
        if self.ensemble == 'nvt':
            expweights = np.exp(-self.QoI.beta*(Hprime)) #-Hprime_avg))
        elif self.ensemble == 'npt':
            expweights = np.exp(-self.QoI.beta*((Hprime) + #-Hprime_avg) + 
                                                self.QoI.P*(Vprime))) #-Vprime_avg))) #(Pprime-Pprime_avg)))

        denominator = np.nanmean(expweights, axis=(0,1))

        # Average the QoI*exponential product
        self.Qperturbed = np.zeros([len(self.rlist), len(self.alist), len(self.clist), len(self.QoI.Q_names)])
        for qc, (q,thermoq) in enumerate(zip(self.QoI.Q_names, self.QoI.Q_thermo)):
            #Qprime = self.Qprime[:,:,:,:,:,qc]
            #Qprime_avg = np.nanmean(Qprime, axis=(0,1,2,3,4))

            #print(np.nanmean(self.Qprime[:,:,:,:,:,qc]), np.nanmean(self.QoI.Q[:,:,:,:,:,qc]))
            numerator = expweights*(Qprime[:,:,:,:,:,qc])
            #numerator = expweights*(Qprime-Qprime_avg)

            numerator = np.nanmean(numerator, axis=(0,1))
            self.Qperturbed[:,:,:,qc] = numerator/denominator

        for qc, q in enumerate(self.QoI.Q_names):
            if not thermoq:
                self.funcder_fluct(q, qc)



class FuncDer_perturb_allatom(FuncDer_perturbative):
    '''
    Subclass for perturbative FuncDer: using LAMMPS dump and rerun to get perturbation contributions
    '''
    def __init__(self, *args, **kwargs):
        super(FuncDer_perturb_allatom, self).__init__(*args, **kwargs)

        self.method = 'perturbative_allatom'
        self.get_names()

        self.reruntemplate = 'rerun.template'
        #self.rerunfile = 'in.rerun_gauss'
        self.rerunpath = os.path.join(self.QoI.initdir, self.reruntemplate)
        self.reruntxt = read_file(self.rerunpath)

        self.rerunpotdir = os.path.join(self.QoI.rundir, 'gauss')

        self.rerunsubtemplate = 'submit_rerun.template'
        self.rerunsubpath = os.path.join(self.QoI.initdir, self.rerunsubtemplate)
        self.rerunsubtxt = read_file(self.rerunsubpath)


    def rerun_gauss(self, mode='PBS'):
        try:
            os.mkdir(self.rerunpotdir)
        except: pass
        
        for copy0, copy in enumerate(range(self.copy_start, self.copy_start + self.Ncopies)):
            rerundir = os.path.join(self.QoI.rundir, self.QoI.copy_folder+str(copy), 'rerun_gauss/')
            try:
                os.mkdir(rerundir)
            except: pass

            for c0c, c0 in enumerate(self.clist):
                for r0c, r0 in enumerate(self.rlist):
                    for a0c, a0 in enumerate(self.alist):

                        out = 'gauss_{}_{}_{}'.format(r0, a0, c0)
                        outname = '{}.log'.format(out)
                        inname = 'in.{}'.format(out)
                        if not copy0:
                            potname = '{}.table'.format(out)
                            potpath = os.path.join(self.rerunpotdir, potname)
                            self.pot.create_table(r0=r0, a0=a0, c0=c0, keep=False,
                                                  only_gauss=True,
                                                  savepath=potpath)

                        replace_in = {'LOGNAME': outname, 'TABLESTYLE':self.pot.pairstyle}
                        coeff = self.pot.paircoeff.replace(self.pot.potfile, potname)
                        if mode == 'nanoHUB_submit':
                            paircoeff = coeff.replace(self.pot.paramdir, '.')
                            replace_in['RUNDIR'] = '.'
                        else:
                            paircoeff = coeff.replace(self.pot.paramdir, self.rerunpotdir)
                            replace_in['RUNDIR'] = self.rerunpotdir
                        replace_in['TABLECOEFF'] = paircoeff

                        reruntxt = replace_template(self.reruntxt, replace_in)
                        rerunpath = os.path.join(rerundir, inname)
                        write_file(reruntxt, rerunpath)

                        if 'submit' in mode:
                            subtxt = self.QoI.submit_paths(reruntxt, potpath=potpath)
                        else: 
                            subtxt = ''
                        # Run one at a time
                        if 'PBS' not in mode:
                            submit_lammps(rerundir, subtxt=subtxt,
                                          infile=inname, intxt=reruntxt, mode=mode)
                            print("Running for perturbation position {}".format(r0))
                            sleep(1)

            # Or submit all together
            if mode == 'PBS':
                replace_sub = {'NAME': 'gauss'}
                subtxt = replace_template(self.rerunsubtxt, replace_sub)
                submit_lammps(rerundir, subfile=self.QoI.subfile, subtxt=subtxt, mode=mode)


    def extract_lammps_gauss(self):
        self.Qprime = np.zeros([self.QoI.Ntimes, self.Ncopies, len(self.rlist),
                                len(self.alist), len(self.clist), len(self.QoI.Q_names)])

        for c0c, c0 in enumerate(self.clist):
            for r0c, r0 in enumerate(self.rlist):
                for a0c, a0 in enumerate(self.alist):
                    for copy0, copy in enumerate(range(self.copy_start, self.copy_start + self.Ncopies)):
                        logfile = os.path.join(self.QoI.rundir, self.QoI.copy_folder+str(copy),
                                               'rerun_gauss/', 'gauss_{}_{}_{}.log'.format(r0, a0, c0))
                        thermo = read_thermo(logfile, returnval='thermo')
                        if np.shape(thermo)[0] != self.QoI.Ntimes:
                            thermo = fix_arr(self.QoI.Ntimes, thermo)

                        for qc, q in enumerate(self.QoI.Q_cols):
                            if self.QoI.Q_thermo[qc]:
                                self.Qprime[:, copy0, r0c, a0c, c0c, qc] = thermo[:,q]

                print("Extracted thermodynamic data for perturbation position {}".format(r0))

    def prepare_FD(self):
        self.extract_lammps_gauss()
        print("Extracted perturbation data")

        self.exp_weighting()
        print("Calculated exponential weights")




class FuncDer_perturb_coord(FuncDer_perturbative):
    '''
    Subclass for perturbative FuncDer: using RDF (coordination) to get perturbation contributions
    '''
    def __init__(self, *args, **kwargs):
        super(FuncDer_perturb_coord, self).__init__(*args, **kwargs)
        
        self.method = 'perturbative_coord'
        self.get_names()

    def get_perturbs(self):
        self.gauss = np.zeros([len(self.rlist), len(self.alist), len(self.clist), self.rdfbins-1])
        self.gaussdiff = np.zeros([len(self.rlist), len(self.alist), len(self.clist), self.rdfbins-1])

        for r0c, r0 in enumerate(self.rlist):
            for a0c, a0 in enumerate(self.alist):
                for c0c, c0 in enumerate(self.clist):
                    self.gauss[r0c,a0c,c0c,:] = (1/(np.sqrt(2*np.pi)*c0)*a0*
                                                 np.exp(-(self.rdf_R-r0)**2/(2*c0**2)))[:-1]
                    self.gaussdiff[r0c,a0c,c0c,:] = ((self.rdf_R-r0)/(np.sqrt(2*np.pi)*c0**3)*a0*
                                                     np.exp(-(self.rdf_R-r0)**2/(2*c0**2)))[:-1]


    def extract_rdf(self, copy):
        bins = self.rdfbins
        
        #for copy0, copy in enumerate(range(self.copy_start, self.copy_start + self.Ncopies)):
        rdffile = os.path.join(self.QoI.rundir, self.QoI.copy_folder+str(copy), self.rdffile)
        
        rdf_slice = np.zeros([self.QoI.Ntimes, 2], dtype=int)
        rdf_slice[:,0] = np.arange(4, self.QoI.Ntimes*(bins+1)+4, bins+1)
        rdf_slice[:,1] = np.arange(4+bins, self.QoI.Ntimes*(bins+1)+4+bins, bins+1)
        with open(rdffile) as f:
            rdf_raw = f.readlines()
                
        #self.coord = np.zeros([self.Ntimes, self.Ncopies, bins])
        #self.coord = np.zeros([self.Ntimes, self.Ncopies,1,1,1,bins])

        coord = np.zeros([self.QoI.Ntimes, bins-1])
        #coord = np.full([self.QoI.Ntimes, bins-1], np.nan)
        #print(np.shape(coord))
        #for tc, t in enumerate(range(self.Ntimes)):
        for tc in range(self.QoI.Ntimes):
            #coord_cum = np.zeros([bins])

            #try: 
            coord_cum = np.array(np.array(list(map(methodcaller('split'), rdf_raw[rdf_slice[tc,0]:rdf_slice[tc,1]])), dtype=float)[:,3])
            coord[tc,:] = (coord_cum[bins-1:0:-1] - coord_cum[bins-2::-1])[::-1]
            #except IndexError: 
                #Nblank = self.QoI.Ntimes - tc
                #coord = fix_arr(self.QoI.Ntimes, coord)
                #print(np.shape(coord))
                #coord = np.pad(coord, (Nblank, 0), 'constant', constant_values=np.nan)
                #break

            #for c in range(bins-1, 0, -1):
            #    self.coord[tc, copy0, c] = coord_cum[c] - coord_cum[c-1]
                #self.coord[tc, copy0, 0,0,0, c] = coord_cum[c] - coord_cum[c-1]


        print("Extracted RDF data for copy {}".format(copy))
        return coord


    # TODO: wastes time if not PE or P
    def calc_Qprime(self):

        #for qc, q in enumerate(self.Q_names):
        for copy0, copy in enumerate(range(self.copy_start, self.copy_start + self.Ncopies)):
            coord = self.extract_rdf(copy)

            for tc in range(self.QoI.Ntimes):
                for r0c, r0 in enumerate(self.rlist):
                    for a0c, a0 in enumerate(self.alist):
                        for c0c, c0 in enumerate(self.clist):
                            #gauss, gaussdiff = self.get_perturb(r0, a0, c0)
                            
                            for qc, q in enumerate(self.QoI.Q_names):
                                
                                if q == 'PotEng' or q == 'E_vdwl':
                                    #self.Qprime[tc,copy,r0c,a0c,c0c,qc] = np.sum(self.gauss[r0c,a0c,c0c, :]*
                                    #                                             self.coord[tc, copy, :])
                                    self.Qprime[tc,copy0,r0c,a0c,c0c,qc] = np.sum(self.gauss[r0c,a0c,c0c, :]*coord[tc,:])
                                elif q == 'Press':
                                    #self.Qprime[tc,copy,r0c,a0c,c0c,qc] = np.sum(self.gaussdiff[r0c,a0c,c0c, :]*
                                    #                                             self.coord[tc, copy, :]*self.rdf_R)
                                    self.Qprime[tc,copy0,r0c,a0c,c0c,qc] = np.sum(self.gaussdiff[r0c,a0c,c0c, :]*coord[tc,:]*self.rdf_R[:-1])

        for qc, q in enumerate(self.QoI.Q_names):
            if q == 'PotEng' or q == 'E_vdwl':
                #self.Qprime[:,:,:,:,:,qc] = np.sum(self.gauss*
                #                                   self.coord, axis=-1)
                self.Qprime[:,:,:,:,:,qc] = self.Qprime[:,:,:,:,:,qc]*self.QoI.Natoms/2.
            elif q == 'Press':
                #self.Qprime[:,:,:,:,:,qc] = np.sum(self.gaussdiff*
                #                                   self.coord*self.rdf_R, axis=-1)
                self.Qprime[:,:,:,:,:,qc] = (self.Qprime[:,:,:,:,:,qc]/3./self.QoI.V)*160.2176*10000.*self.QoI.Natoms/2.
            elif q == 'HeatCapacityVol': # copying issue? 
                self.Qprime[:,:,:,:,:,qc] = deepcopy(self.Qprime[:,:,:,:,:,self.QoI.PEcol])**2 # eV**2
            elif q == 'HeatCapacityPress':
                self.Qprime[:,:,:,:,:,qc] = (self.Qprime[:,:,:,:,:,self.QoI.PEcol] + 
                                             self.Qprime[:,:,:,:,:,self.QoI.Pcol]*self.Qprime[:,:,:,:,:,self.QoI.Vcol])**2
            elif q == 'Compressibility':
                self.Qprime[:,:,:,:,:,qc] = self.Qprime[:,:,:,:,:,self.QoI.Vcol]**2
            elif q == 'ThermalExpansion':
                self.Qprime[:,:,:,:,:,qc] = self.Qprime[:,:,:,:,:,self.QoI.Vcol]**2
            # Volume: pass

        #for qc, q in enumerate(self.QoI.Q_names):
        #    self.Qprime[:,:,:,:,:,qc] = self.Qprime[:,:,:,:,:,qc]*self.QoI.conversions[qc]


    def prepare_FD(self):
        self.get_perturbs()
        print("Calculated perturbations")

        self.calc_Qprime()
        print("Calculated perturbation Hamiltonian contributions")
        self.exp_weighting()
        print("Calculated exponential weights")



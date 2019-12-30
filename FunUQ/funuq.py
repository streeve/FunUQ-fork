# FunUQ v0.1, 2018; Sam Reeve; Strachan Research Group
# https://github.rcac.purdue.edu/StrachanGroup

# import general
import sys, os, subprocess, shutil, numpy as np
from random import random; from glob import glob
from time import sleep
from matplotlib import pyplot as plt
from scipy import integrate

from .utils import read_file, replace_template, submit_lammps, FUQerror
from .parsetools import read_thermo, find_columns


class FunUQ(object): 
    '''
    Class for Functional Uncertainty Quantification
    '''
    def __init__(self, pot_main, pot_correct, Q_names, Qavg, Qavg_correct=None, FD=None, R=None, Q_units=None, PE_units='', output='verbose'):
        '''
        Defaults for base class; to be modified by user through dictionary as needed
        '''
        
        if FD is not None:
            self.funcder = FD
            self.rlist = R
        else:
            self.funcder = None
            self.rlist = None
        self.discrep = None

        self.pot_main = pot_main
        self.pot_correct = pot_correct

        self.Q_names = Q_names
        self.Qavg = Qavg
        self.Q_units = Q_units
        self.PE_units = PE_units
        self.Qavg_correct = Qavg_correct

        self.output = output 

        self.rerun_folder = 'rerun_hist'
        self.reruntemplate = 'rerun.template'
        self.hist_out = 'hist_{}_{}'
        self.hist_outname = '{}.log'
        self.hist_inname = 'in.{}'

    def read_FD(self, resultsdir, FDnames):
        for qc, q in enumerate(FDnames):
            data = np.loadtxt(os.path.join(resultsdir, q), skiprows=2)
            if not qc:
                self.rlist = data[:,0]
                self.funcder = np.zeros([np.shape(data)[0], len(FDnames)])
            self.funcder[:,qc] = data[:,1]


    def discrepancy(self, rmin=1.):
        if self.funcder is None:
            print("Functional derivative (funcder) required for discrepancy calculation.")
        if self.rlist is None: 
            print("Positions of sensitivity (rlist) required for discrepancy calculation.")
        if self.funcder is None or self.rlist is None:
            return

        use = np.where(self.rlist > rmin)
        self.rlist = self.rlist[use] #self.rlist > rmin]
        self.funcder = self.funcder[use]

        self.main_PE = np.interp(self.rlist, self.pot_main.R, self.pot_main.PE)
        #self.main_PE = main_PE[(self.rlist > rmin)]
        self.correct_PE = np.interp(self.rlist, self.pot_correct.R, self.pot_correct.PE)
        #self.correct_PE = correct_PE[(self.rlist > rmin)]

        self.discrep = self.correct_PE - self.main_PE

        if self.output == 'verbose':
            print("Calculated discrepancy\n")


    def correct(self):
        if self.discrep is None:
            print("Discrepancy (discrepancy) must be calculated to correct.")
            return

        self.funcerr = np.zeros(np.shape(self.funcder))
        self.Qcorrection = np.zeros(len(self.Q_names))
        self.Qcorrected = np.zeros(len(self.Q_names))
        self.Qavg_diff = np.zeros(len(self.Q_names))
        self.percent_error = np.zeros(len(self.Q_names))

        if self.output == 'verbose':
            print("Calculated correction terms")

        for qc, q in enumerate(self.Q_names):
            if self.output == 'verbose':
                if self.Q_units is not None:
                    print('***** {} ({}) *****'.format(q, self.Q_units[qc]))
                else: 
                    print('***** {} *****'.format(q))

            self.funcerr[:,qc] = self.funcder[:,qc]*self.discrep

            self.Qcorrection[qc] = integrate.trapz(self.funcerr[:,qc], x=self.rlist)
            self.Qcorrected[qc] = self.Qavg[qc] + self.Qcorrection[qc]
            if self.Qavg_correct is not None:
                self.percent_error[qc] = np.abs(self.Qcorrected[qc] - self.Qavg_correct[qc])/self.Qavg_correct[qc] * 100
                self.Qavg_diff[qc] = self.Qavg_correct[qc] - self.Qavg[qc]

                if self.output == 'verbose':
                    print('FunUQ\t\t FunUQ\t\t Direct\t\t Main\t\t Correction')
                    print('% error\t\t correction\t simulation\t QoI\t\t QoI')
                    print('{0:.2f}%\t\t {1:.3e}\t '
                          '{2:.3e}\t {3:.3e}\t {4:.3e}\n'
                          .format(self.percent_error[qc], self.Qcorrection[qc],
                                  self.Qavg_diff[qc], self.Qavg[qc], 
                                  self.Qavg_correct[qc]))
            else:
                if self.output == 'verbose':
                    print('FunUQ\t\t Main\t\t ')
                    print('correction\t QoI\t\t ')
                    print('{0:.3e}\t {1:.3e}\t\n'.format(self.Qcorrection[qc],
                                                         self.Qavg[qc]))

    # FIXME Need to use p0 = [potID, dumpID] and loop
    def rerun_histogram(self, QoI, QoI_corr, mode='PBS'):

        self.rerunpath = os.path.join(QoI.initdir, self.reruntemplate)
        self.reruntxt = read_file(self.rerunpath)

        potlist = [self.pot_main, self.pot_correct]
        qoilist = [QoI, QoI_corr]
        combinations = [[0,0],[1,1],[1,0],[0,1]]
        for c0 in combinations:

            copy_rerundir = os.path.join(qoilist[c0[1]].rundir, 
                                         qoilist[c0[1]].copy_folder+'{}', self.rerun_folder)
            
            out = self.hist_out.format(potlist[c0[0]].potname, potlist[c0[1]].potname)
            outname = self.hist_outname.format(out)
            inname = self.hist_inname.format(out)

            potfile = '{}.table'.format(potlist[c0[0]].potname)
            potpath = os.path.join(potlist[c0[0]].paramdir, potfile)
            
            replace_in = {'LOGNAME': outname, 'TABLESTYLE':potlist[c0[0]].pairstyle}
            #coeff = potlist[c0[0]].paircoeff.replace(potlist[c0[0]].potfile, potfile)
            if mode == 'nanoHUB_submit':
                paircoeff = potlist[c0[0]].paircoeff.replace(potlist[c0[0]].paramdir, '.')
                replace_in['RUNDIR'] = '.'
                initdir=potlist[c0[0]].paramdir
            else:
                paircoeff = potlist[c0[0]].paircoeff #coeff.replace(potlist[c0[0]].paramdir, 
                replace_in['RUNDIR'] = potlist[c0[0]].paramdir
                initdir=None
            replace_in['TABLECOEFF'] = paircoeff

            copy_list = np.arange(qoilist[c0[1]].copy_start, 
                                  qoilist[c0[1]].Ncopies, dtype='int')
            
            # Run one at a time
            submit_lammps(replace_in, inname, self.reruntxt,
                          copy_rerundir, copy_list, True,
                          potpath=potpath, initdir=initdir,
                          mode=mode)
            
            if self.output == 'verbose':
                print("Re-ran for {} potential "
                      "with {} trajectory".format(potlist[c0[0]].potname, 
                                                  potlist[c0[1]].potname))
            sleep(1)


    def plot_histogram(self, QoI, QoI_corr, Nbins=None):
        fig1, ax1 = plt.subplots(1,1)

        clist = ['g','b']*2
        energy = np.zeros([QoI.Ntimes, QoI.Ncopies, 4])
        diff = np.zeros([QoI.Ntimes, QoI.Ncopies, 2])
        if Nbins is None:
            Nbins = int(np.sqrt(QoI.Ntimes*QoI.Ncopies))

        potlist = [self.pot_main, self.pot_correct]
        qoilist = [QoI, QoI_corr]
        combinations = [[0,0],[1,1],[1,0],[0,1]]
        for col, c0 in enumerate(combinations):
            copy_rerundir = os.path.join(qoilist[c0[1]].rundir,
                                         qoilist[c0[1]].copy_folder+'{}', self.rerun_folder)

            out = self.hist_out.format(potlist[c0[0]].potname, potlist[c0[1]].potname)
            outname = self.hist_outname.format(out)

            for copy0, copy in enumerate(range(qoilist[c0[1]].copy_start, qoilist[c0[1]].copy_start + qoilist[c0[1]].Ncopies)):
                logfile = os.path.join(copy_rerundir.format(copy), outname)

                thermo = read_thermo(logfile, returnval='thermo')
                #if np.shape(thermo)[0] != qoilist[c0[1]].Ntimes:
                #    thermo = fix_arr(qoilist[c0[1]].Ntimes, thermo)

                energy[:, copy0, col] = thermo[qoilist[c0[1]].times[:,copy0], qoilist[c0[1]].Q_cols[qoilist[c0[1]].PEcol]]

            if self.output == 'verbose':
                print("Extracted thermodynamic data for {} potential "
                      "with {} trajectory".format(potlist[c0[0]].potname,
                                                 potlist[c0[1]].potname))

        # forward
        diff[:,:,0] = (energy[:,:,2] - energy[:,:,0])
        # reverse
        diff[:,:,1] = -(energy[:,:,3] - energy[:,:,1])

        shape0 = np.shape(energy)
        diff = np.reshape(diff, (shape0[0]*shape0[1], 2))
        
        #energy = np.reshape(energy, (shape0[0]*shape0[1], 4))  ###
        hist = np.zeros([Nbins, 2]) #
        bins_centered= np.zeros([Nbins, 2]) #
        for k in range(2): 
            hist[:,k], bins = np.histogram(diff[:,k], bins=Nbins, density=True)
            #hist[:,k], bins = np.histogram(energy[:,k], bins=Nbins, density=True)
            bins_centered[:,k] = 0.5*(bins[1:]+bins[:-1])
            
            ax1.plot(bins_centered[:,k], hist[:,k], c=clist[k], linewidth=4)
            ax1.set_ylabel("Probability", fontsize='large')
            ax1.set_xlabel("Potential energy{}".format(self.PE_units), fontsize='large')
        plt.show()
            
        #np.savetxt(savedir+savename, result)


    def plot_discrep(self):
        if self.discrep is None:
            print("Discrepancy (discrepancy) must be calculated to plot.")
            return

        fig1, ax1 = plt.subplots(1,1)
        ax1.plot(self.rlist, self.discrep, c='navy', linewidth=4)
        #ax1.scatter(self.rlist, self.discrep, c='navy')
        ax1.set_ylabel("Functional discrepancy{}".format(self.PE_units), fontsize='large')
        ax1.set_xlabel("Position ($\AA$)", fontsize='large')
        #ax1.set_xlim([1.5, 6.5])
        #ax1.set_ylim([-0.04, 0.03])
        plt.show()


    def plot_funcerr(self, qc):
        if self.funcder is None:
            print("Functional derivative (funcder) required for discrepancy calculation.")
        if self.rlist is None:
                print("Positions of sensitivity (rlist) required for discrepancy calculation.")
        if self.discrep is None:
            print("Discrepancy (discrepancy) must be calculated to plot.")
        if self.funcder is None or self.rlist is None or self.discrep is None:
            return

        fig1, ax1 = plt.subplots(1,1)
        ax1.plot(self.rlist, self.funcerr[:,qc], c='navy', linewidth=4)
        #ax1.scatter(self.rlist, self.funcerr[:,qc], c='navy')
        ax1.fill_between(self.rlist, 0, self.funcerr[:,qc], color='lightblue')
        ax1.set_ylabel("Functional error {} ({})".format(self.Q_names[qc], self.Q_units[qc]), fontsize='large')
        ax1.set_xlabel("Position ($\AA$)", fontsize='large')
        #ax1.set_xlim([1.5, 6.5])
        plt.show()


    def plot_correction(self, qc):
        if self.funcder is None:
            print("Functional derivative (funcder) required for discrepancy calculation.")
        if self.rlist is None:
            print("Positions of sensitivity (rlist) required for discrepancy calculation.")
        if self.discrep is None:
            print("Discrepancy (discrepancy) must be calculated to plot.")
        if self.funcder is None or self.rlist is None or self.discrep is None:
            return

        if self.Qavg_correct is not None:
            fig1, ax1 = plt.subplots(1,1)
            ind = np.arange(2)
            bar = ax1.bar(ind, [self.Qcorrection[qc], self.Qavg_diff[qc]],
                          color='grey', linewidth=4)
            bar[0].set_color('navy')
            ax1.set_ylabel("Correction {} ({})".format(self.Q_names[qc], self.Q_units[qc]), fontsize='large')
            ax1.set_xticks(ind)
            ax1.set_xticklabels(('FunUQ correction', 'Direct Simulation'))

            if np.abs(self.Qavg_diff[qc]) < 1e-10:
                print (self.Qavg_diff[qc])
                if self.Qcorrection[qc] > 0:
                    ax1.set_ylim([0, self.Qcorrection[qc]*1e2])
                else: 
                    ax1.set_ylim([self.Qcorrection[qc]*1e2, 0])

            plt.show()

        else:
            print("No correction simulation to compare FunUQ prediction with!")

    def write_all():
        self.write_discrepancy()
        self.write_funcder()
        self.write_funcerr()



# FunUQ v0.1, 2018; Sam Reeve; Strachan Research Group
# https://github.rcac.purdue.edu/StrachanGroup

# import general
import sys, os, subprocess, shutil, numpy as np
from random import random; from glob import glob
from matplotlib import pyplot as plt
from scipy import integrate



class FunUQ(object): 
    '''
    Class for Functional Uncertainty Quantification
    '''
    def __init__(self, pot_main, pot_correct, Q_names, Qavg, Qavg_correct, FD=None, R=None, Q_units=None, PE_units=''):
        '''
        Defaults for base class; to be modified by user through dictionary as needed
        '''
        
        if FD is not None:
            self.funcder = FD
            self.rlist = R
        else:
            self.funcder = []

        self.pot_main = pot_main
        self.pot_correct = pot_correct

        self.Q_names = Q_names
        self.Qavg = Qavg
        self.Qavg_correct = Qavg_correct
        self.Q_units = Q_units
        self.PE_units = PE_units


    def read_FD(self, resultsdir, FDnames):
        for qc, q in enumerate(FDnames):
            data = np.loadtxt(os.path.join(resultsdir, q), skiprows=2)
            if not qc:
                self.rlist = data[:,0]
                self.funcder = np.zeros([np.shape(data)[0], len(FDnames)])
            self.funcder[:,qc] = data[:,1]


    def discrepancy(self, rmin=1.):

        use = np.where(self.rlist > rmin)
        self.rlist = self.rlist[use] #self.rlist > rmin]
        self.funcder = self.funcder[use]

        self.main_PE = np.interp(self.rlist, self.pot_main.R, self.pot_main.PE)
        #self.main_PE = main_PE[(self.rlist > rmin)]
        self.correct_PE = np.interp(self.rlist, self.pot_correct.R, self.pot_correct.PE)
        #self.correct_PE = correct_PE[(self.rlist > rmin)]

        self.discrep = self.correct_PE - self.main_PE

        print("Calculated discrepancy\n")


    def correct(self):
        self.funcerr = np.zeros(np.shape(self.funcder))
        self.Qcorrection = np.zeros(len(self.Q_names))
        self.Qcorrected = np.zeros(len(self.Q_names))
        self.percent_error = np.zeros(len(self.Q_names))

        print("Calculated correction terms")

        for qc, q in enumerate(self.Q_names):
            self.funcerr[:,qc] = self.funcder[:,qc]*self.discrep

            self.Qcorrection[qc] = integrate.trapz(self.funcerr[:,qc], x=self.rlist)
            self.Qcorrected[qc] = self.Qavg[qc] + self.Qcorrection[qc]
            self.percent_error[qc] = np.abs(self.Qcorrected[qc] - self.Qavg_correct[qc])/self.Qavg_correct[qc] * 100

            print('***** {} *****'.format(q))
            print('FunUQ\t\t FunUQ\t\t Direct\t\t Main\t\t Correction')
            print('% error\t\t correction\t simulation\t QoI\t\t QoI')
            print('{0:.2f}%\t\t {1:.3e}\t {2:.3e}\t {3:.3e}\t {4:.3e}\n'.format(self.percent_error[qc], self.Qcorrection[qc], (self.Qavg_correct[qc] - self.Qavg[qc]), self.Qavg[qc], self.Qavg_correct[qc]))


    def plot_discrep(self):
        fig1, ax1 = plt.subplots(1,1)
        ax1.plot(self.rlist, self.discrep, c='navy', linewidth=4)
        #ax1.scatter(self.rlist, self.discrep, c='navy')
        ax1.set_ylabel("Functional discrepancy{}".format(self.PE_units), fontsize='large')
        ax1.set_xlabel("Position ($\AA$)", fontsize='large')
        #ax1.set_xlim([1.5, 6.5])
        #ax1.set_ylim([-0.04, 0.03])
        plt.show()


    def plot_funcerr(self, qc):
        fig1, ax1 = plt.subplots(1,1)
        ax1.plot(self.rlist, self.funcerr[:,qc], c='navy', linewidth=4)
        #ax1.scatter(self.rlist, self.funcerr[:,qc], c='navy')
        ax1.fill_between(self.rlist, 0, self.funcerr[:,qc], color='lightblue')
        ax1.set_ylabel("Functional error {} ({})".format(self.Q_names[qc], self.Q_units[qc]), fontsize='large')
        ax1.set_xlabel("Position ($\AA$)", fontsize='large')
        #ax1.set_xlim([1.5, 6.5])
        plt.show()


    def plot_correction(self, qc):
        fig1, ax1 = plt.subplots(1,1)
        ind = np.arange(2)
        bar = ax1.bar(ind,
                      [self.Qcorrection[qc], self.Qavg_correct[qc] - self.Qavg[qc]], 
                      color='grey', linewidth=4)
        bar[0].set_color('navy')
        ax1.set_ylabel("Correction {} ({})".format(self.Q_names[qc], self.Q_units[qc]), fontsize='large')
        ax1.set_xticks(ind)
        ax1.set_xticklabels(('FunUQ correction', 'Direct Simulation'))

        if self.Qavg[qc] - self.Qavg_correct[qc] < 1e-10:
            if self.Qcorrection[qc] > 0:
                ax1.set_ylim([0, self.Qcorrection[qc]*1e2])
            else: 
                ax1.set_ylim([self.Qcorrection[qc]*1e2, 0])

        plt.show()


    def write_all():
        self.write_discrepancy()
        self.write_funcder()
        self.write_funcerr()



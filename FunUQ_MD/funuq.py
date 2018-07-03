# FunUQ v0.4, 2018; Sam Reeve; Strachan Research Group
# https://github.rcac.purdue.edu/StrachanGroup

# import general
import sys, os, subprocess, shutil, numpy as np
from random import random; from glob import glob
from matplotlib import pyplot as plt
from scipy import integrate

# import local functions
from parsetools import *
from potential import *



class FunUQ(object): 
    '''
    Class for FUNctional Uncertainty Quantification
    '''
    def __init__(self, pot_main, pot_correct, Q_names, Qavg, Qavg_correct, FD=None, R=None):
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
        print self.rlist 

        self.main_PE = np.interp(self.rlist, self.pot_main.R, self.pot_main.PE)
        #self.main_PE = main_PE[(self.rlist > rmin)]
        self.correct_PE = np.interp(self.rlist, self.pot_correct.R, self.pot_correct.PE)
        #self.correct_PE = correct_PE[(self.rlist > rmin)]

        self.discrep = self.main_PE - self.correct_PE

        print("Calculated discrepancy")


    def correct(self):
        #self.funcerr = self.funcder*self.discrep
        self.funcerr = np.zeros(np.shape(self.funcder))
        self.Qcorrection = np.zeros(len(self.Q_names))
        self.Qcorrected = np.zeros(len(self.Q_names))

        for qc in range(len(self.Q_names)):
            self.funcerr[:,qc] = self.funcder[:,qc]*self.discrep

            self.Qcorrection[qc] = integrate.trapz(self.funcerr[:,qc], x=self.rlist)
            self.Qcorrected[qc] = self.Qavg[qc] + self.Qcorrection[qc]
            
            print self.Qcorrection[qc], (self.Qavg[qc] - self.Qavg_correct[qc]), self.Qavg[qc], self.Qavg_correct[qc]
        print("Calculated correction terms")



    def plot_discrep(self):
        plt.plot(self.rlist, self.discrep, c='navy')
        plt.scatter(self.rlist, self.discrep, c='navy')
        plt.show()


    def plot_funcerr(self, qc):
        plt.plot(self.rlist, self.funcerr[:,qc], c='navy')
        plt.scatter(self.rlist, self.funcerr[:,qc], c='navy')
        plt.show()


    def write_all():
        self.write_discrepancy()
        self.write_funcder()
        self.write_funcerr()


        
class FunUQexception(Exception):
    pass



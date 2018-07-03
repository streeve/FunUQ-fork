# FunUQ v0.4, 2018; Sam Reeve; Strachan Research Group
# https://github.rcac.purdue.edu/StrachanGroup

# import general
import sys, os, subprocess, shutil, numpy as np
from random import random; from glob import glob
from matplotlib import pyplot as plt

# import local functions
from parsetools import *
from qoi import *
from potential import create_table


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

        '''
        if self.ensemble == 'nvt' and 'Volume' in self.QoI.Q_names:
            self.QoI.Q_names.remove('Volume')
            print "Cannot predict volume with constant volume"
        if self.ensemble == 'npt' and 'Press' in self.QoI.Q_names:
            self.QoI.Q_names.remove('Press')
            print "Cannot predict pressure with constant pressure"
        '''
        # Debugging
        self.showA = False

        # Perturbations
        self.rdfbins = 2000
        self.rdffile = "funcder_1_{}.rdf".format(self.rdfbins)
        self.rmin = 0.001
        self.rmax = 6.
        self.rN = 30
        self.rlist = np.linspace(self.rmin, self.rmax, self.rN)
        self.alist = [-0.0008, -0.0004, 0.0004, 0.0008]
        #self.alist = [-0.000375, -0.00075, 0.00075, 0.000375]
        self.clist = [0.1]

        self.newshape = (self.QoI.Ntimes*self.QoI.Ncopies, len(self.rlist), len(self.alist),
                         len(self.clist), len(self.QoI.Q_names))
        self.Qprime = np.zeros([self.QoI.Ntimes, self.QoI.Ncopies, len(self.rlist), len(self.alist),
                         len(self.clist), len(self.QoI.Q_names)])
        self.method = 'none'
        self.FDnames = []

        # User overrides defaults
        if input_dict != None:
            for key, val in input_dict.iteritems():
                try: 
                    setattr(self, key, val)
                except:
                    raise KeyError("{} is not a valid input parameter.".format(key))


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
        txt = ("FunUQ v0.4, 2018; Sam Reeve\n"
               "Functional derivative for {} in {} with {} at {}K with {} atoms; "
               "Description: {}\n"
               .format(self.QoI.Q_names[qc], self.QoI.units[qc], self.pot.potname, 
                       self.QoI.T, self.QoI.Natoms, self.QoI.description))
        for r, fd in zip(self.rlist, self.funcder[:,qc]):
            txt += "{} {}\n".format(r, fd)

        self.FDnames[qc] = '{}_{}.funcder'.format(self.QoI.Q_names[qc], self.method)
        with open(os.path.join(self.QoI.resultsdir, self.FDnames[qc]), 'w') as f:
            f.write(txt)


    def plot_FD(self, qc):
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
        plt.ylabel("Functional Derivative ({}/$\AA$)".format(self.QoI.units[qc]), fontsize='large')
        plt.show()

    
    def calc_FD(self):
        #self.alist = np.append(self.alist, 0)

        self.funcder = np.zeros([len(self.rlist), len(self.QoI.Q_names)])
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
                    m = np.linalg.lstsq(A.T, self.Qperturbed[r0c, :, c0c, qc])[0]
                    self.funcder[r0c, qc] = m[0]

                    if self.showA and 'HeatCapacity' in q:
                        print r0, self.Qperturbed[r0c, :, c0c, qc], self.alist
                        plt.xlim([np.min(self.alist), np.max(self.alist)])
                        plt.ylim([np.min(self.Qperturbed[r0c, :, c0c, qc]), np.max(self.Qperturbed[r0c, :, c0c, qc])])
                        plt.scatter(self.alist, self.Qperturbed[r0c, :, c0c, qc])
                        plt.plot(self.alist, m[0]*np.array(self.alist) + m[1])
                        plt.show()

            self.funcder[:,qc] = self.funcder[:,qc]*self.QoI.conversions[qc]
            '''
            # Unit conversion
            if q == 'PotEng':
                self.funcder[:,qc] = self.funcder[:,qc]/self.QoI.Natoms
            elif q == 'Press':
                self.funcder[:,qc] = self.funcder[:,qc]*0.0001
            '''
        print("Calculated functional derivative")



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

        self.method = 'bruteforce'
        self.get_names()
        

    # TODO replace with call to sub_lmp
    def run_lammps(self):
    
        for c0 in self.clist:
            cdir = os.path.join(self.rundir, 'c{}'.format(c0))
            try:
                os.mkdir(cdir)
            except: pass

            for r0 in self.rlist: 
                rdir = os.path.join(cdir, 'r{}'.format(r0))
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
                    R, PE, style, coeff = create_table(self.pot.potname, self.pot.parampath,
                                                       #self.pot, self.parampath,
                                                       r0=r0, a0=a0, c0=c0, 
                                                       outname=potfile, savedir=adir)
                    
                    # This is to fix data path IF ../ is there (as it should be for multiple copies
                    replace_in = {'../':'../../../../', 
                                  'TABLECOEFF':coeff, 'TABLESTYLE':style, 'SEED':'0'}
                    replace_sub = {'NAME': out, 'INFILE': self.QoI.infile}
                    copy_files(self.QoI.initdir, self.rundir)
                    #shutil.copy(potfile, adir)

                    for copy in range(self.QoI.copy_start, self.QoI.copy_start + self.QoI.Ncopies):
                        copydir = os.path.join(adir, self.QoI.copy_folder+str(copy))
                        replace_in['SEED'] = str(int(random()*100000))
                        #replace_sub['NAME'] = '{}_{}'.format(self.pot,copy)
                        intxt = replace_template(self.QoI.intxt, replace_in)
                        subtxt = replace_template(self.QoI.subtxt, replace_sub)
                        submit_lammps(self.QoI.subfile, subtxt,
                                      self.QoI.infile, intxt, copydir)


    # Replace with call to extrct_lmp
    def extract_lammps(self):
        Qperturbed = np.zeros([self.QoI.Ntimes, self.QoI.Ncopies, len(self.rlist),                                                                
                               len(self.alist), len(self.clist), len(self.QoI.Q_names)])
        first = True
        for c0c, c0 in enumerate(self.clist):
            for r0c, r0 in enumerate(self.rlist):
                for a0c, a0 in enumerate(self.alist):
                    for copy0, copy in enumerate(range(self.QoI.copy_start, self.QoI.copy_start + self.QoI.Ncopies)):
                        logfile = os.path.join(self.rundir,
                                               'c{}'.format(c0), 'r{}'.format(r0), 'a{}'.format(a0),
                                               self.QoI.copy_folder+str(copy), self.QoI.logfile)
                        thermo = read_thermo(logfile, returnval='thermo')

                        '''
                        if first:
                            first = False
                            self.Natoms = Natoms
                            self.Ntimes = np.shape(thermo)[0]
                            #self.times = np.array(sample(range(startsteps, totalsteps), self.Ntimes))
                            Qperturbed = np.zeros([self.Ntimes, self.Ncopies, len(self.rlist),
                                                    len(self.alist), len(self.clist), len(self.Q_names)])
                            self.Q_cols = find_columns(cols, self.Q_names)
                            self.T = thermo[0, find_columns(cols, ['Temp'])[0]]
                        '''
 
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

                print("Extracted thermo data for position {}".format(r0))

        self.Qperturbed = np.mean(Qperturbed, axis=(0,1))

        for qc, q in enumerate(self.QoI.Q_names): 
            #if self.Q_thermo[qc]:
                #tmp = Qperturbed.reshape(self.newshape)
                #self.Qperturbed[ = np.mean(Qperturbed, axis=(0,1))

            if not self.QoI.Q_thermo[qc]:
                self.funcder_fluct(q, qc)
            '''
            if q == 'HeatCapacityVol':
                #Q = self.Qperturbed[:,:,0,0,0,self.PEcol]
                #Q2 = self.Q[:,:,0,0,0,qc] # already squared                                                                                   
                self.Qperturbed[:,:,:,qc] = fluctuation(self.QoI.beta/self.QoI.T, self.Qperturbed[:,:,:,qc],
                                                        self.Qperturbed[:,:,:,self.QoI.PEcol])
                #self.QoI.beta/(self.QoI.T)*(self.Qperturbed[:,:,:,qc] - self.Qperturbed[:,:,:,self.QoI.PEcol]**2)
            elif q == 'HeatCapacityPress':
                self.Qperturbed[:,:,:,qc] = fluctuation(self.QoI.beta/self.QoI.T, self.Qperturbed[:,:,:,qc],
                                                        self.Qperturbed[:,:,:,self.QoI.PEcol]+
                                                        self.Qperturbed[:,:,:,self.QoI.Pcol]*self.Qperturbed[:,:,:,self.QoI.Vcol])
            elif q == 'Compressibility':
                self.Qperturbed[:,:,:,qc] = fluctuation(self.beta/self.V, self.Qperturbed[:,:,:,qc], 
                                                        self.Qperturbed[:,:,:,self.QoI.Vcol])
            elif q == 'ThermalExpansion':
                self.Qperturbed[:,:,:,qc] = fluctuation(self.beta/self.V, self.Qperturbed[:,:,:,qc],
                                                        self.Qperturbed[:,:,:,self.QoI.PEcol]+
                                                        self.Qperturbed[:,:,:,self.QoI.Pcol]*self.Qperturbed[:,:,:,self.QoI.Vcol])
            '''
            
            
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
        #newshape = (self.Ntimes*self.Ncopies, len(self.rlist), len(self.alist), len(self.clist))

        # Get Q for weighting 
        Hprime = self.Qprime[:,:,:,:,:,self.QoI.PEcol]
        #tmp = Hprime.reshape(newshape)
        #Hprime_avg = np.mean(tmp, axis=0)
        Hprime_avg = np.mean(Hprime, axis=(0,1))
        if self.ensemble == 'npt':
            Vprime = self.Qprime[:,:,:,:,:,self.QoI.Vcol]
            Vprime_avg = np.mean(Vprime, axis=(0,1))
            Pprime = self.Qprime[:,:,:,:,:,self.QoI.Pcol]
            Pprime_avg = np.mean(Pprime, axis=(0,1))

        # Add unperturbed QoI
        Qprime = self.Qprime + self.QoI.Q

        # Average the exponential
        if self.ensemble == 'nvt':
            print self.ensemble
            expweights = np.exp(-self.QoI.beta*(Hprime-Hprime_avg))
        elif self.ensemble == 'npt':
            print self.ensemble
            expweights = np.exp(-self.QoI.beta*((Hprime-Hprime_avg) + 
                                                (Vprime-Vprime_avg)*(Pprime-Pprime_avg)))

        #tmp = expweights.reshape(newshape)
        #denominator = np.mean(tmp, axis=0)
        denominator = np.mean(expweights, axis=(0,1))

        # Average the QoI*exponential product
        self.Qperturbed = np.zeros([len(self.rlist), len(self.alist), len(self.clist), len(self.QoI.Q_names)])
        for qc, (q,thermoq) in enumerate(zip(self.QoI.Q_names, self.QoI.Q_thermo)):
            numerator = expweights*Qprime[:,:,:,:,:,qc]

            #tmp = numerator.reshape(newshape)
            #numerator = np.mean(tmp, axis=0)
            numerator = np.mean(numerator, axis=(0,1))
            self.Qperturbed[:,:,:,qc] = numerator/denominator

            if not thermoq:
                self.funcder_fluct(q, qc)
            #if q == 'HeatCapacityVol':
                #self.QoI.beta/(self.QoI.T)*(self.Qperturbed[:,:,:,self.QoI.PEcol]**2 - self.Qperturbed[:,:,:,qc])


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
        self.rerunpath = os.path.join(self.initdir, self.reruntemplate)
        self.reruntxt = read_template(self.rerunpath)

        self.rerunpotdir = os.path.join(self.rundir, 'gauss')

        self.rerunsubtemplate = 'submit_rerun.template'
        self.rerunsubpath = os.path.join(self.initdir, self.rerunsubtemplate)
        self.rerunsubtxt = read_template(self.rerunsubpath)

        #super(FuncDer_perturb_allatom, self).__init__(*args, **kwargs)


    def rerun_gauss(self):
        try:
            os.mkdir(self.rerunpotdir)
        except: pass
        
        for copy0, copy in enumerate(range(self.QoI.copy_start, self.QoI.copy_start + self.QoI.Ncopies)):
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
                            R, PE, style, coeff = create_table('gauss', '', r0=r0, a0=a0, c0=c0, 
                                                               outname=potname,
                                                               savedir=self.rerunpotdir)
                        replace = {'LOGNAME': outname,
                                   'TABLECOEFF':coeff, 'TABLESTYLE':style, 
                                   '* * ../': '* * ../../gauss/'}
                        reruntxt = replace_template(self.reruntxt, replace)
                        write_file(reruntxt, os.path.join(rerundir, inname))

            replace_sub = {'NAME': 'gauss'}
            subtxt = replace_template(self.rerunsubtxt, replace_sub)
            submit_lammps(self.subfile, subtxt,
                          'tmp', '', rerundir)


    def extract_lammps_gauss(self):
        self.Qprime = np.zeros([self.QoI.Ntimes, self.QoI.Ncopies, len(self.rlist),
                                len(self.alist), len(self.clist), len(self.QoI.Q_names)])

        for c0c, c0 in enumerate(self.clist):
            for r0c, r0 in enumerate(self.rlist):
                for a0c, a0 in enumerate(self.alist):
                    for copy0, copy in enumerate(range(self.QoI.copy_start, self.QoI.copy_start + self.QoI.Ncopies)):
                        logfile = os.path.join(self.QoI.rundir, self.QoI.copy_folder+str(copy),
                                               'rerun_gauss/', 'gauss_{}_{}_{}.log'.format(r0, a0, c0))
                        cols, thermo, Natoms = read_thermo(logfile)
                        for qc, q in enumerate(self.QoI.Q_cols):
                            self.Qprime[:, copy0, r0c, a0c, c0c, qc] = thermo[:,q]

                print("Extracted thermo data for perturbation position {}".format(r0))


    def prepare_FD(self):
        #self.extract_lammps(self.mainpot)
        #print("Extracted thermo data")

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
        #for tc, t in enumerate(range(self.Ntimes)):
        for tc in range(self.QoI.Ntimes):
            coord_cum = np.zeros([bins])
            coord_cum = np.array(np.array(map(methodcaller('split'), rdf_raw[rdf_slice[tc,0]:rdf_slice[tc,1]]), dtype=float)[:,3])

            coord[tc,:] = (coord_cum[bins-1:0:-1] - coord_cum[bins-2::-1])[::-1]
            #for c in range(bins-1, 0, -1):
            #    self.coord[tc, copy0, c] = coord_cum[c] - coord_cum[c-1]
                #self.coord[tc, copy0, 0,0,0, c] = coord_cum[c] - coord_cum[c-1]


        print("Extracted RDF data for copy {}".format(copy))
        return coord


    # TODO: wastes time if not PE or P
    def calc_Qprime(self):

        #for qc, q in enumerate(self.Q_names):
        for copy0, copy in enumerate(range(self.QoI.copy_start, self.QoI.copy_start + self.QoI.Ncopies)):
            coord = self.extract_rdf(copy)

            for tc in range(self.QoI.Ntimes):
                for r0c, r0 in enumerate(self.rlist):
                    for a0c, a0 in enumerate(self.alist):
                        for c0c, c0 in enumerate(self.clist):
                            #gauss, gaussdiff = self.get_perturb(r0, a0, c0)
                            
                            for qc, q in enumerate(self.QoI.Q_names):

                                if q == 'PotEng':
                                    #self.Qprime[tc,copy,r0c,a0c,c0c,qc] = np.sum(self.gauss[r0c,a0c,c0c, :]*
                                    #                                             self.coord[tc, copy, :])
                                    self.Qprime[tc,copy0,r0c,a0c,c0c,qc] = np.sum(self.gauss[r0c,a0c,c0c, :]*coord[tc,:])
                                elif q == 'Press':
                                    #self.Qprime[tc,copy,r0c,a0c,c0c,qc] = np.sum(self.gaussdiff[r0c,a0c,c0c, :]*
                                    #                                             self.coord[tc, copy, :]*self.rdf_R)
                                    self.Qprime[tc,copy0,r0c,a0c,c0c,qc] = np.sum(self.gaussdiff[r0c,a0c,c0c, :]*coord[tc,:]*self.rdf_R[:-1])

        for qc, q in enumerate(self.QoI.Q_names):
            if q == 'PotEng':
                #self.Qprime[:,:,:,:,:,qc] = np.sum(self.gauss*
                #                                   self.coord, axis=-1)
                self.Qprime[:,:,:,:,:,qc] = self.Qprime[:,:,:,:,:,qc]*self.QoI.Natoms/2.
            elif q == 'Press':
                #self.Qprime[:,:,:,:,:,qc] = np.sum(self.gaussdiff*
                #                                   self.coord*self.rdf_R, axis=-1)
                self.Qprime[:,:,:,:,:,qc] = (self.Qprime[:,:,:,:,:,qc]/3./self.QoI.V)*160.2176*10000.*self.QoI.Natoms/2.
            elif q == 'HeatCapacityVol': # copying issue? 
                self.Qprime[:,:,:,:,:,qc] = np.copy(self.Qprime[:,:,:,:,:,self.QoI.PEcol]**2)
            elif q == 'HeatCapacityPress':
                self.Qprime[:,:,:,:,:,qc] = (self.Qprime[:,:,:,:,:,self.QoI.PEcol] + 
                                             self.Qprime[:,:,:,:,:,self.QoI.Pcol]*self.Qprime[:,:,:,:,:,self.QoI.Vcol])**2
            elif q == 'Compressibility':
                self.Qprime[:,:,:,:,:,qc] = self.Qprime[:,:,:,:,:,self.QoI.Vcol]**2
            elif q == 'ThermalExpansion':
                self.Qprime[:,:,:,:,:,qc] = self.Qprime[:,:,:,:,:,self.QoI.Vcol]**2
            # Volume: pass


    def prepare_FD(self):
        self.get_perturbs()
        print("Calculated perturbations")

        self.calc_Qprime()
        print("Calculated perturbation Hamiltonian contribution")
        self.exp_weighting()
        print("Calculated exponential weights")


        
class FuncDerException(Exception):
    pass



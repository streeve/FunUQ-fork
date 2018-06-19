# FunUQ v0.3, 2018; Sam Reeve; Strachan Research Group
# https://github.rcac.purdue.edu/StrachanGroup

# import general
import sys, os, subprocess, shutil, numpy as np
from random import random; from glob import glob
from matplotlib import pyplot as plt

# import local functions
from parsetools import *
from create_table import *


class potential(object):

    def __init__(self, pot, paramfile=None, paramdir=None, 
                 potfile=None, potdir='', create=True):
        self.pot = pot
        if paramfile == None:
            self.paramfile = '{}.params'.format(self.pot)

        if paramdir != None:
            self.parampath = os.path.join(paramdir, self.paramfile)
        else:
            self.parampath = self.paramfile

        if potfile == None:
            self.potfile = '{}.table'.format(self.pot)
            self.potpath = os.path.join(potdir, self.potfile)
        
    def create(self, potdir):
        self.potdir = potdir
        R, PE, style, coeff = create_table(self.pot, self.parampath, savedir=potdir)
        self.R = R
        self.PE = PE
        self.pairstyle = style
        self.paircoeff = coeff



class FunUQ(object): 
    '''
    Base class for FUNctional Uncertainty Quantification
    '''
    def __init__(self, pot_name, Qlist, initdir, maindir, correctpot_name=None, input_dict=None):
        '''
        Defaults for base class; to be modified by user through dictionary as needed
        '''

        # Debugging
        self.showA = False

        # Interatomic potential
        self.pot_name = pot_name
        self.createpot = True
        self.correctpot_name = correctpot_name
        #self.paramfile = '{}.params'.format(self.pot)
        #self.potfile = '{}.table'.format(self.pot)
        #self.pot_hifid = ''


        # Quantity of interest
        self.Q_names = list(Qlist)
        if 'PotEng' not in self.Q_names:
            self.Q_names += 'PotEng'
        self.units = []
        #self.get_units()

        # Perturbations
        self.Ncopies = 5
        self.rdfbins = 2000
        self.rmin = 0.001
        self.rmax = 6.
        self.rlist = np.linspace(self.rmin, self.rmax, 30)
        self.alist = [-0.0008, -0.0004, 0.0004, 0.0008]
        #self.alist = [-0.0016, -0.0008, -0.0004, -0.0002, 0.0002, 0.0004, 0.0008, 0.0016]
        self.clist = [0.1]

        # Files and directories
        self.intemplate = 'in.template'
        self.subtemplate = 'submit.template'
        self.infile = 'in.lammps'
        self.subfile = 'run.pbs'
        self.logfile = 'log.lammps'
        self.rdffile = "funcder_1_{}.rdf".format(self.rdfbins)
        self.initdir = initdir
        self.maindir = maindir
        self.FDname = 'out.funcder'
        
        #self.parampath = os.path.join(initdir, self.paramfile)
        self.inpath = os.path.join(self.initdir, self.intemplate)
        self.subpath = os.path.join(self.initdir, self.subtemplate)

        # Other
        self.description = ''
        self.name = self.pot_name
        self.rerun_folder = 'rerun_'
        self.copy_folder = 'copy_'
        #self.code_path = os.getcwd()

        # User overrides defaults
        if input_dict != None:
            for key, val in input_dict.iteritems():
                try: 
                    setattr(self, key, val)
                except:
                    raise KeyError("{} is not a valid input parameter.".format(key))
        
        # Create potential table
        #    R, PE, style, coeff = create_table(self.pot, self.parampath)
        #    self.pot_R = R
        #    self.pot_PE = PE
        #    self.pairstyle = style
        #    self.paircoeff = coeff
        #else:
        #    self.pot_R = 'XXX'
        #    self.pot_PE = 'XXX'
        #    self.pairstyle = 'XXX'
        #    self.paircoeff = 'XXX'
        # TODO need to get potential through pair_write instead


        # Get user templates
        self.intxt = read_template(self.inpath)
        self.subtxt = read_template(self.subpath)

        # Create/find simulation directories
        self.resultsdir = os.path.join(self.maindir, 'results/')
        self.rundir = os.path.join(self.maindir, 'main_runs/')
        if self.correctpot_name != None:
            self.correctdir = os.path.join(self.maindir, '{}_runs/'.format(self.correctpot_name))
        else: self.correctdir = None

        for d in [self.maindir, self.rundir, self.resultsdir, self.correctdir]:
            try:
                os.mkdir(d)
            except: pass

        # Create potential table
        if self.createpot:
            self.mainpot = potential(self.pot_name, paramdir=self.initdir, potdir=self.rundir)
            self.mainpot.create(self.rundir)

            if self.correctpot_name != None:
                self.correctpot = potential(self.correctpot_name, paramdir=self.initdir, potdir=self.rundir)
                self.correctpot.create(self.correctdir)

            #shutil.copy(self.potfile, self.rundir)
            #shutil.copy(self.mainpot.potfile, self.rundir)
            #if correctpot_name != None:
            #    shutil.copy(self.correctpot.potfile, self.rundir)


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


    def run_lammps(self, pot):
        '''
        Run unmodified potential (multiple copies)
        Main perturbative or verification second potential

        Args: potential class object
        '''
        if pot == None:
            replace_in = {'SEED':'0'}
            rundir = self.rundir
        else:
            replace_in = {'TABLECOEFF':pot.paircoeff, 'TABLESTYLE':pot.pairstyle, 'SEED':'0'}
            replace_sub = {'NAME': self.name, 'INFILE': self.infile}
            rundir = pot.potdir
        copy_files(self.initdir, rundir)

        for copy in range(self.Ncopies):
            rundir = os.path.join(rundir, self.copy_folder+str(copy))
            replace_in['SEED'] = str(int(random()*100000))
            replace_sub['NAME'] = '{}_{}'.format(self.name, copy)
            intxt = replace_template(self.intxt, replace_in)
            subtxt = replace_template(self.subtxt, replace_sub)
            submit_lammps(self.subfile, subtxt,
                          self.infile, intxt, rundir)


    '''
    def run_lammps_NOPOTENTIAL(self):
        ### Want user accessable, but need to know where to run (self.rundir); should be combined with above
        
        replace_in = {'SEED':'0'}
        replace_sub = {'NAME': self.name, 'INFILE': self.infile}
        copy_files(self.initdir, self.rundir)                                                      

        for copy in range(self.Ncopies):
            rundir = os.path.join(self.rundir,
                                  self.copy_folder+str(copy))

            replace_in['SEED'] = str(int(random()*100000))
            replace_sub['NAME'] = '{}_{}'.format(self.name, copy)
            intxt = replace_template(self.intxt, replace_in)
            subtxt = replace_template(self.subtxt, replace_sub)
            submit_lammps(self.subfile, subtxt,
                          self.infile, intxt, rundir)

    def run_lammps(self, ):
        replace_in = {'TABLECOEFF':self.paircoeff, 'TABLESTYLE':self.pairstyle, 'SEED':'0'}
        replace_sub = {'NAME': self.pot, 'INFILE': self.infile}
        copy_files(self.initdir, self.rundir)

        for copy in range(self.Ncopies):
            rundir = os.path.join(self.rundir, self.copy_folder+str(copy))

            replace_in['SEED'] = str(int(random()*100000))
            replace_sub['NAME'] = '{}_{}'.format(self.pot, copy)
            intxt = replace_template(self.intxt, replace_in)
            subtxt = replace_template(self.subtxt, replace_sub)
            submit_lammps(self.subfile, subtxt,
                          self.infile, intxt, rundir)
    '''

    def extract_lammps(self, log='log.lammps'):
        '''
        Read lammps log thermo data
        Used for base properties and perturbative calculations
        '''
        for copy in range(self.Ncopies):
            logfile = os.path.join(self.rundir, self.copy_folder+str(copy), log)
            print logfile
            cols, thermo, Natoms = read_thermo(logfile)

            # Only extract these values once
            if not copy:
                self.Natoms = Natoms
                self.Ntimes = np.shape(thermo)[0]
                #self.times = np.array(sample(range(startsteps, totalsteps), self.Ntimes))
                self.Qprime = np.zeros([self.Ntimes, self.Ncopies, len(self.rlist),
                                        len(self.alist), len(self.clist), len(self.Q_names)])
                #self.Q = np.zeros([self.Ntimes, self.Ncopies, len(self.Q_names)])
                self.Q = np.zeros([self.Ntimes, self.Ncopies, 1,1,1, len(self.Q_names)])
                self.Q_cols = find_columns(cols, self.Q_names)
                self.V = thermo[0, find_columns(cols, ['Volume'])[0]]
                self.T = thermo[0, find_columns(cols, ['Temp'])[0]]
                self.beta = 1./8.617e-5/self.T

            for qc, q in enumerate(self.Q_cols):
                if self.Q_names[qc] == 'PotEng':
                    self.PEcol = qc
                self.Q[:, copy, 0,0,0, qc] = thermo[:,q]
                #self.Q[:,copy,qc] = thermo[:,q]

            # TODO: if there are issues, fill with NaN to ignore
            # Works okay while jobs running IF the first copy stays ahead
            
        newshape = (self.Ntimes*self.Ncopies, len(self.Q_names))
        Qavg = self.Q.reshape(newshape)
        self.Qavg = np.mean(Qavg, axis=0)
        self.Qstd = np.std(Qavg, axis=0)


    def write_FD(self, qc):
        txt = ("FunUQ v0.3, 2018; Sam Reeve\n"
               "Functional derivative for {} in {} with {} at {}K with {} atoms; "
               "Description: {}\n"
               .format(self.Q_names[qc], self.units[qc], self.pot_name, self.T, self.Natoms, self.description))
        for r, fd in zip(self.rlist, self.funcder[:,qc]):
            txt += "{} {}\n".format(r, fd)

        name = '{}_{}'.format(self.Q_names[qc], self.FDname)
        with open(os.path.join(self.resultsdir, name), 'w') as f:
            f.write(txt)


    def read_FD(self, qc):
        data = np.loadtxt(os.path.join(self.resultsdir, self.FDname), skiprows=2)
        self.rlist = data[:,0]
        self.funcder = data[:,1:]


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
        plt.ylabel("Functional Derivative ({}/$\AA$)".format(self.units[qc]), fontsize='large')
        plt.show()

    
    def calc_FD(self):
        #self.alist = np.append(self.alist, 0)

        self.funcder = np.zeros([len(self.rlist), len(self.Q_names)])
        for qc, q in enumerate(self.Q_names):
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

                    if self.showA:
                        plt.scatter(self.alist, self.Qperturbed[r0c, :, c0c, qc])
                        plt.plot(self.alist, m[0]*np.array(self.alist) + m[1])
                        plt.show()


            if q == 'PotEng':
                self.funcder[:,qc] = self.funcder[:,qc]/self.Natoms
            elif q == 'Press':
                self.funcder[:,qc] = self.funcder[:,qc]*0.0001

        print("Calculated functional derivative")


    def discrepancy(self):
        self.discrep = self.mainpot.PE - self.correctpot.PE
        print("Calculated discrepancy")


    def correct(self):
        self.correction = magic()
        print("Calculated correction terms")



class FunUQ_bruteforce(FunUQ):
    '''
    Subclass for bruteforce FunUQ: running direct simulations with perturbed potentials
    '''
    def __init__(self, *args, **kwargs):
        super(FunUQ_bruteforce, self).__init__(*args, **kwargs)
        
        self.rundir = os.path.join(self.maindir, 'bruteforce_runs/')
        try:
            os.mkdir(self.rundir)
        except: pass

        self.FDname = 'bruteforce.funcder'

    def run_lammps_bruteforce(self):
    
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

                    out = '{}_{}_{}_{}'.format(self.mainpot.pot, r0, a0, c0)
                    potfile = '{}.table'.format(out)
                    R, PE, style, coeff = create_table(self.mainpot.pot, self.mainpot.parampath,
                                                       #self.pot, self.parampath,
                                                       r0=r0, a0=a0, c0=c0, 
                                                       outname=potfile, savedir=adir)
                    
                    # This is to fix data path IF ../ is there (as it should be for multiple copies
                    replace_in = {'../':'../../../../', 
                                  'TABLECOEFF':coeff, 'TABLESTYLE':style, 'SEED':'0'}
                    replace_sub = {'NAME': out, 'INFILE': self.infile}
                    copy_files(self.initdir, self.rundir)
                    #shutil.copy(potfile, adir)

                    for copy in range(self.Ncopies):
                        copydir = os.path.join(adir, self.copy_folder+str(copy))
                        replace_in['SEED'] = str(int(random()*100000))
                        #replace_sub['NAME'] = '{}_{}'.format(self.pot,copy)
                        intxt = replace_template(self.intxt, replace_in)
                        subtxt = replace_template(self.subtxt, replace_sub)
                        submit_lammps(self.subfile, subtxt,
                                      self.infile, intxt, copydir)


    def extract_lammps(self):
        first = True
        for c0c, c0 in enumerate(self.clist):
            for r0c, r0 in enumerate(self.rlist):
                for a0c, a0 in enumerate(self.alist):
                    for copy in range(self.Ncopies):
                        logfile = os.path.join(self.rundir,
                                               'c{}'.format(c0), 'r{}'.format(r0), 'a{}'.format(a0),
                                               self.copy_folder+str(copy), self.logfile)
                        cols, thermo, Natoms = read_thermo(logfile)

                        if first:
                            first = False
                            self.Natoms = Natoms
                            self.Ntimes = np.shape(thermo)[0]
                            #self.times = np.array(sample(range(startsteps, totalsteps), self.Ntimes))
                            self.Qperturbed = np.zeros([self.Ntimes, self.Ncopies, len(self.rlist),
                                                    len(self.alist), len(self.clist), len(self.Q_names)])
                            self.Q_cols = find_columns(cols, self.Q_names)
                            self.T = thermo[0, find_columns(cols, ['Temp'])[0]]

 
                        for qc, q in enumerate(self.Q_cols):
                            self.Qperturbed[:, copy, r0c, a0c, c0c, qc] = thermo[:,q]
                            
                print("Extracted thermo data for position {}".format(r0))


        newshape = (self.Ntimes*self.Ncopies, len(self.rlist), len(self.alist),
                    len(self.clist), len(self.Q_names))
        Qperturbed = self.Qperturbed.reshape(newshape)
        self.Qperturbed = np.mean(Qperturbed, axis=0)


    def prepare_FD(self):
        self.extract_lammps()
        


class FunUQ_perturbative(FunUQ):
    '''
    Subclass for perturbative FunUQ: running ONLY unperturbed simulation and using exponential weighting
    '''
    def __init__(self, *args, **kwargs):
        super(FunUQ_perturbative, self).__init__(*args, **kwargs)

        self.FDname = 'perturbative.funcder'


    # Only used for coord
    def get_perturbs(self):
        #self.gauss = np.zeros([len(self.rlist), len(self.alist), len(self.clist), self.rdfbins])
        #self.gaussdiff = np.zeros([len(self.rlist), len(self.alist), len(self.clist), self.rdfbins])
        self.gauss = np.zeros([1, 1, len(self.rlist), len(self.alist), len(self.clist), 1, self.rdfbins])
        self.gaussdiff = np.zeros([1, 1, len(self.rlist), len(self.alist), len(self.clist), 1, self.rdfbins])
        self.rdf_R = np.linspace(self.rmin, self.rmax, self.rdfbins)
        for r0c, r0 in enumerate(self.rlist):
            for a0c, a0 in enumerate(self.alist):
                for c0c, c0 in enumerate(self.clist):
                    #self.gauss[r0c,a0c,c0c,:] = (1/(np.sqrt(2*np.pi)*c0)*a0*
                    #                             np.exp(-(self.rdf_R-r0)**2/(2*c0**2)))
                    #self.gaussdiff[r0c,a0c,c0c,:] = ((self.rdf_R-r0)/(np.sqrt(2*np.pi)*c0**3)*a0*
                    #                                 np.exp(-(self.rdf_R-r0)**2/(2*c0**2)))
                    self.gauss[0,0,r0c,a0c,c0c,0,:] = (1/(np.sqrt(2*np.pi)*c0)*a0*
                                                       np.exp(-(self.rdf_R-r0)**2/(2*c0**2)))
                    self.gaussdiff[0,0,r0c,a0c,c0c,0,:] = ((self.rdf_R-r0)/(np.sqrt(2*np.pi)*c0**3)*a0*
                                                           np.exp(-(self.rdf_R-r0)**2/(2*c0**2)))


    def exp_weighting(self):
        newshape = (self.Ntimes*self.Ncopies, len(self.rlist), len(self.alist), len(self.clist))

        # Get PE for weighting 
        Hprime = self.Qprime[:,:,:,:,:,self.PEcol]
        tmp = Hprime.reshape(newshape)
        Hprime_avg = np.mean(tmp, axis=0)

        # Add unperturbed QoI
        Qprime = self.Qprime + self.Q

        # Average the exponential
        expweights = np.exp(-self.beta*(Hprime-Hprime_avg))
        tmp = expweights.reshape(newshape)
        denominator = np.mean(tmp, axis=0)

        # Average the QoI*exponential product
        self.Qperturbed = np.zeros([len(self.rlist), len(self.alist), len(self.clist), len(self.Q_names)])
        for qc, q in enumerate(self.Q_names):
            numerator = expweights*Qprime[:,:,:,:,:,qc]
            tmp = numerator.reshape(newshape)
            numerator = np.mean(tmp, axis=0)
            self.Qperturbed[:,:,:,qc] = numerator/denominator



class FunUQ_perturb_allatom(FunUQ_perturbative):
    '''
    Subclass for perturbative FunUQ: using LAMMPS dump and rerun to get perturbation contributions
    '''
    def __init__(self, *args, **kwargs):
        super(FunUQ_perturb_allatom, self).__init__(*args, **kwargs)

        self.FDname = 'perturbative_allatom.funcder'

        self.reruntemplate = 'rerun.template'
        #self.rerunfile = 'in.rerun_gauss'
        self.rerunpath = os.path.join(self.initdir, self.reruntemplate)
        self.reruntxt = read_template(self.rerunpath)

        self.rerunpotdir = os.path.join(self.rundir, 'gauss')

        self.rerunsubtemplate = 'submit_rerun.template'
        self.rerunsubpath = os.path.join(self.initdir, self.rerunsubtemplate)
        self.rerunsubtxt = read_template(self.rerunsubpath)

        #super(FunUQ_perturb_allatom, self).__init__(*args, **kwargs)


    def rerun_gauss(self):
        try:
            os.mkdir(self.rerunpotdir)
        except: pass
        
        for copy in range(self.Ncopies):
            rerundir = os.path.join(self.rundir, self.copy_folder+str(copy), 'rerun_gauss/')
            try:
                os.mkdir(rerundir)
            except: pass

            for c0c, c0 in enumerate(self.clist):
                for r0c, r0 in enumerate(self.rlist):
                    for a0c, a0 in enumerate(self.alist):

                        out = 'gauss_{}_{}_{}'.format(r0, a0, c0)
                        outname = '{}.log'.format(out)
                        inname = 'in.{}'.format(out)
                        if not copy:
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
        self.Qprime = np.zeros([self.Ntimes, self.Ncopies, len(self.rlist),
                                len(self.alist), len(self.clist), len(self.Q_names)])

        for c0c, c0 in enumerate(self.clist):
            for r0c, r0 in enumerate(self.rlist):
                for a0c, a0 in enumerate(self.alist):
                    for copy in range(self.Ncopies):
                        logfile = os.path.join(self.rundir, self.copy_folder+str(copy),
                                               'rerun_gauss/', 'gauss_{}_{}_{}.log'.format(r0, a0, c0))
                        print logfile
                        cols, thermo, Natoms = read_thermo(logfile)
                        for qc, q in enumerate(self.Q_cols):
                            self.Qprime[:, copy, r0c, a0c, c0c, qc] = thermo[:,q]

                print("Extracted thermo data for position {}".format(r0))


    def prepare_FD(self):
        #self.rerun_gauss()
        self.extract_lammps()
        print("Extracted thermo data")

        self.extract_lammps_gauss()
        print("Calculated perturbations")

        #self.calc_Qprime()
        #print("Calculated perturbation Hamiltonian contribution")
        self.exp_weighting()
        print("Calculated exponential weights")




class FunUQ_perturb_coord(FunUQ_perturbative):
    '''
    Subclass for perturbative FunUQ: using RDF (coordination) to get perturbation contributions
    '''
    def __init__(self, *args, **kwargs):
        super(FunUQ_perturb_coord, self).__init__(*args, **kwargs)
        
        self.FDname = 'perturbative_coord.funcder'


    def extract_rdf(self):
        bins = self.rdfbins

        for copy in range(self.Ncopies):
            rdffile = os.path.join(self.rundir, self.copy_folder+str(copy), self.rdffile)

            rdf_slice = np.zeros([self.Ntimes, 2], dtype=int)
            rdf_slice[:,0] = np.arange(4, self.Ntimes*(bins+1)+4, bins+1)
            rdf_slice[:,1] = np.arange(4+bins, self.Ntimes*(bins+1)+4+bins, bins+1)
            with open(rdffile) as f:
                rdf_raw = f.readlines()
                
            #self.coord = np.zeros([self.Ntimes, self.Ncopies, bins])
            self.coord = np.zeros([self.Ntimes, self.Ncopies,1,1,1,1,bins])
            for tc, t in enumerate(range(self.Ntimes)):
                coord_cum = np.zeros([bins])
                coord_cum = np.array(np.array(map(methodcaller('split'), rdf_raw[rdf_slice[t,0]:rdf_slice[t,1]]), dtype=float)[:,3])
                for c in range(bins-1, 0, -1):
                    #self.coord[tc, copy, c] = coord_cum[c] - coord_cum[c-1]
                    self.coord[tc, copy, 0,0,0,0, c] = coord_cum[c] - coord_cum[c-1]


            print("Extracted RDF data for copy {}".format(copy))


    def calc_Qprime(self):
        '''
        for qc, q in enumerate(self.Q_names):
            for tc in range(self.Ntimes):
                for copy in range(self.Ncopies):
                    for r0c, r0 in enumerate(self.rlist):
                        for a0c, a0 in enumerate(self.alist):
                            for c0c, c0 in enumerate(self.clist):
                                if q == 'PotEng':
                                    self.Qprime[tc,copy,r0c,a0c,c0c,qc] = np.sum(self.gauss[r0c,a0c,c0c, :]*
                                                                                 self.coord[tc, copy, :])
                                elif q == 'Press':
                                    self.Qprime[tc,copy,r0c,a0c,c0c,qc] = np.sum(self.gaussdiff[r0c,a0c,c0c, :]*
                                                                                  self.coord[tc, copy, :]*self.rdf_R)
        '''
        for qc, q in enumerate(self.Q_names):
            if q == 'PotEng':
                self.Qprime[:,:,:,:,:,qc] = np.sum(self.gauss[:,:,:,:,:,0,:]*
                                                   self.coord[:,:,:,:,:,0,:], axis=-1)
                self.Qprime[:,:,:,:,:,qc] = self.Qprime[:,:,:,:,:,qc]*self.Natoms/2.
            elif q == 'Press':
                self.Qprime[:,:,:,:,:,qc] = np.sum(self.gaussdiff[:,:,:,:,:,0,:]*
                                                   self.coord[:,:,:,:,:,0,:]*self.rdf_R, axis=-1)
                self.Qprime[:,:,:,:,:,qc] = (self.Qprime[:,:,:,:,:,qc]/3./self.V)*160.2176*10000.*self.Natoms/2.


    def prepare_FD(self):
        self.get_perturbs()
        print("Calculated perturbations")

        self.extract_lammps()
        print("Extracted thermo data")
        self.extract_rdf()
        print("Extracted RDF data")

        self.calc_Qprime()
        print("Calculated perturbation Hamiltonian contribution")
        self.exp_weighting()
        print("Calculated exponential weights")


        
class FUQexception(Exception):
    pass



def read_template(fname):
    tname = glob(fname)
    if not tname:
        raise FUQexception("Could not find template file: {}".format(fname))
    else:
        with open(tname[0]) as f:
            txt = f.read()
    return txt


def replace_template(txt, rdict):
    for key, val in rdict.iteritems():
        txt = txt.replace(key, val)
    return txt


def copy_files(src, dest):
    files = os.listdir(src)
    for fname in files:
        srcpath = os.path.join(src, fname)
        #destpath = os.path.join(dest, fname)
        if (os.path.isfile(srcpath)):
            shutil.copy(srcpath, dest)


def write_file(txt, dest):
    with open(dest, 'w') as f:
        f.write(txt)


'''
def rerun_lammps(infile, intxt, name, fold):
    codedir = os.getcwd()
    os.chdir(fold)

    with open(infile, 'w') as f:
        f.write(intxt)

    subprocess.call(['lmp', '-i', infile])
    shutil.move('log.lammps', name)
    os.chdir(codedir)
'''

def submit_lammps(subfile, subtxt, infile, intxt, fold):
    '''
    Submit LAMMPS simulation given a PBS file and input file
    '''

    codedir = os.getcwd()
    try:
        os.mkdir(fold)
        print(fold)
    except:
        print("{} is being overwritten".format(fold))

    os.chdir(fold)
    write_file(subtxt, subfile)
    write_file(intxt, infile)

    subprocess.call(['qsub', subfile])
    os.chdir(codedir)

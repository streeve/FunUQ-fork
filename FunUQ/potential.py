# FunUQ v0.1, 2018; Sam Reeve; Strachan Research Group
# https://github.rcac.purdue.edu/StrachanGroup

import sys, os, numpy as np, shutil, subprocess
from matplotlib import pyplot as plt


class Potential(object):
    '''
    Simple potential class
    Supports creation of pair potential through LAMMPS pair_style table
    TODO: Allow direct LAMMPS styles
          Bool for table or LAMMPS, using either "create" or "write"
          Each should use copy and accept all kwargs currently in "create_table"
    '''
    def __init__(self, pot, create=True, paramfile=None, paramdir=None,
                 potfile=None, key=None, 
                 N=8000, rmin=0.001, rmax=8.0, cut=6.,
                 r0=1.0, c0=0.0, a0=0.0):

        self.potname = pot
        self.paramdir = paramdir
        self.rmin = rmin
        self.rmax = rmax
        self.N = N
        self.cut = cut

        if paramfile == None and create:
            self.paramfile = '{}.params'.format(self.potname)
        else: 
            self.paramfile = paramfile

        if self.paramdir != None and create:
            self.parampath = os.path.join(self.paramdir, self.paramfile)
        else:
            self.parampath = self.paramfile

        if potfile == None and create:
            self.potfile = '{}.table'.format(self.potname)
        else: 
            self.potfile = potfile
        self.potpath = os.path.join(self.paramdir, self.potfile)


        if key == None:
            self.key = "{}_1".format(self.potname)
        else:
            self.key = key

        self.head = ("# pair_style table for LAMMPS; Created by FunUQ Python;"
                     " Sam Reeve; potential: {}; \n\n"
                     "{}\n"
                     "N {}\n\n").format(self.potname, self.key, N)

        if create:
            self.create_table()
        else:
            R, PE, key = read_file(self.potpath)
            self.R = R
            self.PE = PE
            self.key = key

        self.pairstyle = "pair_style table linear {}".format(N)
        # Use rmax instead of cut so that the RDF goes out beyond the cutoff
        self.paircoeff = "pair_coeff * * {} {} {}".format(self.potpath, self.key, self.rmax)


    # Need to copy later
    def copy(self, initdir, potdir):
        self.potdir = potdir
        potfile = os.path.join(initdir, self.potname+'.table')
        shutil.copy(potfile, potdir)

        self.paircoeff = "pair_coeff * * {} {} {}".format(potdir, self.key, self.rmax)

        R, PE, key = read_file(potfile)
        self.R = R
        self.PE = PE


    def plot(self, ax=None, color='navy', label=None, unit=''):
        if ax == None:
            fig1, ax = plt.subplots(1,1)
        if label == None:
            label = self.potname

        ax.plot(self.R, self.PE, c=color, linewidth=4, label=label)
        ax.set_ylim([-0.2, 0.2])
        ax.set_xlim([np.min(self.R), np.max(self.R)])
        ax.set_ylabel("Potential Energy{}".format(unit), fontsize='large')
        ax.set_xlabel("Position ($\AA$)", fontsize='large')
        ax.legend(loc='best')
        #ax.set_xlim([1.5, 6.5])
        #ax.set_ylim([-0.17, 0.05])
        plt.show()
        return ax 


    def create_table(self, smooth='4th', smooth_w=1.5,
                     r0=1.0, c0=0.0, a0=0.0, 
                     keep=True, write=True,
                     only_gauss=False,
                     savepath=None):
        '''
        Creates LAMMPS "pair_style table" file
        
        Current options:
        style: LJ, Morse
        smooth: 4th order multiplicative smoothing function
        
        N: Number of points
        minR/maxR: Limits of distance tabulation
        r0/a0/c0: Gaussian perturbations for FunUQ
        
        To be used with LAMMPS "metal" (eV and Angstrom)
        '''
        
        if savepath == None:
            savepath = self.potpath
        
        R = np.linspace(self.rmin, self.rmax, self.N)
        cut = self.cut
        cut_id = np.where(R > cut)[0][0]


        if not only_gauss:
            params = read_params(self.parampath)

            if self.potname == 'lj':
                epsilon, sigma = params

                F = 24*epsilon/sigma*(2*(sigma/R)**13 - (sigma/R)**7)
                PE = 4*epsilon*((sigma/R)**12 - (sigma/R)**6)
            elif self.potname == 'morse':
                D0, alpha, r0_M = params
                
                F = 2*alpha*D0*(np.exp(-2*alpha*(R - r0_M)) - np.exp(-alpha*(R - r0_M)))
                PE = D0*(np.exp(-2*alpha*(R - r0_M)) - 2*np.exp(-alpha*(R - r0_M)))
                
            elif self.potname == 'exp6':
                epsilon, alpha, r0_E6 = params
                
                F = 6*epsilon/(1.-6./alpha)/r0_E6*(np.exp(alpha*(1 - R/r0_E6)) - (r0_E6/R)**7)
                PE = epsilon/(1.-6./alpha)*(6./alpha*np.exp(alpha*(1 - R/r0_E6)) - (r0_E6/R)**6)
                
            elif self.potname == 'fake_exp6':
                epsilon, alpha, r0_E6 = params
                
                F = 2*alpha*epsilon*(np.exp(-2*alpha*(R - r0_E6)) - (r0_E6/R)**7)
                PE = epsilon*(np.exp(-2*alpha*(R - r0_E6)) - 2*(r0_E6/R)**6)
                
            else:
                raise TableException("Style not supported.\n\n")
            

            if smooth == '4th':
                smooth = ((R - cut)/smooth_w)**4/(1 + ((R - cut)/smooth_w)**4)
                smoothdiff = (4*smooth_w**4*(cut-R)**3)/(smooth_w**4+(cut-R)**4)**2
                
                F = PE*smoothdiff + F*smooth
                PE = PE*smooth
                
                F[cut_id:] = np.zeros(np.shape(R[cut_id:])) # shift
                PE[cut_id:] = np.zeros(np.shape(R[cut_id:])) # shift
  
        else:
            F = np.zeros([self.N])
            PE = np.zeros([self.N])


        if abs(a0) > 0.0:
            # add gauss after smoothing
            gaussdiff = (R-r0)/(np.sqrt(2*np.pi)*c0**3) * a0*np.exp(-(R - r0)**2/(2*c0**2)) ### sign
            F += gaussdiff
            gauss = 1/(np.sqrt(2*np.pi)*c0) * a0*np.exp(-(R - r0)**2/(2*c0**2))
            PE += gauss

        if write:
            write_file(savepath, self.head, self.N, R, PE, F)

        if keep:
            self.PE = PE
            self.R = R




class TableException(Exception):
    pass



def write_file(fname, txt, N, R, PE, F):
    for n, r, p, f in zip(range(1,N+1), R, PE, F):
        txt += "{} {} {} {}\n".format(n, r, p, f)

    with open(fname, 'w') as f:
        f.write(txt)


def read_file(fname): 
    data = np.loadtxt(fname, skiprows=4)
    R = data[:,1]
    PE = data[:,2]

    with open(fname) as f:
        for k in range(3):
            key = f.readline()
    key = key.strip()

    return R, PE, key

                   
def read_params(fname):
    '''
    Read file: (ONE comment line), followed by parameters
    '''
    with open(fname) as f:
        params = f.readlines()
    params = map(float, params[1:])
    return params


if __name__ == '__main__':
    
    create_table(sys.argv[1], sys.argv[2], r0=float(sys.argv[3]), a0=float(sys.argv[4]), c0=float(sys.argv[5]))

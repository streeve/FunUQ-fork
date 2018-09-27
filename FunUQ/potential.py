# FunUQ v0.4, 2018; Sam Reeve; Strachan Research Group
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
                 potfile=None, potdir='', 
                 N=8000, rlo=0.001, rhi=8.0, cut=6.,
                 r0=1.0, c0=0.0, a0=0.0):

        self.potname = pot

        # Here, paramfile is only the coeff for the table creation function below
        if paramfile == None and create:
            self.paramfile = '{}.params'.format(self.potname)
        else: 
            self.paramfile = paramfile

        if paramdir != None and create:
            self.parampath = os.path.join(paramdir, self.paramfile)
        else:
            self.parampath = self.paramfile

        if potfile == None and create:
            self.potfile = '{}.table'.format(self.potname)
        else: 
            self.potfile = potfile
        self.potpath = os.path.join(paramdir, self.potfile)

        if create:
            self.create_table(N, rlo, rhi, cut) #self.potname, self.parampath) #, savedir=potdir)
        else:
            #subprocess.call(['lmp', '-i', self.parampath]) #TODO REMOVE
            R, PE, style, coeff = read_file(self.potpath)
            self.R = R
            self.PE = PE
            self.pairstyle = style
            self.paircoeff = coeff


    # Need to copy later
    def copy(self, initdir, potdir):
        self.potdir = potdir
        potfile = os.path.join(initdir, self.potname+'.table')
        shutil.copy(potfile, potdir)
        R, PE, style, coeff = read_file(potfile)
        self.R = R
        self.PE = PE
        self.pairstyle = style
        self.paircoeff = coeff


    def plot(self):
        plt.scatter(self.R, self.PE, c='navy')
        plt.ylim([-0.2, 0.2])
        plt.xlim([2, 12])
        plt.show()



    def create_table(self, N, rlo, rhi, cut, 
                     potname=None, smooth='4th', 
                     r0=1.0, c0=0.0, a0=0.0, 
                     smooth_w=1.5, 
                     outname=None, savedir=None):
        '''
        Creates LAMMPS "pair_style table" file
        
        Current options:
        style: LJ, Morse
        smooth: 4th order multiplicative smoothing function
        
        N: Number of points
        minR/maxR: Limits of distance tabulatec
        r0/a0/c0: Gaussian perturbations for FunUQ
        
        To be used with LAMMPS "metal" (eV and Angstrom)
        '''
        
        if potname == None: potname = self.potname
        if outname == None: outname = self.potfile
        if savedir == None: savedir = self.potpath
        #if not outname:
        #    outname = '{}.table'.format(style)
    
        key = "{}_1".format(potname)
        head = ("# pair_style table for LAMMPS; Created by {};"
                " Sam Reeve; potential: {}; smoothing: {} \n\n"
                "{}\n"
                "N {}\n\n").format(sys.argv[0], potname, smooth, key, N)
        self.pairstyle = "pair_style table linear {}".format(N)
        self.paircoeff = "pair_coeff * * ../{} {} {}".format(outname, key, cut)

        R = np.linspace(rlo, rhi, N)
        cut_id = np.where(R > cut)[0][0]


        if potname != 'gauss':
            params = read_params(self.parampath)

        if potname == 'lj':
            epsilon, sigma = params

            F = 24*epsilon/sigma*(2*(sigma/R)**13 - (sigma/R)**7)
            PE = 4*epsilon*((sigma/R)**12 - (sigma/R)**6)
        elif potname == 'morse':
            D0, alpha, r0_M = params

            F = 2*alpha*D0*(np.exp(-2*alpha*(R - r0_M)) - np.exp(-alpha*(R - r0_M)))
            PE = D0*(np.exp(-2*alpha*(R - r0_M)) - 2*np.exp(-alpha*(R - r0_M)))

        elif potname == 'exp6':
            epsilon, alpha, r0_E6 = params
        
            F = 6*epsilon/(1.-6./alpha)/r0_E6*(np.exp(alpha*(1 - R/r0_E6)) - (r0_E6/R)**7)
            PE = epsilon/(1.-6./alpha)*(6./alpha*np.exp(alpha*(1 - R/r0_E6)) - (r0_E6/R)**6)

        elif potname == 'fake_exp6':
            epsilon, alpha, r0_E6 = params
        
            F = 2*alpha*epsilon*(np.exp(-2*alpha*(R - r0_E6)) - (r0_E6/R)**7)
            PE = epsilon*(np.exp(-2*alpha*(R - r0_E6)) - 2*(r0_E6/R)**6)
        
        elif potname == 'sin_test':
            epsilon, sigma = params

            F = 24*epsilon/sigma*(2*(sigma/R)**13 - (sigma/R)**7) -0.0782*np.cos(0.17*(24.2+R))
            PE = 4*epsilon*((sigma/R)**12 - (sigma/R)**6) +0.44+0.49*np.sin(0.19*(R+24.))

        elif potname == 'gauss':
            F = np.zeros([N])
            PE = np.zeros([N])
            
        else:
            raise TableException("Style not supported.\n\n")
            exit()


        if smooth == '4th':
            smooth = ((R - cut)/smooth_w)**4/(1 + ((R - cut)/smooth_w)**4)
            smoothdiff = (4*smooth_w**4*(cut-R)**3)/(smooth_w**4+(cut-R)**4)**2

            F = PE*smoothdiff + F*smooth
            PE = PE*smooth

            F[cut_id:] = np.zeros(np.shape(R[cut_id:])) # shift
            PE[cut_id:] = np.zeros(np.shape(R[cut_id:])) # shift
  

        if abs(a0) > 0.0:
            # add gauss after smoothing
            gaussdiff = (R-r0)/(np.sqrt(2*np.pi)*c0**3) * a0*np.exp(-(R - r0)**2/(2*c0**2)) ### sign
            F += gaussdiff
            gauss = 1/(np.sqrt(2*np.pi)*c0) * a0*np.exp(-(R - r0)**2/(2*c0**2))
            PE += gauss


        write_file(savedir, head, N, R, PE, F)
        self.PE = PE
        self.R = R

        return



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
    style = ''
    coeff = ''
    return R, PE, style, coeff

                   
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

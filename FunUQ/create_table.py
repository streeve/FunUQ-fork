import sys, os, numpy as np


class TableException(Exception):
    pass


def create_table(style, potparams, smooth='4th', N=8000, 
                 rlo=0.001, rhi=8.0, cut=6.0,
                 r0=1.0, c0=0.0, a0=0.0, 
                 smooth_w=1.5, 
                 outname='', savedir=''):
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

    if not outname:
        outname = '{}.table'.format(style)
    
    key = "{}_1".format(style)
    head = ("# pair_style table for LAMMPS; Created by {};"
            " Sam Reeve; potential: {}; smoothing: {} \n\n"
            "{}\n"
            "N {}\n\n").format(sys.argv[0], style, smooth, key, N)
    pairstyle = "pair_style table linear {}".format(N)
    paircoeff = "pair_coeff * * ../{} {} {}".format(outname, key, cut)

    R = np.linspace(rlo, rhi, N)
    cut_id = np.where(R > cut)[0][0]

    if smooth == '4th':
        smooth = ((R - cut)/smooth_w)**4/(1 + ((R - cut)/smooth_w)**4)
        smoothdiff = (4*smooth_w**4*(cut-R)**3)/(smooth_w**4+(cut-R)**4)**2

        if style != 'gauss':
            params = read_params(potparams)

        if style == 'lj':
            epsilon, sigma = params

            F = 24*epsilon/sigma*(2*(sigma/R)**13 - (sigma/R)**7)
            PE = 4*epsilon*((sigma/R)**12 - (sigma/R)**6)
        elif style == 'morse':
            D0, alpha, r0_M = params

            F = 2*alpha*D0*(np.exp(-2*alpha*(R - r0_M)) - np.exp(-alpha*(R - r0_M)))
            PE = D0*(np.exp(-2*alpha*(R - r0_M)) - 2*np.exp(-alpha*(R - r0_M)))

        elif style == 'buck':
            epsilon, alpha, r0_B = params

            F = 12*epsilon*(1/r0_B*np.exp(12*(1 - R/r0_B)) - (r0_B/R)**7)
            PE = epsilon*(np.exp(12*(1 - R/r0_B)) - 2*(r0_B/R)**6)

        elif style == 'gauss':
            F = np.zeros([N])
            PE = np.zeros([N])
        else:
            raise TableException("Style not supported.\n\n")
            exit()

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


    write_file(os.path.join(savedir, outname), head, N, R, PE, F)

    return R, PE, pairstyle, paircoeff



def write_file(fname, txt, N, R, PE, F):
    for n, r, p, f in zip(range(1,N+1), R, PE, F):
        txt += "{} {} {} {}\n".format(n, r, p, f)

    with open(fname, 'w') as f:
        f.write(txt)
                   
def read_params(fname):
    '''
    Read file: (ONE comment line), followed by parameters
    '''
    with open(fname) as f:
        params = f.readlines()
    params = map(float, params[1:])
    return params


if __name__ == '__main__':
    
    create_table(sys.argv[1], sys.argv[2], r0=sys.argv[3], a0=sys.argv[4], c0=sys.argv[5])

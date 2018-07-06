# FunUQ v0.3, 2018; Sam Reeve; Strachan Research Group
# https://github.rcac.purdue.edu/StrachanGroup

# import general
import sys, os, subprocess, shutil, numpy as np
from random import random; from glob import glob


class FUQerror(Exception):
    pass



def is_thermo(q):
    return q in ['Temp', 'Press', 'PotEng', 'KinEng', 'TotEng', 'Enthalpy',
                    'Evdwl', 'Ecoul', 'Epair', 'Ebond', 'Eangle', 'Edihed', 'Eimp',
                    'Emol', 'Elong', 'Etail',
                    'Volume', 'Density', 'Lx', 'Ly', 'Lz',
                    'Xy', 'Xz', 'Yz', 'xlat', 'ylat', 'zlat',
                    'Pxx', 'Pyy', 'Pzz', 'Pxy', 'Pxz', 'Pyz',
                    'Cella', 'Cellb', 'Cellc', 'Cellalpha', 'Cellbeta', 'Cellgamma']


def is_fluct(q):
    return q in ['HeatCapacityVol', 'HeatCapacityPress', 'Compressibility', 'ThermalExpansion']


def read_template(fname):
    tname = glob(fname)
    if not tname:
        raise FUQerror("Could not find template file: {}".format(fname))
    else:
        with open(tname[0]) as f:
            txt = f.read()
    return txt


def replace_template(txt, rdict):
    for key, val in rdict.items():
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


def submit_lammps(subfile, subtxt, infile, intxt, fold):
    '''
    Submit LAMMPS simulation given a PBS file and input file
    '''

    codedir = os.getcwd()
    try:
        os.mkdir(fold)
    except:
        print("{} is being overwritten".format(fold))

    os.chdir(fold)
    write_file(subtxt, subfile)
    write_file(intxt, infile)

    subprocess.call(['qsub', subfile])
    os.chdir(codedir)



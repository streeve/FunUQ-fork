# FunUQ v0.1, 2018; Sam Reeve; Strachan Research Group
# https://github.rcac.purdue.edu/StrachanGroup

# import general
import sys, os, subprocess, shutil, numpy as np, shlex
from random import random; from glob import glob


class FUQerror(Exception):
    pass



def is_thermo(q):
    return q in ['Temp', 'Press', 'PotEng', 'KinEng', 'TotEng', 'Enthalpy',
                 'E_vdwl', 'E_coul', 'E_pair', 'E_bond', 'E_angle', 'E_dihed', 'E_impro',
                 'E_mol', 'E_long', 'E_tail',
                 'Volume', 'Density', 'Lx', 'Ly', 'Lz',
                 'Xy', 'Xz', 'Yz', 'xlat', 'ylat', 'zlat',
                 'Pxx', 'Pyy', 'Pzz', 'Pxy', 'Pxz', 'Pyz',
                 'Cella', 'Cellb', 'Cellc', 'Cellalpha', 'Cellbeta', 'Cellgamma']


def is_fluct(q):
    print("WARNING: fluctionation properties still in dev.")
    return q in ['HeatCapacityVol', 'HeatCapacityPress', 'Compressibility', 'ThermalExpansion']


def read_file(fname):
    tname = glob(fname)
    if not tname:
        raise FUQerror("Could not find file: {}".format(fname))
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


def submit_lammps(fold, subfile=None, subtxt=None, 
                  infile=None, intxt=None, 
                  mode='PBS', resub=False,
                  cores=1, minutes=30):
    '''
    Submit LAMMPS simulation given a PBS file and input file
    '''

    codedir = os.getcwd()
    try:
        os.mkdir(fold)
    except:
        if 'rerun_gauss' not in fold:
            print("{} is being overwritten".format(fold))

    os.chdir(fold)
    if not resub:
        write_file(intxt, infile)

    if mode == 'PBS':
        PBS_submit(subfile, subtxt)
    elif mode == 'nanoHUB_submit':
        nanoHUB_submit(infile, subtxt, 
                       cores, minutes)
    elif mode == 'nanoHUB_local':
        nanoHUB_local(infile)
    elif mode == 'nanoHUB_local_wait':
        nanoHUB_local_wait(infile)

    os.chdir(codedir)


def PBS_submit(subfile, subtxt):
    write_file(subtxt, subfile)
    subprocess.call(['qsub', subfile])
    return 


def nanoHUB_submit(infile, subtxt,
                   cores, minutes):
    # Here, subtxt is extra file args
    command = ("submit -n {} -w {} {} "
                   "lammps-09Dec14-parallel -i {}".format(cores, minutes,
                                                          subtxt, infile))
    subprocess.Popen(shlex.split(command),
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
    return


def nanoHUB_local(infile):
    command = 'lmp_serial -i {}'.format(infile)
    subprocess.Popen(shlex.split(command),
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
    return


def nanoHUB_local_wait(infile):
    command = 'lmp_serial -i {}'.format(infile)
    subprocess.call(shlex.split(command))
    return

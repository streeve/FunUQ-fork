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


def submit_paths(intxt, initdir, potpath):
    extra = ''
    if 'read_data' in intxt:
        dfile = intxt.split('read_data')[-1].split('\n')[0].split('/')[-1].strip()
        extra += ' -i '
        extra += os.path.join(initdir, dfile)
    #elif 'pair_coeff' in intxt: # Always present
    extra += ' -i '
    #if potpath == None:
    extra += potpath
        
    return extra


def submit_lammps(replace_in, infile, intxt, 
                  copy_rundir, copy_list, overwrite,
                  subfile=None, subtxt=None,
                  potpath=None, initdir=None, replace_sub={},
                  mode='PBS', cores=1, minutes=30):
    '''                                                                        
    Run LAMMPS simulation given an input file and optional submission details
    '''
    maxcopy = 0
    if not overwrite:
        existing = glob(copy_rundir.format('*')) #os.path.join(rundir, copy_folder+'*'))
        for x in existing:
            nextcopy = int(x.split(copy_folder)[-1])
            if nextcopy >= maxcopy:
                maxcopy = nextcopy +1
        print(" - NOT overwriting existing - creating new folders")
    else:
        if 'rerun_gauss' not in copy_rundir:
            print(" - Overwriting existing runs")

    ### Doesn't really make sense for copy_start to supercede overwrite=False
    #if copy_start > 0 or maxcopy < copy_start:
    #    maxcopy = copy_start
    copy_list += maxcopy 

    for copy in copy_list:
    #for copy in range(maxcopy, maxcopy+Ncopies):
        #copydir = os.path.join(rundir, copy_folder+str(copy))
        copydir = copy_rundir.format(copy)
        try:
            os.mkdir(rerundir)
        except: pass

        replace_in['SEED'] = str(int(random()*100000))
        intxt_replaced = replace_template(intxt, replace_in)
        if 'PBS' in mode:
            replace_sub['NAME'] += '_{}'.format(copy)
            subtxt = replace_template(subtxt, replace_sub)
        elif 'submit' in mode:
            subtxt = submit_paths(intxt_replaced, initdir, potpath)
        else:
            subtxt = ''
        #submit_lammps(cdir, infile, intxt, 
        #              subfile=subfile, subtxt=subtxt, mode=mode)

        codedir = os.getcwd()
        try:
            os.mkdir(copydir)
            print("Creating {}".format(copydir))
        except:
            if 'rerun_gauss' not in copydir:
                print("{} is being overwritten".format(copydir))

        os.chdir(copydir)
        write_file(intxt_replaced, infile)
        
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
        
    return


def PBS_submit(subfile, subtxt):
    write_file(subtxt, subfile)
    subprocess.call(['qsub', subfile])
    return 


def nanoHUB_submit(infile, subtxt,
                   cores, minutes):
    # Here, subtxt is extra file args
    print(subtxt)
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

#!/usr/bin/python
# parselammps.py 
import sys, os, re, numpy as np
from operator import methodcaller

from .utils import FUQerror


def read_thermo(f, ignore_equil=False, only_last=True, last_half=False, 
                returnval='most'):
    '''
    LAMMPS LOG FILE PARSING
    TODO: option to return everything (atom count, etc.)
    OPTIONS
    ignore_equil: keep all thermo data, except first section # TODO: change to numeric choice
    only_last: keep only last section
    last_half: keep only last half (use first half for equilibration)
    '''

    with open(f, 'r') as f_log:
        loglines = f_log.readlines()
    Nlog = len(loglines)
    equil_flag = 0
    head_list = []; tail_list = []
    for lc, line in enumerate(loglines):
        if 'TotEng' in line: # and equil_flag:
            head_list.append(lc + 1)
            col_log = line.split() # columns should always be same length
        elif 'Loop time ' in line: # and equil_flag:
            tail_list.append(lc)
            Natoms = int(line.split()[-2]) # number of atoms should stay the same

        # If the simulation isn't done, need to find the atom count somewhere else
        elif 'Created ' in line and 'atoms' in line:
            Natoms = int(line.split()[1])
        elif 'atoms' in line and 'new total' in line:
            Natoms = int(line.split()[-1])
        elif 'reading atoms' in line:
            Natoms = int(loglines[lc+1].split()[0])

    if not head_list:
        raise FUQerror("No thermo data (read_thermo)")

    Nhead = len(head_list)
    Ntail = len(tail_list)
    if Nhead != Ntail:
        if Nhead == Ntail + 1:
            tail_list.append(Nlog)
        else:
            raise FUQerror("Parsing error (read_thermo)")

    diff_list = np.array(tail_list, dtype='int32') - np.array(head_list, dtype='int32')

    istart = 0
    if ignore_equil: istart = 1
    if only_last: istart = -1

    if last_half:
        head_list = list(diff_list/2 + np.array(head_list, dtype='int32'))

    head_list = np.array(head_list[istart:])
    tail_list = np.array(tail_list[istart:])
    Nhead = len(head_list)
    Ntail = len(tail_list)
    Ncol = len(col_log)

    Nthermo = int(np.sum(tail_list - head_list))
    thermo = np.zeros([Nthermo, Ncol])
    headarr = []; tailarr = [0]
    for lc, head, tail in zip(range(Nhead), head_list, tail_list):
        headarr.append(tailarr[-1])
        tailarr.append(headarr[-1] + tail_list[lc] - head_list[lc])
        thermo_tmp = list(map(methodcaller('split'), loglines[head:tail]))
        # Mapping from list to array doesn't work if all sublists are not the same length
        if len(list(thermo_tmp)[-1]) < Ncol:
            thermo_tmp = thermo_tmp[:-1]
            thermo = np.delete(thermo, (-1), axis=0)

        thermo[headarr[-1]:tailarr[-1], :] = np.array([[float(x2) for x2 in x1] for x1 in thermo_tmp])

    if returnval == 'everything':
        return col_log, thermo, Natoms, headarr, tailarr[1:] 
    elif returnval == 'most':
        return col_log, thermo, Natoms
    elif returnval == 'thermo':
        return thermo


def find_columns(allcol, findcol):
    '''
    Find columns from LAMMPS log 
    '''

    allcol = np.array(allcol, dtype=object)
    col_ind = [0]*len(findcol)
    for k, col in enumerate(findcol):
        col_ind[k] = int(np.where(allcol == col)[0])

        
    return col_ind


def block_avg(arr, samples, axis=1):
    '''
    Local separate averaging
    By default, ignores trailing information that doesn't fit block size
    '''

    shape = np.shape(arr)
    sets = int(shape[0]/samples)
    mostarr = arr[:sets*samples,:]
    newarr = np.reshape(mostarr, (sets, samples, -1))
    newarr = np.mean(newarr, axis=1)
    
    return newarr


def read_data(f, vel=False, head=False, box=False):
    '''
    LAMMPS DATA PARSING
    Return NatomsxNcol array for positions and, if requested, velocites
    Returns header text or box size, if requested
    '''

    with open(f, 'r') as f_data:
        datatxt = f_data.readlines()

    head = []; tail = []
    for lc, line in enumerate(datatxt):
        if 'Atoms' in line:
            head.append(lc + 2)
            headtxt = ''.join(datatxt[:lc+2])
        elif 'Velocities' in line:
            tail.append(lc - 1)
            head.append(lc + 2)
        elif "xlo xhi" in line:
            box[0,:] = map(float, line.split()[0:2])
        elif "ylo yhi" in line:
            box[1,:] = map(float, line.split()[0:2])
        elif "zlo zhi" in line:
            box[2,:] = map(float, line.split()[0:2])
        elif "xy xz yz" in line:
            tiltbox = float(line.split()[0])

    # No velocities
    if not tail:
        tail.append(len(datatxt))
    if len(tail) < len(head):
        tail.append(len(datatxt))

    # No data
    if not head:
        raise FUQerror("No data (read_thermo)")

    data_tmp = map(methodcaller('split'), datatxt[head[0]:tail[0]])
    dataA = np.array([map(float, x) for x in data_tmp])
    data_tmp = map(methodcaller('split'), datatxt[head[1]:tail[1]])
    dataV = np.array([map(float, x) for x in data_tmp])

    if vel and head and box:
        return dataA, dataV, headtxt, box 
    elif vel and head:
        return dataA, dataV, headtxt 
    elif vel and box:
        return dataA, dataV, box 
    elif head and box:
        return dataA, headtxt, box 
    elif vel:
        return dataA, dataV
    elif head:
        return dataA, headtxt
    elif box:
        return dataA, box 
    else:
        return dataA


# TODO testing
def read_dump(f, style='NxNxN'):
    ''' 
    LAMMPS DUMP PARSING
    OPTIONS
    style
        NxNxN returns NstepsxNatomsxNcol array
        6xN   returns (x y z Vx Vy Vz)xNsteps array
        6N    returns flattened version of 6xN array
    '''

    with open(f, 'r') as f_dump:
        dumptxt = f_dump.readlines()
    # Find total lines, atoms, steps, and variable names in dump file
    # TODO: could also read box for each step
    Ntxt = int(len(dumptxt))
    Natoms = int(dumptxt[3]) # always line 3
    Nsteps = int(len(dumptxt)/(Natoms+8))
    col_dump = dumptxt[8].split()[2:] # always line 8
    Ncol = len(col_dump)

    if style == 'NxNxN':
        atoms = np.zeros([Natoms, Ncol, Nsteps])
    elif style == '6xN':
        atoms = np.zeros([Natoms*Nsteps, 6])
    elif style == '6N':
        atoms = np.zeros([Nsteps,Natoms*len(col_dump)])

    dumparr = np.array(dumptxt, dtype=object)
    for a in range(Nsteps):
        atomlines = [9+(9+Natoms)*a, 9+(9+Natoms)*a+Natoms]
        atoms_tmp = list(dumparr[atomlines[0]:atomlines[1]])
        atoms_tmp = map(methodcaller('split'), atoms_tmp)
        atoms_tmp = [map(float, x) for x in atoms_tmp]
        atoms_tmp = np.array(atoms_tmp)

        if style == 'NxNxN':
            atoms[:,:,a] = np.array(atoms_tmp)
        elif style == '6xN':
            # Need to automate (skips atom ids, etc.)
            atoms[Natoms*a:Natoms*a+Natoms,:] = np.array(atoms_tmp[:,1:]) 
        elif style == '6N':
            atoms[:,a*Ncol:a*Ncol+Ncol] = np.array(atoms_tmp)

    return col_dump, Natoms, atoms


def sort_files(filelist, glob1='.dump', n1=0, glob2='.', n2=-1, reverse=False, zero_first=True):
    '''
    SORTED FILE GLOB
    Input file glob from command; returns numeric sorted list
    OPTIONS
        glob1/2: Strings to split numeric portion of filename (default *.#.dump)
        reverse: Sort low to high
        zero_first: Move initial file to beginning (only used if reversed)
    '''

    N = len(filelist)
    Nlist = range(N)
    filelist = np.array(filelist, dtype=object)
    # sorts by int within dumpfile name
    filenum = np.zeros([N])
    for nf,f in zip(range(N), filelist):
        filenum[nf] = int(f.split(glob1)[n1].split(glob2)[n2])
    ids = np.argsort(filenum)
    # reverses and shifts zero back to begining
    if reverse:
        ids = ids[::-1]
        if zero_first:
            ids = np.roll(ids, 1)
    # set numbers and filelist to sorting above
    filenum = filenum[ids]
    filelist = filelist[ids]


    return filenum, filelist


def write_file0(arr, fname, transpose=False, typ='w'):
    '''
    Write numpy array to file
    Send multiple arrays with np.c_[arr1, arr2]
    OPTIONS
        transpose: Transpose array before save
        typ: pass 'a' to append to file
    '''

    with open(fname, typ) as f:
        if transpose:
            np.savetxt(f, np.transpose(arr))
        else: 
            np.savetxt(f, arr)

    return 


def write_file1(list0, fname, typ='w'):
    '''
    Write list to file
    OPTIONS
        typ: pass 'a' to append to file
    '''

    txt = ''.join(list0)
    with open(fname, typ) as f:
        f.write(txt)

    return 


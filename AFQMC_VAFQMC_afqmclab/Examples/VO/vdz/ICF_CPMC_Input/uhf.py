import numpy as np
import sys
import pyscf
from   pyscf  import scf
import random

def doUHFCalculation(mol=None, conv=1e-12, lindep=1e-8, lvl_shft=None, maxMem=16000, maxCycle=5000, dm_init_guess=None):

    if( dm_init_guess is None):
      uhf = scf.UHF(mol)
    else:
      uhf = scf.UHF(mol).newton()

    uhf.conv_tol   = conv
    uhf.max_memory = maxMem
    uhf.max_cycle  = maxCycle
    if(lvl_shft!=None): uhf.level_shift=lvl_shft
    scf.remove_linear_dep_(uhf,lindep)

    if( dm_init_guess is None):
      uhf.kernel()
    else:
      uhf.kernel(dm_init_guess)
  
    print( 'Total UHF energy = {:26.18e}'.format(uhf.energy_tot()) )
    print('\n')

    return uhf

def writeUHFSD2s(mol=None, uhf=None, canonic=None, filename=None, noise=0.0):

    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2

    wfUp = np.array(uhf.mo_coeff)[0, :, 0:Nup]
    wfUp = np.dot( canonic.XInv, wfUp )
    wfDn = np.array(uhf.mo_coeff)[1, :, 0:Ndn]
    wfDn = np.dot( canonic.XInv, wfDn )

    f = open(filename, 'w')

    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0))

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(canonic.L,Nup))
    for i in range(Nup):
        for j in range(canonic.L):
            f.write( '{:26.18e} {:26.18e} \n'.format( wfUp[j,i]+noise*random.random(),0.0 ) )

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(canonic.L,Ndn))
    for i in range(Ndn):
        for j in range(canonic.L):
            f.write( '{:26.18e} {:26.18e} \n'.format( wfDn[j,i]+noise*random.random(),0.0 ) )
    f.close()

def writeUHFMDCas2s(mol=None, uhf=None, canonic=None, filename=None):

    L = canonic.L
    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2

    wfUp = np.array(uhf.mo_coeff)[0, :, 0:Nup]
    wfUp = np.dot( canonic.XInv, wfUp )
    wfDn = np.array(uhf.mo_coeff)[1, :, 0:Ndn]
    wfDn = np.dot( canonic.XInv, wfDn )

    f = open(filename, 'w')
    f.write( '{:10d} {:10d} {:10d} \n'.format(L, Nup, Ndn) )
    f.write( '{:10d} {:10d} {:10d} {:10d} {:10d}\n'.format(1, 1, 1, Nup, Ndn) )
    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0)) #logw=0.0
    f.write( '{:26.18e} {:26.18e} \n'.format(1.0,0.0) )
    f.write( '{:10d} \n'.format(0) )
    f.write( '{:10d} \n'.format(0) )
    for i in range(Nup): f.write( '{:3d}'.format(i) )
    f.write("\n")
    for i in range(Ndn): f.write( '{:3d}'.format(i) )
    f.write("\n")
    for i in range(Nup):
        for j in range(L):
            f.write( '{:26.18e} {:26.18e} \n'.format(wfUp[j,i],0.0) )
    for i in range(Ndn):
        for j in range(L):
            f.write( '{:26.18e} {:26.18e} \n'.format(wfDn[j,i],0.0) )
    for i in range(Nup):
        for j in range(Nup):
            if(i==j): f.write( '{:26.18e} {:26.18e} \n'.format(1.0,0.0) )
            else: f.write( '{:26.18e} {:26.18e} \n'.format(0.0,0.0) )
    for i in range(Ndn):
        for j in range(Ndn):
            if(i==j): f.write( '{:26.18e} {:26.18e} \n'.format(1.0,0.0) )
            else: f.write( '{:26.18e} {:26.18e} \n'.format(0.0,0.0) )
    f.close()

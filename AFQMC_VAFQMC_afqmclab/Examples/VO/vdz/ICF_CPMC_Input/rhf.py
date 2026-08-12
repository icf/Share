import numpy as np
import sys
import pyscf
from   pyscf  import scf
import random

def doROHFCalculation(basis, el, charge, mol=None):

    m=scf.ROHF(mol)
    dm=m.from_chk(el+basis+str(charge)+".chk")

    print("modified dm up",dm[0].diagonal())
    print("modified dm down",dm[1].diagonal())
    m=scf.newton(m)
    m.max_memory = 50000
    m.chkfile=el+basis+str(charge)+".chk"
    energy=m.kernel(dm)
    m.analyze()

    print( 'Total RHF energy = {:26.18e}'.format(m.energy_tot()) )
    print('\n')

    return m

def writeROHFSD2is(mol=None, rhf=None, canonic=None, filename=None, noise=0.0):

    Nup  = (mol.nelectron+mol.spin)//2
    Ndn  = (mol.nelectron-mol.spin)//2
    maxN = max(Nup, Ndn)

    wf = rhf.mo_coeff[:, 0:maxN]
    wf = np.dot( canonic.XInv, wf )
    f = open(filename, 'w')
    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0))
    f.write('{:26d} \n'.format(Nup ))
    f.write('{:26d} \n'.format(Ndn ))

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(canonic.L,maxN))
    for i in range(maxN):
        for j in range(canonic.L):
            f.write( '{:26.18e} {:26.18e} \n'.format( wf[j,i]+noise*random.random(),0.0 ) )
    f.close()

def writeROHFSD2s(mol=None, rhf=None, canonic=None, filename=None, noise=0.0):

    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2

    wf = rhf.mo_coeff[:, 0:max(Nup, Ndn)]
    wf = np.dot( canonic.XInv, wf )

    for i in range( max(Nup, Ndn) ):
        for j in range(canonic.L):
            wf[j,i] = wf[j,i]+noise*random.random()

    f = open(filename, 'w')

    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0))

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(canonic.L,Nup))
    for i in range(Nup):
        for j in range(canonic.L):
            f.write( '{:26.18e} {:26.18e} \n'.format( wf[j,i],0.0 ) )

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(canonic.L,Ndn))
    for i in range(Ndn):
        for j in range(canonic.L):
            f.write( '{:26.18e} {:26.18e} \n'.format( wf[j,i],0.0 ) )
    f.close()


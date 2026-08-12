import numpy as np
import pyscf
import pyscf.lib
from   pyscf  import gto,dft
import random

def doB3lypCalculation(mol=None, a1=0.8):

    a2 = (1-a1)*0.1
    a3 = 1-a1-a2

    nb=mol.nao_nr()
    rdm=[np.random.rand(nb,nb),np.random.rand(nb,nb)]

    umf = dft.UKS(mol)
    umf.xc = 'HF*{} + {}*LDA + {}*B88, .81*LYP + .19*VWN'.format(a1, a2, a3)
    umf.kernel(rdm)

    if(not umf.converged):
       rdm = umf.make_rdm1()
       umf.diis_space = 20
       diis_start_cycle = 0
       umf.kernel(rdm)

    if(not umf.converged):
       rdm = umf.make_rdm1()
       umf = umf.newton()
       umf.kernel(rdm)

    print( 'Total UMF energy = {:26.18e}'.format(umf.energy_tot()) )

    return umf

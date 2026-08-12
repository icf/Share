import json
import numpy as np
# import pandas as pd
import pyscf
from   pyscf  import gto

def setupMolecule(el, chrg=None,spn=None,basis=None):

    df=json.load(open("trail.json"))

    re={'ScO':1.668,
        'TiO':1.623,
        'VO':1.591,
        'CrO':1.621,
        'MnO':1.648,
        'FeO':1.616,
        'CuO':1.725,
        }
   
    molname=el+'O'
    mol=gto.Mole()
 
    mol.ecp={}
    mol.basis={}
    for e in [el,'O']:
      mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
      mol.basis[e]=gto.basis.parse(df[e][basis])
    mol.charge=chrg
    mol.spin=spn
    mol.max_memory = 50000
    print('spin',molname,mol.spin)
    mol.build(atom="%s 0. 0. 0.; O 0. 0. %g"%(el,re[molname]),verbose=4)

    Enuc    = gto.energy_nuc(mol)
    nbasis  = mol.nao_nr()
    nelec_a = (mol.nelectron+mol.spin)//2
    nelec_b = (mol.nelectron-mol.spin)//2

    print('Molecule [geometry in Bohr]')
    print(el+'O',re[molname])

    print('Nuclear repulsion energy = {:26.18e} '.format(Enuc))

    print('AO basis ',basis)
    basis_label = gto.spheric_labels(mol)
    for index, item in enumerate( basis_label ):
        print( "{:<9} {:<16}".format(index, item) )

    print('charge          {:>9d}'.format(chrg)   )
    print('spin            {:>9d}'.format(spn)    )
    print('orbitals        {:>9d}'.format(nbasis) )
    print('alpha electrons {:>9d}'.format(nelec_a))
    print('beta  electrons {:>9d}'.format(nelec_b))
    print('\n')

    return mol

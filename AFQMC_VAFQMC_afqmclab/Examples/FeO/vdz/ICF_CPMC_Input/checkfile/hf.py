import json
import pymatgen
from pyscf import gto,scf,mcscf, fci,lo
from pyscf.scf import ROHF, UHF,ROKS
import numpy as np
import pandas as pd

df=json.load(open("trail.json"))

spins={'ScO':1,'TiO':2,'VO':3,'CrO':4,'MnO':5,'FeO':4,'CuO':1}
re={'ScO':1.668,
    'TiO':1.623,
    'VO':1.591,
    'CrO':1.621,
    'MnO':1.648,
    'FeO':1.616,
    'CuO':1.725,
    }

nd={'Sc':(1,0),'Ti':(2,0),'V':(3,0),'Cr':(5,0),'Mn':(5,0),'Fe':(5,1),
     'Cu':(5,4) } 

datacsv={}
for nm in ['molecule','charge','method','basis','pseudopotential','totalenergy',
           'totalenergy-stocherr','totalenergy-syserr']:
  datacsv[nm]=[]

for basis in ['vdz','vtz','vqz','v5z']:
  for el in ['Sc','Ti','V','Cr','Mn','Fe','Cu']:
    for charge in [0]: #,1]:
      molname=el+'O'
      mol=gto.Mole()

      mol.ecp={}
      mol.basis={}
      for e in [el,'O']:
        mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
        mol.basis[e]=gto.basis.parse(df[e][basis])
      mol.charge=charge
      mol.spin=spins[molname]
      print('spin',molname,mol.spin)
      mol.build(atom="%s 0. 0. 0.; O 0. 0. %g"%(el,re[molname]),verbose=4)
      m=ROHF(mol)
      dm=m.from_chk("../B3LYP/"+el+basis+str(charge)+".chk")

      print("modified dm up",dm[0].diagonal())
      print("modified dm down",dm[1].diagonal())
      m=scf.newton(m)
      m.chkfile=el+basis+str(charge)+".chk"
      energy=m.kernel(dm)
      m.analyze()

      datacsv['molecule'].append(molname)
      datacsv['charge'].append(charge)
      datacsv['method'].append('HF')
      datacsv['basis'].append(basis)
      datacsv['pseudopotential'].append('trail')
      datacsv['totalenergy'].append(energy)
      datacsv['totalenergy-stocherr'].append(0.0)
      datacsv['totalenergy-syserr'].append(0.0)

      pd.DataFrame(datacsv).to_csv("monoxides.csv",index=False)


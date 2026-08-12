import os
import sys; sys.path.append( os.environ['AFQMCLAB_DIR']+"/scripts/pyscf" )
from molecule import *
from rhf import *
from uhf import *
from model import *
import numpy as np

import json
from molecule import *

import pyscf
from   pyscf  import scf

def writeROHFSD_icf(mol=None, rhf=None, canonic=None, filename=None, noise=0.0):

    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2

    wf = rhf.mo_coeff[:, 0:max(Nup, Ndn)]
    wf = np.dot( canonic.XInv, wf )
    f = open(filename, 'w')

    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0))

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(2*canonic.L,Nup+Ndn))
    for i in range(Nup):
        for j in range(2*canonic.L):
            if j<canonic.L :
                f.write( '{:26.18e} {:26.18e} \n'.format( wf[j,i]+noise*random.random(),0.0 ) )
            else:
                f.write( '{:26.18e} {:26.18e} \n'.format( 0.0,0.0 ) )

    for i in range(Ndn):
        for j in range(2*canonic.L):
            if j<canonic.L :
                f.write( '{:26.18e} {:26.18e} \n'.format( 0.0,0.0 ) )
            else:
                f.write( '{:26.18e} {:26.18e} \n'.format( wf[j-canonic.L,i]+noise*random.random(),0.0 ) )
    f.close()

def writeUHFSD_icf(mol=None, uhf=None, canonic=None, filename=None, noise=0.0):

    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2

    wfUp = np.array(uhf.mo_coeff)[0, :, 0:Nup]
    wfUp = np.dot( canonic.XInv, wfUp )
    wfDn = np.array(uhf.mo_coeff)[1, :, 0:Ndn]
    wfDn = np.dot( canonic.XInv, wfDn )

    f = open(filename, 'w')

    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0))

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(2*canonic.L,Nup+Ndn))
    for i in range(Nup):
        for j in range(2*canonic.L):
            if j<canonic.L :
                f.write( '{:26.18e} {:26.18e} \n'.format( wfUp[j,i]+noise*random.random(),0.0 ) )
            else:
                f.write( '{:26.18e} {:26.18e} \n'.format( 0.0,0.0 ) )

    for i in range(Ndn):
        for j in range(2*canonic.L):
            if j<canonic.L :
                f.write( '{:26.18e} {:26.18e} \n'.format( 0.0,0.0 ) )
            else:
                f.write( '{:26.18e} {:26.18e} \n'.format( wfDn[j-canonic.L,i]+noise*random.random(),0.0 ) )
    f.close()

import pickle
def save_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
    import time
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    max_error : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`np.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao*nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = np.zeros((nchol_max, nao*nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0,mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2*l+1)*nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0,mol.nbas):
        shls = (i,i+1,0,mol.nbas,i,i+1,0,mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag:ndiag+di*nao] = buf.reshape(di*nao,di*nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs."%nchol_max)
        print("# max number of cholesky vectors = %d"%nchol_max)
        print("# iteration %5d: delta_max = %f"%(0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao*nao)
    # ERI[:,jl]
    eri_col = mol.intor('int2e_sph',
                         shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
    cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
    chol_vecs[0] = np.copy(eri_col[:,:,cj,cl].reshape(nao*nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor('int2e_sph',
                            shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
        # Select correct ERI chunk from shell.
        cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
        Munu0 = eri_col[:,:,cj,cl].reshape(nao*nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[:nchol+1,nu], chol_vecs[:nchol+1,:])
        chol_vecs[nchol+1] = (Munu0 - R) / (delta_max)**0.5
        #
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print ("# iteration %5d: delta_max = %13.8e: time = %13.8e"%info)

    return chol_vecs[:nchol]
    
class CanonicalBais:

    def __init__(self, mol, rhf, lindep=1e-8):

        self.nbasis = mol.nao_nr()
        ovlp   = rhf.get_ovlp()

        value, vector = np.linalg.eigh( ovlp )
        print( "Eigenvalue of overlap matrix: " )
        for index, item in enumerate( value ):
            print( "{:<9} {:26.18e}".format(index, item) )

        if( lindep >= value[-1] ):
            print("Error!!! lindep = {:.12f}, too big for determining the dependency!".format(lindep))
            sys.exit(1)
        numberOfDependent = next(i for i, v in enumerate(value) if v > lindep)

        print( "Number of dependent obritals is {}.".format(numberOfDependent) )
        print('\n')

        self.L = self.nbasis - numberOfDependent

        value = value[numberOfDependent:self.nbasis]
        vector = vector[:,numberOfDependent:self.nbasis]
        sqrtValue = np.sqrt(value)
        self.X = vector / sqrtValue
        self.XInv = vector.T * sqrtValue[:, None]
        self.XT = self.X.T
        #
        # print("HHHHHHHHHHHHHHHH: ", self.X.conj().T @ self.X)
        # X = rhf.mo_coeff
        # print("sssssssssssssss: ", X.conj().T @ X)
        # X = rhf.mo_coeff
        # self.X = X
        # self.XInv = aux_inv=np.linalg.inv(X)
        # self.XT = self.X.T

def getCholeskyAO(mol=None, tol=1e-8):

    nbasis  = mol.nao_nr()
    eri = scf._vhf.int2e_sph(mol._atm,mol._bas,mol._env)
    V   = ao2mo.restore(1, eri, nbasis)
    V   = V.reshape( nbasis*nbasis, nbasis*nbasis )

    choleskyVecAO = []; choleskyNum = 0
    Vdiag = V.diagonal().copy()
    while True:
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        print( "Inside modified Cholesky {:<9} {:26.18e}.".format(choleskyNum, vmax) )
        if(vmax<tol or choleskyNum==nbasis*nbasis):
            print( "Number of Cholesky fields is {:9}".format(choleskyNum) )
            print('\n')
            break
        else:
            oneVec = V[imax]/np.sqrt(vmax)
            #
            choleskyVecAO.append( oneVec )
            choleskyNum+=1
            V -= np.dot(oneVec[:, None], oneVec[None,:])
            Vdiag -= oneVec**2
        #
    return choleskyNum, choleskyVecAO

def getCholeskyMO_AO(mol=None, canonic=None, tol=1e-8):

    nbasis  = mol.nao_nr()
    choleskyNum, choleskyVecAO = getCholeskyAO(mol, tol)

    choleskyVecMO = np.zeros((choleskyNum, canonic.L*canonic.L))
    for i in range(choleskyNum):
        oneVec = choleskyVecAO[i].reshape(nbasis, nbasis)
        choleskyVecMO[i] = np.dot( canonic.XT, np.dot( oneVec, canonic.X ) ).ravel()

    return choleskyNum, choleskyVecAO, choleskyVecMO

dt                                   = 0.01
thermalSize                          = 2000
writeNumber                          = 200
measureNumberPerWrite                = 2
measureSkipStep                      = 50
walkerSizePerThread                  = 2
forceType                            = "dynamicForce" # "dynamicForce", "constForce"
forceCap                             = 10
initialPhiTFlag                      = "readFromFile" #"setFromModel", "setRandomly", "readFromFile"
initialWalkerFlag                    = "readFromFile"  #"setFromModel", "setRandomly", "sampleFromPhiT","readFromFile","readAllWalkers"
mgsStep                              = 10
popControlStep                       = 10
initPopControlMaxSize                = 0
logEnergyCap                         = 4
ET                                   = -10
backGroundETInit                       = "EstimateFromPhiTWalker" #"EstimateFromPhiTWalker", "EstimateFromPhiT", "readFromFile"
ETAdjustStep                         = 50
ETAdjustMaxSize                      = 200
ETAndBackGroundGrowthEstimateStep    = 50
ETAndBackGroundGrowthEstimateMaxSize = 200
seed                                 = 111

#For release
numOfReleasedSlice                   = 1

#For Metropolis
numOfChains                          = 20   #Due to TwoJastrow to One Jastrow, we only support one chain for now

numOfSweeps                          = 1
numOfThermalSweeps                   = 1
numOfMeasureSweeps                   = 1

numOfBrackets                        = 2

#parameter for update and measure
MetroUpdateSkip                      = 1
numOfSweepMeasurements               = 20

#write method_param
f = open('afqmc_param', 'w')
f.write(" {:<36} {:<26.18e} \n".format("dt", dt ) )
f.write(" {:<36} {:<26} \n".format("thermalSize", thermalSize) )
f.write(" {:<36} {:<26} \n".format("writeNumber", writeNumber) )
f.write(" {:<36} {:<26} \n".format("measureNumberPerWrite", measureNumberPerWrite) )
f.write(" {:<36} {:<26} \n".format("measureSkipStep", measureSkipStep) )
f.write(" {:<36} {:<26} \n".format("walkerSizePerThread",walkerSizePerThread) )
f.write(" {:<36} {:<26} \n".format("forceType", forceType) )
f.write(" {:<36} {:<26.18e} \n".format("forceCap", forceCap) )
f.write(" {:<36} {:<26} \n".format("initialPhiTFlag",initialPhiTFlag) )
f.write(" {:<36} {:<26} \n".format("initialWalkerFlag",initialWalkerFlag) )
f.write(" {:<36} {:<26} \n".format("mgsStep", mgsStep) )
f.write(" {:<36} {:<26} \n".format("popControlStep", popControlStep) )
f.write(" {:<36} {:<26} \n".format("initPopControlMaxSize", initPopControlMaxSize) )
f.write(" {:<36} {:<26.18e} \n".format("logEnergyCap", logEnergyCap) )
f.write(" {:<36} {:<26.18e} \n".format("ET", ET) )
f.write(" {:<36} {:<26} \n".format("backGroundETInit", backGroundETInit) )
f.write(" {:<36} {:<26} \n".format("ETAdjustStep", ETAdjustStep) )
f.write(" {:<36} {:<26} \n".format("ETAdjustMaxSize", ETAdjustMaxSize) )
f.write(" {:<36} {:<26} \n".format("ETAndBackGroundGrowthEstimateStep", ETAndBackGroundGrowthEstimateStep) )
f.write(" {:<36} {:<26} \n".format("ETAndBackGroundGrowthEstimateMaxSize", ETAndBackGroundGrowthEstimateMaxSize) )
f.write(" {:<36} {:<26} \n".format("seed", seed) )

f.write(" {:<36} {:<26} \n".format("numOfReleasedSlice",numOfReleasedSlice) )

f.write(" {:<36} {:<26} \n".format("numOfChains",numOfChains) )
f.write(" {:<36} {:<26} \n".format("numOfBrackets",numOfBrackets) )
f.write(" {:<36} {:<26} \n".format("numOfSweeps",numOfSweeps) )
f.write(" {:<36} {:<26} \n".format("numOfThermalSweeps",numOfThermalSweeps) )
f.write(" {:<36} {:<26} \n".format("numOfMeasureSweeps",numOfMeasureSweeps) )

f.write(" {:<36} {:<26} \n".format("MetroUpdateSkip",MetroUpdateSkip) )
f.write(" {:<36} {:<26} \n".format("numOfSweepMeasurements",numOfSweepMeasurements) )
f.close()

##################################################################
def setupMolecule_icf(atoms=None,chrg=None,spn=None,basis=None,psp=None,sym=None):

    mol          = pyscf.gto.Mole()
    mol.verbose  = 4
    mol.output   = 'mole.dat'
    mol.atom     = atoms
    mol.charge   = chrg
    mol.spin     = spn
    mol.basis    = basis
    mol.symmetry = sym
    mol.ecp      = psp
    mol.unit     = 'Angstrom'
    mol.build()

    Enuc    = gto.energy_nuc(mol)
    nbasis  = mol.nao_nr()
    nelec_a = (mol.nelectron+mol.spin)//2
    nelec_b = (mol.nelectron-mol.spin)//2

    print('Molecule [geometry in Angstrom]')
    print(atoms)

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

def writeInputForModel_forGeneralHamiltonian(scale=1.0, mol=None, rhf=None, canonic=None, tol=1e-8, name="model_param"):

    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2

    choleskyNum, choleskyVecMO = getCholeskyMO(mol, canonic, tol)
    #
    choleskyVecMO = choleskyVecMO * np.sqrt(scale)
    t = np.dot( canonic.XT, np.dot( rhf.get_hcore(), canonic.X ) ) * scale
    #
    K = t.copy()
    for i in range(choleskyNum):
        oneVec = choleskyVecMO[i].reshape(canonic.L, canonic.L)
        K += (-0.5)*np.dot( oneVec, oneVec )
    ###############################
    svdVecs=choleskyVecMO
    svdNumber=choleskyNum
    ############################
    #ATTention: there is a transpose between python and c++ i/o trans
    KT=np.zeros((2*canonic.L, 2*canonic.L),dtype=np.complex128)
    KT[0:canonic.L,0:canonic.L]=K.transpose()
    KT[canonic.L:2*canonic.L,canonic.L:2*canonic.L]=K.transpose()

    svdVecsT=np.zeros((svdNumber, 2*canonic.L, 2*canonic.L),dtype=np.complex128)
    for i in range(svdNumber):
        svdVecsT[i,0:canonic.L,0:canonic.L]=svdVecs[i].reshape(canonic.L, canonic.L).transpose()
        svdVecsT[i,canonic.L:2*canonic.L,canonic.L:2*canonic.L]=svdVecs[i].reshape(canonic.L, canonic.L).transpose()

    f = h5py.File(name, "w")
    f.create_dataset("L",              (1,),                                data=[2*canonic.L],           dtype='int')
    f.create_dataset("Nup",              (1,),                                data=[Nup],           dtype='int')
    f.create_dataset("Ndn",              (1,),                                data=[Ndn],           dtype='int')
    f.create_dataset("N",            (1,),                                data=[Nup+Ndn],                     dtype='int')
    f.create_dataset("svdNumber", (1,),                                data=[svdNumber],           dtype='int')
    f.create_dataset("K_r",              ((2*canonic.L)**2,),                 data=KT.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_r",   (svdNumber*(2*canonic.L)**2,), data=svdVecsT.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdBg_r",     (svdNumber,), data=np.zeros(svdNumber),   dtype='float64')
    f.create_dataset("K_i",              ((2*canonic.L)**2,),                 data=KT.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_i",   (svdNumber*(2*canonic.L)**2,), data=svdVecsT.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdBg_i",     (svdNumber,),                  data=np.zeros(svdNumber),   dtype='float64')
    f.close()
    

def writeInputForModel_forGeneralHamiltonian_load_hamiltonian_and_selfCheck(mol=None, rhf=None, canonic=None, tol=1e-8, hamiltonian_path="AFQMC_hamiltonian.pkl", name="model_param"):
    # from pauxy_scripts import chunked_cholesky
    def get_orth_ao(mf):
        X = mf.mo_coeff
        return X
    #
    def rotate_h1e(h1e, X):
        return X.conj().T @ h1e @ X
    def rotate_eri_chol(ceri, X):
        return np.einsum("kpr,pi,rl->kil", ceri, X.conj(), X)
    def rotate_integrals(h1e, eri, X):
        # the shape of X has to be [nao x nb]
        # normally nb is nao but can also be less
        # 1-body term
        assert X.ndim == 2
        h1e = rotate_h1e(h1e, X)
        # eri, in cholesky vector
        if isinstance(eri, np.ndarray) and eri.ndim == 3:
            print("rotate_eri_chol!!!")
            eri = rotate_eri_chol(eri, X)
        # eri, in dense form or just mol
        return h1e, eri
    #############################################################################################
    hamil = load_pickle(hamiltonian_path)
    # hamil = load_pickle('Zixiang_d2.1/hamiltonian.pkl')
    # hamil = load_pickle('AFQMC_hamiltonian.pkl')
    ###############################
    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2
    choleskyNum, choleskyVecAO, choleskyVecMO = getCholeskyMO_AO(mol, canonic, tol)
    #
    t = np.dot( canonic.XT, np.dot( rhf.get_hcore(), canonic.X ) )
    #
    K = t.copy()
    for i in range(choleskyNum):
        oneVec = choleskyVecMO[i].reshape(canonic.L, canonic.L)
        K += (-0.5)*np.dot( oneVec, oneVec )
    ###############################
    h1e_read, ceri_read, enuc_read, (wfn_a_read, wfn_b_read), aux = hamil
    aux_inv=np.linalg.inv(aux["orth_mat"])
    print("===========================")
    enuc_icf    = gto.energy_nuc(mol)
    print("nuclear repulsion this script:", enuc_icf)
    print("nuclear repulsion read:", enuc_read)
    print("===========================")
    #
    ###############################
    #check orthornormal basis and get basis trans matrix
    counter = 0.0
    test_matrx=np.array(canonic.XT.conj().T - aux["orth_mat"])
    for i in range(test_matrx.shape[0]): 
        for j in range(test_matrx.shape[1]): 
            counter += abs(test_matrx[i,j])
            if abs(test_matrx[i,j]) <= 10e-7:
                test_matrx[i,j] = 0.0
    print("absSum(canonic.XT.conj().T - aux): ",counter)
    print("===========================")
    # X_yixiao = get_orth_ao(rhf)
    # #
    # counter = 0.0
    # test_matrx=np.array(X_yixiao - aux["orth_mat"])
    # for i in range(test_matrx.shape[0]): 
    #     for j in range(test_matrx.shape[1]): 
    #         counter += abs(test_matrx[i,j])
    #         if abs(test_matrx[i,j]) <= 10e-7:
    #             test_matrx[i,j] = 0.0
    # print("absSum(X_yixiao - aux): ",counter)
    # print("===========================")
    # counter = 0.0
    # test_matrx=np.array(X_yixiao) - canonic.X
    # for i in range(test_matrx.shape[0]): 
    #     for j in range(test_matrx.shape[1]): 
    #         counter += abs(test_matrx[i,j])
    #         if abs(test_matrx[i,j]) <= 10e-7:
    #             test_matrx[i,j] = 0.0
    # print("HAFQMC and AFQMCLAB take different way to choose orthonormal basis:")
    # print("absSum(X_yixiao - canonic.XT): ",counter)
    # print("===========================")
    HAFQMC2AFQMC_matrix_temp = np.dot( canonic.XT, aux_inv.conj().T  )
    HAFQMC2AFQMC_matrix = np.zeros((2*canonic.L, 2*canonic.L),dtype=np.complex128)
    HAFQMC2AFQMC_matrix[0:canonic.L, 0:canonic.L] = HAFQMC2AFQMC_matrix_temp
    HAFQMC2AFQMC_matrix[canonic.L:2*canonic.L, canonic.L:2*canonic.L] = HAFQMC2AFQMC_matrix_temp
    #
    h1e_read_trans = np.dot(HAFQMC2AFQMC_matrix_temp,np.dot(np.array(h1e_read),HAFQMC2AFQMC_matrix_temp.conj().T))
    counter = 0.0
    test_matrx=np.array(h1e_read_trans - t)
    for i in range(test_matrx.shape[0]): 
        for j in range(test_matrx.shape[1]): 
            counter += abs(test_matrx[i,j])
            if abs(test_matrx[i,j]) <= 10e-7:
                test_matrx[i,j] = 0.0
    print("absSum(h1e_read_trans - t): ",counter)
    print("===========================")
    ###############################
    #check wfn_a, wfn_b
    wfn_read = np.zeros((2*wfn_a_read.shape[0], 2*wfn_a_read.shape[1]),dtype=np.complex128)
    for i in range(wfn_a_read.shape[0]):
        for j in range(wfn_a_read.shape[1]):
            wfn_read[i,j]=wfn_a_read[i,j]
            wfn_read[i+wfn_a_read.shape[0],j+wfn_a_read.shape[1]]=wfn_b_read[i,j]
    wfn_read_trans = np.dot(HAFQMC2AFQMC_matrix,wfn_read)
    #
    wfn_a_icf = rhf.mo_coeff[:, 0:Nup]
    wfn_b_icf = rhf.mo_coeff[:, 0:Ndn]
    wfn_a_icf = np.dot( canonic.XInv, wfn_a_icf )
    wfn_b_icf = np.dot( canonic.XInv, wfn_b_icf )
    wfn_icf = np.zeros((2*wfn_a_icf.shape[0], 2*wfn_a_icf.shape[1]),dtype=np.complex128)
    for i in range(wfn_a_icf.shape[0]):
        for j in range(wfn_a_icf.shape[1]):
            wfn_icf[i,j]=wfn_a_icf[i,j]
            wfn_icf[i+wfn_a_icf.shape[0],j+wfn_a_icf.shape[1]]=wfn_b_icf[i,j]
    #
    counter = 0.0
    test_matrx=np.array(wfn_read_trans - wfn_icf)
    for i in range(test_matrx.shape[0]): 
        for j in range(test_matrx.shape[1]): 
            counter += abs(test_matrx[i,j])
            if abs(test_matrx[i,j]) <= 10e-7:
                test_matrx[i,j] = 0.0
    print("absSum(wfn_read_trans - wfn): ",counter)
    print("===========================")
    ###############################
    ceri_AO_read = np.einsum("kpr,pi,rl->kil", ceri_read, aux_inv.conj(), aux_inv)
    eri_AO_read = np.einsum("kpr,kqs->prqs", np.array(ceri_AO_read), np.array(ceri_AO_read)) 
    #
    choleskyVecAO_matrix=np.array(choleskyVecAO)
    choleskyVecAO_matrix = choleskyVecAO_matrix.reshape(choleskyNum, mol.nao, mol.nao)
    eri_icf = np.einsum("kpr,kqs->prqs", choleskyVecAO_matrix, choleskyVecAO_matrix) 
    #
    counter = 0.0
    test_matrx=np.array(eri_AO_read- eri_icf)
    for i in range(test_matrx.shape[0]): 
        for j in range(test_matrx.shape[1]): 
            for k in range(test_matrx.shape[2]): 
                for l in range(test_matrx.shape[3]):
                    counter += abs(test_matrx[i,j,k,l])
                    if abs(test_matrx[i,j,k,l]) <= 10e-7:
                        test_matrx[i,j,k,l] = 0.0
    print("absSum(eri_AO_read- eri_AO_icf): ",counter)
    print("===========================")
    #
    ceri_trans_read = np.einsum("kpr,ip,rl->kil", ceri_read, HAFQMC2AFQMC_matrix_temp, HAFQMC2AFQMC_matrix_temp.conj().T)
    eri_trans_read = np.einsum("kpr,kqs->prqs", np.array(ceri_trans_read), np.array(ceri_trans_read)) 
    #
    choleskyVecMO_matrix=np.array(choleskyVecMO)
    choleskyVecMO_matrix = choleskyVecMO_matrix.reshape(choleskyNum, mol.nao, mol.nao)
    eri_icf = np.einsum("kpr,kqs->prqs", choleskyVecMO_matrix, choleskyVecMO_matrix) 
    #
    counter = 0.0
    test_matrx=np.array(eri_trans_read- eri_icf)
    for i in range(test_matrx.shape[0]): 
        for j in range(test_matrx.shape[1]): 
            for k in range(test_matrx.shape[2]): 
                for l in range(test_matrx.shape[3]):
                    counter += abs(test_matrx[i,j,k,l])
                    if abs(test_matrx[i,j,k,l]) <= 10e-7:
                        test_matrx[i,j,k,l] = 0.0
    print("absSum(eri_trans_read- eri_icf): ",counter)
    print("===========================")
    ###############################
    svdVecs=choleskyVecMO
    svdNumber=choleskyNum
    ############################
    #ATTention: there is a transpose between python and c++ i/o trans
    KT=np.zeros((2*canonic.L, 2*canonic.L),dtype=np.complex128)
    KT[0:canonic.L,0:canonic.L]=K.transpose()
    KT[canonic.L:2*canonic.L,canonic.L:2*canonic.L]=K.transpose()

    svdVecsT=np.zeros((svdNumber, 2*canonic.L, 2*canonic.L),dtype=np.complex128)
    for i in range(svdNumber):
        svdVecsT[i,0:canonic.L,0:canonic.L]=svdVecs[i].reshape(canonic.L, canonic.L).transpose()
        svdVecsT[i,canonic.L:2*canonic.L,canonic.L:2*canonic.L]=svdVecs[i].reshape(canonic.L, canonic.L).transpose()

    f = h5py.File(name, "w")
    f.create_dataset("L",              (1,),                                data=[2*canonic.L],           dtype='int')
    f.create_dataset("Nup",              (1,),                                data=[Nup],           dtype='int')
    f.create_dataset("Ndn",              (1,),                                data=[Ndn],           dtype='int')
    f.create_dataset("N",            (1,),                                data=[Nup+Ndn],                     dtype='int')
    f.create_dataset("svdNumber", (1,),                                data=[svdNumber],           dtype='int')
    f.create_dataset("K_r",              ((2*canonic.L)**2,),                 data=KT.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_r",   (svdNumber*(2*canonic.L)**2,), data=svdVecsT.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdBg_r",     (svdNumber,), data=np.zeros(svdNumber),   dtype='float64')
    f.create_dataset("K_i",              ((2*canonic.L)**2,),                 data=KT.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_i",   (svdNumber*(2*canonic.L)**2,), data=svdVecsT.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdBg_i",     (svdNumber,),                  data=np.zeros(svdNumber),   dtype='float64')
    f.close()
    #############################
    hamil = t, choleskyVecMO_matrix, enuc_icf, (wfn_a_icf, wfn_b_icf), {"orth_mat": canonic.XT.conj().T} 
    save_pickle("AFQMC_hamiltonian.pkl", hamil)


def writeInputForModel_forGeneralHamiltonian_HAFQMC(mol=None, rhf=None, canonic=None, tol=1e-8, hamiltonian_path="AFQMC_hamiltonian.pkl", name="model_param"):
    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2
    choleskyNum, choleskyVecAO, choleskyVecMO = getCholeskyMO_AO(mol, canonic, tol)
    #
    t = np.dot( canonic.XT, np.dot( rhf.get_hcore(), canonic.X ) )
    #
    K = t.copy()
    for i in range(choleskyNum):
        oneVec = choleskyVecMO[i].reshape(canonic.L, canonic.L)
        K += (-0.5)*np.dot( oneVec, oneVec )
    ###############################
    wfn_a_icf = rhf.mo_coeff[:, 0:Nup]
    wfn_b_icf = rhf.mo_coeff[:, 0:Ndn]
    wfn_a_icf = np.dot( canonic.XInv, wfn_a_icf )
    wfn_b_icf = np.dot( canonic.XInv, wfn_b_icf )
    #
    choleskyVecMO_matrix=np.array(choleskyVecMO)
    choleskyVecMO_matrix = choleskyVecMO_matrix.reshape(choleskyNum, mol.nao, mol.nao)
    ###############################
    svdVecs=choleskyVecMO
    svdNumber=choleskyNum
    ############################
    #ATTention: there is a transpose between python and c++ i/o trans
    KT=np.zeros((2*canonic.L, 2*canonic.L),dtype=np.complex128)
    KT[0:canonic.L,0:canonic.L]=K.transpose()
    KT[canonic.L:2*canonic.L,canonic.L:2*canonic.L]=K.transpose()
    #
    svdVecsT=np.zeros((svdNumber, 2*canonic.L, 2*canonic.L),dtype=np.complex128)
    for i in range(svdNumber):
        svdVecsT[i,0:canonic.L,0:canonic.L]=svdVecs[i].reshape(canonic.L, canonic.L).transpose()
        svdVecsT[i,canonic.L:2*canonic.L,canonic.L:2*canonic.L]=svdVecs[i].reshape(canonic.L, canonic.L).transpose()
    #
    f = h5py.File(name, "w")
    f.create_dataset("L",              (1,),                                data=[2*canonic.L],           dtype='int')
    f.create_dataset("Nup",              (1,),                                data=[Nup],           dtype='int')
    f.create_dataset("Ndn",              (1,),                                data=[Ndn],           dtype='int')
    f.create_dataset("N",            (1,),                                data=[Nup+Ndn],                     dtype='int')
    f.create_dataset("svdNumber", (1,),                                data=[svdNumber],           dtype='int')
    f.create_dataset("K_r",              ((2*canonic.L)**2,),                 data=KT.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_r",   (svdNumber*(2*canonic.L)**2,), data=svdVecsT.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdBg_r",     (svdNumber,), data=np.zeros(svdNumber),   dtype='float64')
    f.create_dataset("K_i",              ((2*canonic.L)**2,),                 data=KT.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_i",   (svdNumber*(2*canonic.L)**2,), data=svdVecsT.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdBg_i",     (svdNumber,),                  data=np.zeros(svdNumber),   dtype='float64')
    f.close()
    #############################
    enuc_icf    = gto.energy_nuc(mol)
    hamil = t, choleskyVecMO_matrix, enuc_icf, (wfn_a_icf, wfn_b_icf), {"orth_mat": canonic.XT.conj().T} 
    save_pickle(hamiltonian_path, hamil)

##################################################################
#model
##################################################################
one_atom          = 'V'   #+O
chrg              = 0
spn               = 3
basis             = 'vdz'

mol = setupMolecule(one_atom, chrg, spn, basis)
# mol = setupMolecule_icf(atoms, chrg, spn, basis)
rhf  = doROHFCalculation(basis, one_atom, chrg, mol)
# rhf  = doROHFCalculation(mol, 1e-12, 1e-8, 0.4, 50000, 5000)
############################################################
#take hf and hamiltonian input from HAFQMC and generate corrsponding input for AFQMC
############################################################
# rhf.__dict__.update(pyscf.scf.chkfile.load('Zixiang_icf_d3.6/AFQMC_rhf.chk', 'scf'))
# rhf.__dict__.update(pyscf.scf.chkfile.load('Zixiang_d2.1/scf.chk', 'scf'))
# rhf.__dict__.update(pyscf.scf.chkfile.load('AFQMC_rhf.chk', 'scf'))
#
# canonic = CanonicalBais(mol, rhf, 1e-8)
#
# writeInputForModel_forGeneralHamiltonian(1.0, mol, rhf, canonic, 1e-6, "model_param")
# writeROHFSD_icf(mol, rhf, canonic, "phi_BG.dat")
#
# writeInputForModel_forGeneralHamiltonian_load_hamiltonian_and_selfCheck(mol, rhf, canonic, 1e-6, "Zixiang_icf_d3.6/AFQMC_hamiltonian.pkl", "model_param")
# writeInputForModel_forGeneralHamiltonian_load_hamiltonian_and_selfCheck(mol, rhf, canonic, 1e-6, "Zixiang_d2.1/hamiltonian.pkl", "model_param")
# writeInputForModel_forGeneralHamiltonian_load_hamiltonian_and_selfCheck(mol, rhf, canonic, 1e-6, "AFQMC_hamiltonian.pkl", "model_param")
############################################################
#generate hf and hamiltonian input for both HAFQMC and AFQMC (HAFQMC and AFQMC take exatltly the same second quantized Hamiltonian with exatltly the same MO)
############################################################
rhf = scf.RHF(mol).set(chkfile ='AFQMC_rhf.chk')
rhf.kernel()
canonic = CanonicalBais(mol, rhf, 1e-8)
writeInputForModel_forGeneralHamiltonian_HAFQMC(mol, rhf, canonic, 1e-6, "AFQMC_hamiltonian.pkl", "model_param")
#####################################
#converge to UHF
#####################################
# uhf = scf.UHF(mol)
# uhf.conv_tol = 1e-6

# uhf.kernel()

# for i in range(10):
#     # Analyze UHF stability
#     mo_new = uhf.stability(internal=True, external=False)[0]
#     # Get new density matrix with new MOs 
#     dm_new = uhf.make_rdm1(mo_new, uhf.mo_occ)
#     # Run HF again with the new density matrix 
#     uhf.kernel(dm0=dm_new)
#     # Once again, check UHF stability
#     uhf.stability(internal=True, external=False)[0]

# uhf.analyze()
# writeUHFSD_icf(mol, uhf, canonic, "phi_BG.dat")
#######################################################################################
# density matrix
#######################################################################################
# from pyscf import mcscf

# dm1_alpha= rhf.make_rdm1()
# dm1_beta = dm1_alpha

# dm1_MO_alpha=np.dot(canonic.XInv,np.dot(dm1_alpha,canonic.XInv.T))
# dm1_MO_beta=np.dot(canonic.XInv,np.dot(dm1_beta,canonic.XInv.T))

# f = open('densityMatrix_RHF.dat', 'w')

# for spin in range(2):
#     for i in range(canonic.L):
#         if spin == 0:
#             for j in range(canonic.L):
#                 f.write( '{:26.18e} {:26.18e} {:26.18e} \n'.format( dm1_MO_alpha[j,i]/2.0,0.0,0.0 ) )
#             for j in range(canonic.L):
#                 f.write( '{:26.18e} {:26.18e} {:26.18e} \n'.format( 0.0,0.0,0.0 ) )
#         if spin == 1:
#             for j in range(canonic.L):
#                 f.write( '{:26.18e} {:26.18e} {:26.18e} \n'.format( 0.0,0.0,0.0 ) )
#             for j in range(canonic.L):
#                 f.write( '{:26.18e} {:26.18e} {:26.18e} \n'.format( dm1_MO_beta[j,i]/2.0,0.0,0.0 ) )
# f.close()

# mycas = mcscf.CASSCF(rhf, 12, (5,5))
# mycas.kernel()
# dm1_alpha, dm1_beta = mycas.make_rdm1s()

# dm1_MO=np.dot(canonic.XInv,np.dot(dm1_alpha,canonic.XInv.T))

# dm1_MO_full=np.zeros((2*canonic.L, 2*canonic.L),dtype=np.complex128)
# dm1_MO_full[:canonic.L,:canonic.L]=dm1_MO
# dm1_MO_full[canonic.L:2*canonic.L,canonic.L:2*canonic.L]=dm1_MO
# np.save('densityMatrix_CASSCF.npy',dm1_MO_full)

# f = open('densityMatrix_CASSCF.dat', 'w')
# for i in range(2*canonic.L):
#     for j in range(2*canonic.L):
#             f.write( '{:26.18e} {:26.18e} {:26.18e} \n'.format( dm1_MO_full[j,i].real,0.0,0.0 ) )
# f.close()

#######################################################################################
#Jastrow param
#############################################################################################
#Method Parameter

ET                                   = -14.0
Metro_dtET                           = 0.1*ET   
seed                                 = 0

##
JastrowSlice                         = [1,1]     #<Phi_T|=<SD|J_0 J_1 ... 
JastrowName                         = ["generalHamiltonian_HAFQMC_icf","generalHamiltonian_K_HAFQMC_icf"] #"realMaterial"    
numOfJastrow                         = len(JastrowSlice)

##
BPMetroSampleCap                     = 5
BPMetroStabilizeStep                 = 1
blockNum                             = 0
BPMetroForceType			         = "constForce"
BPMetroInitialAuxiliaryFlag		     = "constForceInitial"         #constForceInitial, dynamicForceInitial

#write method_param
f = open('afqmc_param_Metro', 'w')
f.write(" {:<36} {:<26} \n".format("numOfJastrow",numOfJastrow) )
for i in range(numOfJastrow):
    f.write(" {:<36} {:<26} \n".format("JastrowSlice_"+str(i),JastrowSlice[i]) )
for i in range(numOfJastrow):
    f.write(" {:<36} {:<26} \n".format("JastrowName_"+str(i),JastrowName[i]) )
f.write(" {:<36} {:<26.18e} \n".format("Metro_dtET", Metro_dtET) )
f.write(" {:<36} {:<26} \n".format("seed", seed) )

f.write(" {:<36} {:<26} \n".format("BPMetroSampleCap", BPMetroSampleCap) )
f.write(" {:<36} {:<26} \n".format("BPMetroStabilizeStep", BPMetroStabilizeStep) )
f.write(" {:<36} {:<26} \n".format("blockNum", blockNum) )
f.write(" {:<36} {:<26} \n".format("BPMetroForceType", BPMetroForceType) )
f.write(" {:<36} {:<26} \n".format("BPMetroInitialAuxiliaryFlag", BPMetroInitialAuxiliaryFlag) )
f.close()

#####################################
def writeInputForModel_fromHAFQMC_forGeneralHamiltonian_slice1(scale=1.0, mol=None, canonic=None, tol=1e-8, hamiltonian_path="hamiltonian.pkl", param_path="ckpt_2000.pkl",  name="model_param"):
    #|\phiT> = \prod [exp(-tT)exp(-tV)]|\phi_0>
    def save_pickle(filename, data):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    def load_pickle(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
    # also 2.4, 2.7, 3.0, 3.6, 4.2
    hamil = load_pickle(hamiltonian_path)
    ckpt = load_pickle(param_path)
    params = ckpt[1][1]['params']['ansatz']

    # see https://github.com/y1xiaoc/hafqmc/blob/master/hafqmc/hamiltonian.py#L334
    # for detailed meaning of the arrays
    h1e, ceri, enuc, (wfn_a, wfn_b), aux = hamil

    # basis size 28
    # number of electrons 14 (7 up + 7 down)
    print('shape of the hamiltonian arrays:')
    print('one body operator K (h1e):', h1e.shape)
    print('two body operator after cholesky (ceri):', ceri.shape)
    print('nuclear contribution of energy (enuc):', enuc)
    print('spin up wfn coeffs (wfn_a):', wfn_a.shape)
    print('spin down wfn coeffs (wfn_b):', wfn_b.shape)
    print('ao2mo rotaton matrix ("orth_mat"):', aux['orth_mat'].shape)

    # remove extra layers
    # params = params['params']['ansatz']

    # one body operators at different slices, in GHF
    hmf=[]
    hmf0 = params['propagators_0']['hmf_ops_0']['hmf']
    hmf.append(hmf0)
    hmf1 = params['propagators_0']['hmf_ops_1']['hmf']
    hmf.append(hmf1)
    print(f"{hmf0.shape = }")

    # decompsed two body operators, same for all slices, 100 aux fields
    vhs=[]
    vhs1 = params['propagators_0']['vhs_ops_0']['vhs']
    # vhs0 = np.zeros((vhs1.shape[0], vhs1.shape[1], vhs1.shape[2]),dtype=np.complex128)
    # vhs.append(vhs0)
    vhs.append(vhs1)
    vhs.append(vhs1)
    print(f"{vhs1.shape = }")

    # time steps, 4 one body steps, 3 two body steps, ts_v is after sqrt
    ts_h = params['propagators_0']['ts_h']
    ts_v = params['propagators_0']['ts_v']
    ts_v_icf = np.zeros(( ts_v.shape[0] + 1),dtype=np.complex128)
    #
    ts_v_icf[0:1] = ts_v
    ts_v_icf[1] = 0.0
    print("ts_v_icf: ",ts_v_icf)
    print(f"{ts_h.shape = }")
    print(f"{ts_v.shape = }")
    #
    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2
    #
    #####################################
    #first three slice for K+V
    #####################################
    aux_inv=np.linalg.inv(aux["orth_mat"])
    HAFQMC2AFQMC_matrix_temp = np.dot( canonic.XT, aux_inv.conj().T  )
    HAFQMC2AFQMC_matrix = np.zeros((2*canonic.L, 2*canonic.L),dtype=np.complex128)
    HAFQMC2AFQMC_matrix[0:canonic.L, 0:canonic.L] = HAFQMC2AFQMC_matrix_temp
    HAFQMC2AFQMC_matrix[canonic.L:2*canonic.L, canonic.L:2*canonic.L] = HAFQMC2AFQMC_matrix_temp
    for Jastrow in [0,1]:
        choleskyNum = vhs[Jastrow].shape[0]
        #####################################
        choleskyVecMO_temp = np.array(vhs[Jastrow] * ts_v_icf[Jastrow])
        # choleskyVecMO_temp = np.array(vhs[Jastrow] * 0.0)
        #
        choleskyVecMO = choleskyVecMO_temp
        for i in range(choleskyVecMO.shape[0]):
            choleskyVecMO[i] = np.dot( HAFQMC2AFQMC_matrix, np.dot( choleskyVecMO_temp[i], HAFQMC2AFQMC_matrix.conj().T ) )
        #####################################
        K_temp = np.array(hmf[Jastrow] *  ts_h[Jastrow])
        K = np.dot( HAFQMC2AFQMC_matrix, np.dot( K_temp, HAFQMC2AFQMC_matrix.conj().T ) )
        #
        svdVecs=choleskyVecMO
        svdNumber=choleskyNum
        ############################
        #ATTention: there is a transpose between python and c++ i/o trans
        KT=np.zeros((2*canonic.L, 2*canonic.L),dtype=np.complex128)
        KT=K.transpose()

        svdVecsT=np.zeros((svdNumber, 2*canonic.L, 2*canonic.L),dtype=np.complex128)
        for i in range(svdNumber):
            svdVecsT[i,0:2*canonic.L,0:2*canonic.L]=svdVecs[i].transpose()

        f = h5py.File(name+"_"+str(Jastrow), "w")
        f.create_dataset("L",              (1,),                                data=[2*canonic.L],           dtype='int')
        f.create_dataset("Nup",              (1,),                                data=[Nup],           dtype='int')
        f.create_dataset("Ndn",              (1,),                                data=[Ndn],           dtype='int')
        f.create_dataset("N",            (1,),                                data=[Nup+Ndn],                     dtype='int')
        f.create_dataset("svdNumber", (1,),                                data=[svdNumber],           dtype='int')
        f.create_dataset("K_r",              ((2*canonic.L)**2,),                 data=KT.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
        f.create_dataset("svdVecs_r",   (svdNumber*(2*canonic.L)**2,), data=svdVecsT.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
        f.create_dataset("svdBg_r",     (svdNumber,), data=np.zeros(svdNumber),   dtype='float64')
        f.create_dataset("K_i",              ((2*canonic.L)**2,),                 data=KT.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
        f.create_dataset("svdVecs_i",   (svdNumber*(2*canonic.L)**2,), data=svdVecsT.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans
        f.create_dataset("svdBg_i",     (svdNumber,),                  data=np.zeros(svdNumber),   dtype='float64')
        f.close()

def writeUHFSD_fromHAFQMC_icfTrans(mol=None, canonic=None, hamiltonian_path="hamiltonian.pkl", param_path="ckpt_2000.pkl", filename=None, noise=0.0):
    def save_pickle(filename, data):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    def load_pickle(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
    # also 2.4, 2.7, 3.0, 3.6, 4.2
    hamil = load_pickle(hamiltonian_path)
    ckpt = load_pickle(param_path)
    params = ckpt[1][1]['params']['ansatz']

    h1e, ceri, enuc, (wfn_a, wfn_b), aux = hamil

    # initial wavefunction in GHF to start projection
    wfn0_a = np.array(params['wfn_a'])
    wfn0_b = np.array(params['wfn_b'])
    wfn0 = np.zeros((2*wfn0_a.shape[0], 2*wfn0_a.shape[1]),dtype=np.complex128)
    for i in range(wfn0_a.shape[0]):
        for j in range(wfn0_a.shape[1]):
            wfn0[i,j]=wfn0_a[i,j]
            wfn0[i+wfn0_a.shape[0],j+wfn0_a.shape[1]]=wfn0_b[i,j]
    #
    aux_inv=np.linalg.inv(aux["orth_mat"])
    HAFQMC2AFQMC_matrix_temp = np.dot( canonic.XT, aux_inv.conj().T  )
    #
    HAFQMC2AFQMC_matrix = np.zeros((2*canonic.L, 2*canonic.L),dtype=np.complex128)
    HAFQMC2AFQMC_matrix[0:canonic.L, 0:canonic.L] = HAFQMC2AFQMC_matrix_temp
    HAFQMC2AFQMC_matrix[canonic.L:2*canonic.L, canonic.L:2*canonic.L] = HAFQMC2AFQMC_matrix_temp
    wf = np.dot( HAFQMC2AFQMC_matrix, wfn0 )
    #
    print(f"{wf.shape = }")
    #
    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2

    f = open(filename, 'w')

    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0))

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(2*canonic.L,Nup+Ndn))
    for i in range(Nup+Ndn):
        for j in range(2*canonic.L):
                f.write( '{:26.18e} {:26.18e} \n'.format( wf[j,i].real+noise*random.random(),0.0 ) )
    f.close()

def writeUHFSD_fromHAFQMC_icfTrans_mix(mol=None, canonic=None, hamiltonian_path="hamiltonian.pkl", param_path="ckpt_2000.pkl", filename=None, noise=0.0):
    def save_pickle(filename, data):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    def load_pickle(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
    # also 2.4, 2.7, 3.0, 3.6, 4.2
    hamil = load_pickle(hamiltonian_path)
    ckpt = load_pickle(param_path)
    params = ckpt[1][1]['params']['ansatz']

    h1e, ceri, enuc, (wfn_a, wfn_b), aux = hamil

    # initial wavefunction in GHF to start projection
    wfn0_a = np.array(params['wfn_a'])
    wfn0 = wfn0_a
    # print(wfn0)
    #
    aux_inv=np.linalg.inv(aux["orth_mat"])
    HAFQMC2AFQMC_matrix_temp = np.dot( canonic.XT, aux_inv.conj().T  )
    #
    HAFQMC2AFQMC_matrix = np.zeros((2*canonic.L, 2*canonic.L),dtype=np.complex128)
    HAFQMC2AFQMC_matrix[0:canonic.L, 0:canonic.L] = HAFQMC2AFQMC_matrix_temp
    HAFQMC2AFQMC_matrix[canonic.L:2*canonic.L, canonic.L:2*canonic.L] = HAFQMC2AFQMC_matrix_temp
    wf = np.dot( HAFQMC2AFQMC_matrix, wfn0 )
    #
    print(f"{wf.shape = }")
    #
    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2

    f = open(filename, 'w')

    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0))

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(2*canonic.L,Nup+Ndn))
    for i in range(Nup+Ndn):
        for j in range(2*canonic.L):
                f.write( '{:26.18e} {:26.18e} \n'.format( wf[j,i].real+noise*random.random(),0.0 ) )
    f.close()

def writeUHFSD_fromHAFQMC_icfRead(mol=None, canonic=None, param_path="ckpt_2000.pkl", filename=None, noise=0.0):
    def save_pickle(filename, data):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    def load_pickle(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    # also 2.4, 2.7, 3.0, 3.6, 4.2
    ckpt = load_pickle(param_path)
    params = ckpt[1][1]['params']['ansatz']

    # initial wavefunction in GHF to start projection
    wfn0_a = np.array(params['wfn_a'])
    wfn0_b = np.array(params['wfn_b'])
    wfn0 = np.zeros((2*wfn0_a.shape[0], 2*wfn0_a.shape[1]),dtype=np.complex128)
    for i in range(wfn0_a.shape[0]):
        for j in range(wfn0_a.shape[1]):
            wfn0[i,j]=wfn0_a[i,j]
            wfn0[i+wfn0_a.shape[0],j+wfn0_a.shape[1]]=wfn0_b[i,j]
    #
    wf = wfn0
    #
    print(f"{wf.shape = }")
    #
    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2

    f = open(filename, 'w')

    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0))

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(2*canonic.L,Nup+Ndn))
    for i in range(Nup+Ndn):
        for j in range(2*canonic.L):
                f.write( '{:26.18e} {:26.18e} \n'.format( wf[j,i].real+noise*random.random(),0.0 ) )
    f.close()

# writeInputForModel_fromHAFQMC_forGeneralHamiltonian_slice1(1.0, mol, canonic, 1e-6, "Zixiang_icf_d3.6/AFQMC_hamiltonian.pkl", "Zixiang_icf_d3.6/no_penalty_ckpt_24000.pkl", "model_Jastrow_param")
# for i in range(numOfChains):
#     writeUHFSD_fromHAFQMC_icfTrans(mol, canonic, "Zixiang_icf_d3.6/AFQMC_hamiltonian.pkl", "Zixiang_icf_d3.6/no_penalty_ckpt_24000.pkl", "phiT_"+str(i)+".dat")

# writeUHFSD_fromHAFQMC_icfTrans(mol, canonic, "Zixiang_icf_d3.6/AFQMC_hamiltonian.pkl", "Zixiang_icf_d3.6/no_penalty_ckpt_24000.pkl", "phi.dat")

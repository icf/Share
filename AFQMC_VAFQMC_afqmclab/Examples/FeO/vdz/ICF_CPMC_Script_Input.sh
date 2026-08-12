###############################
## set model parameters

#Jastrow data
Jastrow_rhf_address="Zixiang_icf_FeO/AFQMC_rhf.chk"
Jastrow_model_address="Zixiang_icf_FeO/AFQMC_hamiltonian.pkl"
Jastrow_address="Zixiang_icf_FeO/ckpt_13000.pkl"

## set AFQMC parameters
dt_list=("0.01")
thermalSize="2000"
## set Metro parameters
JastrowExpM="2"
numOfChains_list=("30")
numOfThermalSweeps_list=("1")
numOfBrackets_list=("1")
numOfSweepMeasurements_list=("100")

MetroUpdateSkip='0'

numOfReleasedSlice_list=("1")

###############################
#AFQMC code
# codeAFQMC='/public1/home/sc31257/Zhi-Yu_Xiao/code/ICF_afqmcPhaselessWithMetroChains_ForGeneralHamiltonian_withHAFQMCInput_1_3_2025_Formal/build/afqmcPhaseless'
codeAFQMC='ICF_afqmcPhaselessWithMetroChains_ForGeneralHamiltonian_withHAFQMCInput_4_28_2025_Formal_signCheck3'
codeRun='_ckpt_13000'
###############################
myAddress=`pwd`         
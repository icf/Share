###############################
## set model parameters

#Jastrow data
Jastrow_rhf_address="Zixiang_icf_TiO/AFQMC_rhf.chk"
Jastrow_model_address="Zixiang_icf_TiO/AFQMC_hamiltonian.pkl"
Jastrow_address="Zixiang_icf_TiO/checkpoint_10000.pkl"

## set AFQMC parameters
dt_list=("0.01")
thermalSize="4000"
## set Metro parameters
JastrowExpM="2"
numOfChains_list=("10" "20")
numOfThermalSweeps_list=("1")
numOfBrackets_list=("1")
numOfSweepMeasurements_list=("20")

MetroUpdateSkip='0'

numOfReleasedSlice_list=("1")

###############################
#AFQMC code
# codeAFQMC='/public1/home/sc31257/Zhi-Yu_Xiao/code/ICF_afqmcPhaselessWithMetroChains_ForGeneralHamiltonian_withHAFQMCInput_1_3_2025_Formal/build/afqmcPhaseless'
codeAFQMC='ICF_afqmcPhaselessWithMetroChains_ForGeneralHamiltonian_withHAFQMCInput_1_7_2025_Formal'
codeRun=''
###############################
myAddress=`pwd`         
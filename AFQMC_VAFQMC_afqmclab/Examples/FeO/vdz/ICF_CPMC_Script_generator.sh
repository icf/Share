#!/bin/sh
set -o nounset
set -o errexit

source ./ICF_CPMC_Script_Input.sh

temp_askFlag='0'
temp_runFlag='0'

address1=
for dt in ${dt_list} 
do
for numOfChains in ${numOfChains_list[@]} 
do
for numOfThermalSweeps in ${numOfThermalSweeps_list} 
do
for numOfBrackets in ${numOfBrackets_list} 
do
for numOfSweepMeasurements in ${numOfSweepMeasurements_list} 
do
for numOfReleasedSlice in ${numOfReleasedSlice_list} 
do
   echo 'dt: '${dt}' numOfChains: '${numOfChains}' numOfBrackets: '${numOfBrackets}' numOfThermalSweeps: '${numOfThermalSweeps}' numOfSweepMeasurements: '${numOfSweepMeasurements}' numOfReleasedSlice: '${numOfReleasedSlice}
   taskName='dt'${dt}'_Metro'${numOfChains}'_'${numOfBrackets}'_'${numOfThermalSweeps}'_'${numOfSweepMeasurements}'_release_'${numOfReleasedSlice}
   address1=${myAddress}'/'${taskName}'/'${codeAFQMC}${codeRun}'/'
   
   if [ ! -d "${address1}" ]; then
      mkdir -p ${address1}
   fi

   cp -R ICF_CPMC_Input/* ${address1}/
   cd ${address1}
   
   cp ${address1}'generateInputFile_templet.py' 'generateInputFile.py'
   sed -i s,_icf_Jastrow_rhf_address_,${Jastrow_rhf_address},g generateInputFile.py  
   sed -i s,_icf_Jastrow_model_address_,${Jastrow_model_address},g generateInputFile.py  
   sed -i s,_icf_Jastrow_address_,${Jastrow_address},g generateInputFile.py 
   sed -i s/_icf_JastrowExpM_/${JastrowExpM}/g generateInputFile.py 
   sed -i s/_icf_dt_/${dt}/g generateInputFile.py 
   sed -i s/_icf_thermalSize_/${thermalSize}/g generateInputFile.py 
   sed -i s/_icf_MetroUpdateSkip_/${MetroUpdateSkip}/g generateInputFile.py 
   sed -i s/_icf_numOfChains_/${numOfChains}/g generateInputFile.py 
   sed -i s/_icf_numOfThermalSweeps_/${numOfThermalSweeps}/g generateInputFile.py 
   sed -i s/_icf_numOfBrackets_/${numOfBrackets}/g generateInputFile.py 
   sed -i s/_icf_numOfSweepMeasurements_/${numOfSweepMeasurements}/g generateInputFile.py 
   sed -i s/_icf_numOfReleasedSlice_/${numOfReleasedSlice}/g generateInputFile.py 

   cp run_templet run
   sed -i s/_icf_codeAFQMC_/${codeAFQMC}/g run  
   if [[ ${temp_askFlag} = '0' ]]; then
      temp_askFlag='1'
      read -p "Submit jobs? " choice
      case "${choice}" in 
         y|Y ) 
               echo "yes"
               temp_runFlag='1';;
         n|N ) echo "no"
               temp_runFlag='0';;
         * ) echo "invalid"
               exit 0;;
      esac
   fi
   if [[ ${temp_runFlag} = '1' ]]; then
      sbatch run
   fi
   cd ${myAddress}
done
done
done
done
done
done
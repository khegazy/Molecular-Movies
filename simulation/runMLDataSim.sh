#!/bin/bash -x

FILETORUN=n2oMLDataSim.exe
OUTPUTFILE=n2oMLData
OUTPUTDIR=/reg/neh/home/khegazy/analysis/machineLearning/simulation/output/
CLUSTER=0
INDEX=0


NMOLS=2
if [ ! -z "$1" ]; then
  NMOLS=${1}
fi


if [ ! -d "${OUTPUTDIR}" ]; then
  mkdir ${OUTPUTDIR}
fi 

if [ ${CLUSTER} -eq 1 ]; then
  echo "RUNNING ON CLUSTER"

  if [ ! -z "$2" ]; then
    INDEX=${2}
  else
    echo "ERROR: Must run cluster jobs with index argument!!!"
    exit
  fi
fi


./${FILETORUN} ${NMOLS} -Ofile ${OUTPUTFILE} -Odir ${OUTPUTDIR} -Index ${INDEX}



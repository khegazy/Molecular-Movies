#!/bin/bash -x

MAINDIR=/reg/neh/home/khegazy/analysis/machineLearning/simulation
RUNDIR=${MAINDIR}/cluster
FILETORUN=runMLDataSim.sh
EXECUTABLE=n2oMLDataSim.exe

if [ -z "$2" ]; then
  echo "ERROR SUBMITTING JOBS!!!   Must give the number of jobs and number of trajectories"
  exit
fi

# Number of jobs to run
NJOBS=${1}

# Number of trajectories
NMOLS=${2}

# Set the output directory for results
if [ ! -z "$3" ]; then
  OUTPUTDIR=${MAINDIR}/output/${3}
  RUNDIR=${RUNDIR}/${3}
else
  OUTPUTDIR=${MAINDIR}/output
fi

if [ ! -d "${OUTPUTDIR}" ]; then
  mkdir ${OUTPUTDIR}
fi

make clean; make

if [ ! -d "${RUNDIR}" ]; then
  mkdir ${RUNDIR}
  mkdir ${RUNDIR}/logs
  mkdir ${RUNDIR}/run
else
  if [ ! -d "logs" ]; then
    mkdir ${RUNDIR}/logs
    mkdir ${RUNDIR}/run
  fi
fi

cd ${RUNDIR}/run
cp ${MAINDIR}/${FILETORUN} .
cp ${MAINDIR}/${EXECUTABLE} .
sleep 5

sed -i 's/CLUSTER=0/CLUSTER=1/g' ${FILETORUN}


######################
##  Submitting Jobs ##
######################

job=1
until [ $job -gt $NJOBS ]; do
  bsub -q psanaq -o "../logs/output"${job}".log" ./${FILETORUN} ${NMOLS} ${job}
  job=$((job + 1))
done

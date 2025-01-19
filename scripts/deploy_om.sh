#!/bin/bash

usage()
{
cat << EOF
$(basename "$0"):
  Script to deploy OM as a hutch operator.

  OPTIONS:
    -h|--help
      Definitions of options
    -e|--experiment
      Experiment to deploy OM for
    -d|--detector
      Detector to be used
    -a|--autosfx-directory
      Path to AutoSFX directory (optional)
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    -h|--help)
      usage
      exit
      ;;
    -e|--experiment)
      EXPT="$2"
      shift
      shift
      ;;
    -d|--detector)
      DETECTOR="$2"
      shift
      shift
      ;;
    -a|--autosfx-directory)
      AUTOSFX_DIR="$2"
      shift
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

# Hard fail if experiment and detector are not provided
if [ -z ${EXPT+x} ]; then
  echo "ERROR! experiment name not provided!"
  usage
  exit
fi
if [ -z ${DETECTOR+x} ]; then
  echo "ERROR! detector name not provided!"
  usage
  exit
fi

AUTOSFX_DIR=${AUTOSFX_DIR:='/cds/sw/package/autosfx/'}

# 0. Identify the operator
OPR=`whoami`
GET_IN=false; for operator in mfxopr cxiopr; do if [ $OPR == $operator ]; then GET_IN=true ; fi; done
if [ $GET_IN = false ]; then
  echo "ERROR: this script must be executed by the hutch operator (CXI or MFX for now)."
  exit
fi
HUTCH=${OPR:0:3}

# 1. Create and populate workspace directory
WORKSPACE_DIR="/cds/home/opr/${OPR}/OM-GUI/${EXPT}/om-workspace/"
if [ -d $WORKSPACE_DIR ]; then
  echo "WARNING: ${WORKSPACE_DIR} already exists. "
else
  mkdir -p ${WORKSPACE_DIR}
  echo "INFO: created ${WORKSPACE_DIR}"
fi

for file in run_om.sh monitor.yaml; do
  file_path="${AUTOSFX_DIR}/omdevteam.github.io/html/files/lcls/${HUTCH}/${file}"
  if [ ! -f ${file_path} ]; then
    echo "ERROR: could not find ${file_path}. Abort..."
    exit
  fi
  cp ${file_path} ${WORKSPACE_DIR}/
  echo "INFO: copied ${file} to ${WORKSPACE_DIR}"
done

# 2. Retrieve latests mask and geometry
for item in masks geometries; do
  file_path="${AUTOSFX_DIR}/mrxv/${item}/${DETECTOR}_latest.*"
  if [ ! -f ${file_path} ]; then
    echo "WARNING! no ${item} found at: ${file_path}"
  fi
  cp ${file_path} ${WORKSPACE_DIR}/
  echo "INFO: copied ${file_path} to ${WORKSPACE_DIR}"
done

# 3. Edit run_om.sh
if [ ${HUTCH} == "cxi" ]; then
  TEMPLATE_MON_NODE_LIST="daq-cxi-mon01,daq-cxi-mon18,daq-cxi-mon19"
elif [ ${HUTCH} == "mfx" ]; then
  TEMPLATE_MON_NODE_LIST="daq-mfx-mon02,daq-mfx-mon03,daq-mfx-mon04,daq-mfx-mon05"
fi
MON_NODE_LIST=`wherepsana`
sed -i "s/host ${TEMPLATE_MON_NODE_LIST}/host ${MON_NODE_LIST}/g" ${WORKSPACE_DIR}/run_om.sh

# 4. Edit monitor.yaml
if [ ${HUTCH} == "cxi" ]; then
  TEMPLATE_EXPT="cxilv4418"
  TEMPLATE_DETECTOR="jungfrau4M"
  TEMPLATE_DETECTOR_PV="CXI:DS1:MMS:06.RBV"
  DETECTOR_PV=${TEMPLATE_DETECTOR_PV}
elif [ ${HUTCH} == "mfx" ]; then
  TEMPLATE_EXPT="mfxlx4219"
  TEMPLATE_DETECTOR="epix10k2M"
  TEMPLATE_DETECTOR_PV="MFX:DET:MMS:04.RBV"
  if [ ${DETECTOR} == 'epix10k2M' ]; then
    DETECTOR_PV="MFX:ROB:CONT:POS:Z"
  else
    DETECTOR_PV=${TEMPLATE_DETECTOR_PV}
  fi
fi
sed -i "s/${TEMPLATE_EXPT}/${EXPT}/g" ${WORKSPACE_DIR}/monitor.yaml
sed -i "s/bad_pixel_map_filename: null/bad_pixel_map_filename: ${DETECTOR}_latest.h5/g" ${WORKSPACE_DIR}/monitor.yaml
sed -i "s/${TEMPLATE_DETECTOR}/${DETECTOR}/g" ${WORKSPACE_DIR}/monitor.yaml
sed -i "s/${TEMPLATE_DETECTOR_PV}/${DETECTOR_PV}/g" ${WORKSPACE_DIR}/monitor.yaml

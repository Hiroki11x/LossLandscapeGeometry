#!/bin/bash
#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:2 
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -o slurm-%j.out

# ======== Module, Virtualenv and Other Dependencies ======
source ../../env/cluster_env.sh
echo "PYTHON Environment: $PYTHON_PATH"
export PYTHONPATH=.
export PATH=$PYTHON_PATH:$PATH

# ======== Configuration ========
PROGRAM="main_language_model.py"
pushd ../../src
PYTHON_ARGS=$@

# ======== Execution ========
CMD="python ${PROGRAM} ${PYTHON_ARGS}"
echo $CMD
eval $CMD

popd

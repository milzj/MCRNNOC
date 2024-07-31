#!/bin/bash

#SBATCH -J SimExp                    # Job name
#SBATCH --account=gts-jmilz3                     # charge account
#SBATCH -N 40 --ntasks-per-node=1                  # number of nodes and cores per node
#SBATCH --mem-per-cpu=16gb                        # Memory per core
#SBATCH -qinferno                               # QOS Name
#SBATCH -o output/Report-%x-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=ALL              # Mail preferences
#SBATCH --mail-user=jmilz3@gatech.edu           # E-mail address for notifications
#SBATCH --time=08:00:00

cd $HOME/MCRNNOC/src/mcrnnoc/examples/bilinear/risk_neutral

module load anaconda3/2022.05.0.1
conda activate MCRNNOC
module load gcc

export PYTHONPATH=$HOME/MCRNNOC:${PYTHONPATH}
export PYTHONPATH=$HOME/MCRNNOC/src:${PYTHONPATH}

date=$(date '+%d-%b-%Y-%H-%M-%S')

source ../../simulation_data.sh

experiment="Monte_Carlo_Rate_Test"
experiment="Monte_Carlo_Rate_Fixed_Control"
experiment="Monte_Carlo_Rate"
mpiexec -n 40 python simulate_experiment.py $date $experiment $Nref
python plot_experiment.py "output/Experiments/$experiment""_$date"















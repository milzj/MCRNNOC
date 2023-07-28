
date=$(date '+%d-%b-%Y-%H-%M-%S')
source ../../simulation_data.sh

experiment="Monte_Carlo_Rate_Test"
experiment="Monte_Carlo_Rate_Synthetic"
Nref="8"
mpiexec -n 2 python simulate_experiment.py $date $experiment $Nref












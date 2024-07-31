
date=$(date '+%d-%b-%Y-%H-%M-%S')
source ../../simulation_data.sh

experiment="Monte_Carlo_Rate_Synthetic"
experiment="Monte_Carlo_Rate_Fixed_Control_Test"
experiment="Monte_Carlo_Rate_Test"
Nref="8"
mpiexec -n 2 python simulate_experiment.py $date $experiment $Nref

python plot_experiment.py "output/Experiments/$experiment""_$date"












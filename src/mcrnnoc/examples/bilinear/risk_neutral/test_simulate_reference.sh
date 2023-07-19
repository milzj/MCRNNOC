date=$(date '+%d-%b-%Y-%H-%M-%S')

source ../../simulation_data.sh
n=128
N=128
mpiexec -n 8 python simulate_reference.py $n $N $date











date=$(date '+%d-%b-%Y-%H-%M-%S')

source ../../simulation_data.sh
n=64
N=512
mpiexec -n 8 python simulate_reference.py $n $N $date











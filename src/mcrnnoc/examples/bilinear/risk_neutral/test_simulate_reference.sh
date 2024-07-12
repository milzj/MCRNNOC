date=$(date '+%d-%b-%Y-%H-%M-%S')

source ../../simulation_data.sh
n=32
N=64
mpirun -np 2 python simulate_reference.py $n $N $date











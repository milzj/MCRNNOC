date=$(date '+%d-%b-%Y-%H-%M-%S')

source ../../simulation_data.sh
n=$nref
N=$Nref
mpiexec -n 4 python simulate_reference.py $n $N $date











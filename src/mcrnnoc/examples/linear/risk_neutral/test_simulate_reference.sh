date=$(date '+%d-%b-%Y-%H-%M-%S')

source ../../simulation_data.sh
n=32
N=64
mpirun -np 2 python simulate_reference.py $n $N $date
dir="Reference_Simulation_n=$n"_N="$N""_date=$date"
python ../plot_control.py "$dir/$date""_reference_solution_mpi_rank=0_N=$N""_n=$n"






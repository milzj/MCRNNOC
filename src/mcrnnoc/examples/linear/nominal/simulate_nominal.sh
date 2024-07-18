date=$(date '+%d-%b-%Y-%H-%M-%S')
source ../../simulation_data.sh
python simulate_nominal.py $nref $date

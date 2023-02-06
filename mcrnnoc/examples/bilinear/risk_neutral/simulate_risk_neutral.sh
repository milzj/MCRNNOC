date=$(date '+%d-%b-%Y-%H-%M-%S')
source ../../simulation_data.sh
N=64
python simulate_risk_neutral.py $nref $date $N

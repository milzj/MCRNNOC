date=$(date '+%d-%b-%Y-%H-%M-%S')
source ../../simulation_data.sh

for seed in 1 1000 2000 3000
do
    for N in 256 512
    do
        python simulate_risk_neutral.py $nref $date $N $seed
    done
done

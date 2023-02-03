#!/bin/bash

rm -rf exp_random_field

python plot_exp_random_field.py

cd exp_random_field

convert -dispose 2 -delay 80 -loop 0 exp_kappa_sample\=*.png exp_random_field.gif

cd ..


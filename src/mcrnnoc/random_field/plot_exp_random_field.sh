#!/bin/bash

rm -rf exp_random_field

time python plot_exp_random_field.py 2>&1 | tee exp_random_field.txt

cd exp_random_field_plots

convert -dispose 2 -delay 80 -loop 0 exp_kappa_sample\=*.png exp_random_field.gif

cd ..


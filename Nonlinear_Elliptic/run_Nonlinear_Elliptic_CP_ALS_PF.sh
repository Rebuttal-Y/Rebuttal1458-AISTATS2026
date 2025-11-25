#!/bin/bash

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR

declare -i count=0

n_x1=( 49 )
n_x2=( 49 )

n_x1ind=( 67 )
n_x2ind=( 67 )

ks=( 5 )
jitt=( 23.5 )
ls=( -2.311 )

rank=( 10 )

lam1=( 700000100.0 )
lam2=( 14900.0 )



store_path=$SCRIPT_DIR"/result/Data_NElliptic_CP_ALS_PF"
store=$store_path
if [ ! -d $store ]; then
    mkdir -p $store
fi


for nxdx in "${!n_x1[@]}"
do
    for jdx in "${!jitt[@]}"
    do
        for lsdx in "${!ls[@]}"
        do
            for ksdx in "${!ks[@]}"
            do
                for l1dx in "${!lam1[@]}"
                do
                    for l2dx in "${!lam2[@]}"
                    do
                        for rdx in "${!rank[@]}"
                        do
                            python Solve_nonLElliptic.py --epochs=1000000 --early-stop=5000 --log-store-path=$store_path"/config"$count \
                             --kernel-s ${ks[$ksdx]} ${ks[$ksdx]} --jitter ${jitt[$jdx]} ${jitt[$jdx]} --log-lsx ${ls[$lsdx]} ${ls[$lsdx]} \
                            --rank=${rank[$rdx]} --n-xtrain ${n_x1[$nxdx]} ${n_x2[$nxdx]} --n-xind ${n_x1ind[$nxdx]} ${n_x2ind[$nxdx]} --log-interval=5000 --lam1=${lam1[$l1dx]} \
                            --lam2=${lam2[$l2dx]} --n-train-collocation=$((n_x1[$nxdx] * n_x2[$nxdx])) --random-sampling --analysis
                            count=$(( count + 1 ))
                        done
                    done
                done
            done
        done
    done
done

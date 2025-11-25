#!/bin/bash

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR

declare -i count=0

n_x1=( 49 )
n_x2=( 49 )

n_x1ind=( 102 )
n_x2ind=( 102 )

ks1=( 5 )
ks2=( 5 )
jitt1=( 21.589 )
jitt2=( 21.589 )
ls1=( -3.2201 )
ls2=( -3.2201 )

rank=( 12 )

lam1=( 199900000.0 )
lam2=( 45000.0 )

a=( 20.0 )


store_path=$SCRIPT_DIR"/result/Data_AllenCahen_CP_ALS_NT_a20"
store=$store_path
if [ ! -d $store ]; then
    mkdir -p $store
fi


for nxdx in "${!n_x1[@]}"
do
    for jdx in "${!jitt1[@]}"
    do
        for lsdx in "${!ls1[@]}"
        do
            for ksdx in "${!ks1[@]}"
            do
                for l1dx in "${!lam1[@]}"
                do
                    for l2dx in "${!lam2[@]}"
                    do
                        for rdx in "${!rank[@]}"
                        do
                            for adx in "${!a[@]}"
                            do
                                python SolveAllenCahen.py --epochs=1000000 --early-stop=20000 --log-store-path=$store_path"/config"$count \
                                 --kernel-s ${ks1[$ksdx]} ${ks2[$ksdx]} --jitter ${jitt1[$jdx]} ${jitt2[$jdx]} --log-lsx ${ls1[$lsdx]} ${ls2[$lsdx]} \
                                 --rank=${rank[$rdx]} --n-xtrain ${n_x1[$nxdx]} ${n_x2[$nxdx]} --n-xind ${n_x1ind[$nxdx]} ${n_x2ind[$nxdx]} --log-interval=5000 --lam1=${lam1[$l1dx]} \
                                 --lam2=${lam2[$l2dx]} --a=${a[$adx]} --n-train-collocation=$((n_x1[$nxdx] * n_x2[$nxdx])) --random-sampling --Newton-M --analysis
                                count=$(( count + 1 ))
                            done
                        done
                    done
                done
            done
        done
    done
done

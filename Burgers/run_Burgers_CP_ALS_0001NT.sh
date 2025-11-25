#!/bin/bash

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR

declare -i count=0

n_x1=( 200 )
n_x2=( 40 )

n_x1ind=( 215 )
n_x2ind=( 38 )

ks1=( 1 )
ks2=( 2 )
jitt1=( 13.98 )
jitt2=( 13.9 )
ls1=( -2.999 )
ls2=( -1.11 )

rank=( 19 )

lam1=( 11000.0 )
lam2=( 1.01 )

nu=( 0.001 )


store_path=$SCRIPT_DIR"/result/Data_Burgers0001_CP_ALS_NT"
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
                            for ndx in "${!nu[@]}"
                            do
                                python SolveBurgers.py --epochs=1000000 --early-stop=20000 --log-store-path=$store_path"/config"$count \
                                 --kernel-s ${ks1[$ksdx]} ${ks2[$ksdx]} --jitter ${jitt1[$jdx]} ${jitt2[$jdx]} --log-lsx ${ls1[$lsdx]} ${ls2[$lsdx]} \
                                 --rank=${rank[$rdx]} --n-xtrain ${n_x1[$nxdx]} ${n_x2[$nxdx]} --n-xind ${n_x1ind[$nxdx]} ${n_x2ind[$nxdx]} --log-interval=100 --lam1=${lam1[$l1dx]} \
                                 --lam2=${lam2[$l2dx]} --nu=${nu[$ndx]} --n-train-collocation=$((n_x1[$nxdx] * n_x2[$nxdx])) --Newton-M --analysis
                                count=$(( count + 1 ))
                            done
                        done
                    done
                done
            done
        done
    done
done

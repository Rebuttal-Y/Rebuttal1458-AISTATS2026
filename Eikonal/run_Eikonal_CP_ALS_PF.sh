#!/bin/bash

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR

declare -i count=0

n_x=( 35 )

n_x1ind=( 36 )
n_x2ind=( 36 )

ks=( 5 )

jitt=( 21.611 )
ls=( -2.4 )

rank=( 10 )

lam1=( 100000000000.0 )
lam2=( 1500.0 )


store_path=$SCRIPT_DIR"/result/Data_Eikonal_CP_ALS_PF"
store=$store_path
if [ ! -d $store ]; then
    mkdir -p $store
fi


for nxdx in "${!n_x[@]}"
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
                            python SolveEikonal.py --epochs=1000000 --early-stop=20000 --log-store-path=$store_path"/config"$count \
                             --kernel-s ${ks[$ksdx]} ${ks[$ksdx]} --jitter ${jitt[$jdx]} ${jitt[$jdx]} --log-lsx ${ls[$lsdx]} ${ls[$lsdx]} \
                             --rank=${rank[$rdx]} --n-xtrain ${n_x[$nxdx]} ${n_x[$nxdx]} --n-xind ${n_x1ind[$nxdx]} ${n_x2ind[$nxdx]} --log-interval=5000 \
                             --lam1=${lam1[$l1dx]} --lam2=${lam2[$l2dx]} --n-xtest=100 --n-train-collocation=$((n_x[$nxdx] * n_x[$nxdx])) --analysis
                            count=$(( count + 1 ))
                        done
                    done
                done
            done
        done
    done
done

#!/bin/bash

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR

declare -i count=0

n_x1=( 200 )
n_x2=( 40 )

n_x1ind=( 210 )
n_x2ind=( 43 )

ks1=( 1 )
ks2=( 2 )
jitt1=( 6.8124 )
jitt2=( 11.5129 )
ls1=( -2.999 )
ls2=( -1.1 )

rank=( 26 )

lam1=( 2000.0 )
lam2=( 1.0 )

nu=( 0.001 )


store_path=$SCRIPT_DIR"/result/Data_Burgers0001_CP_ALS_PF"
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
                                 --lam2=${lam2[$l2dx]} --nu=${nu[$ndx]} --n-train-collocation=$((n_x1[$nxdx] * n_x2[$nxdx])) --analysis
                                count=$(( count + 1 ))
                            done
                        done
                    done
                done
            done
        done
    done
done

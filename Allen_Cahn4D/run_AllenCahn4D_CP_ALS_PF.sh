#!/bin/bash

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR

declare -i count=0

n_train_col=( 16000 )
n_train_bound=( 3200 )

n_xind=( 60 )
ks=( 5 )

jitt=( 21.59 )
ls=( -3.009 )

rank=( 10 )

lam1=( 21430000000.0 )
lam2=( 1500.0 )

a=( 15.0 )


store_path=$SCRIPT_DIR"/result/Data_AllenCahn4D_CP_ALS_PF"
store=$store_path
if [ ! -d $store ]; then
    mkdir -p $store
fi

dataset_path=$SCRIPT_DIR"/dataset/Dim4_3"

for nxdx in "${!n_train_col[@]}"
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
                            for adx in "${!a[@]}"
                            do
                                python SolveAllenCahn4D_CP.py --epochs=1000000 --early-stop=10000 --log-store-path=$store_path"/config"$count \
                                 --kernel-s ${ks[$ksdx]} ${ks[$ksdx]} ${ks[$ksdx]} ${ks[$ksdx]} --jitter ${jitt[$jdx]} ${jitt[$jdx]} ${jitt[$jdx]} \
                                 ${jitt[$jdx]} --log-lsx ${ls[$lsdx]} ${ls[$lsdx]} ${ls[$lsdx]} ${ls[$lsdx]} --rank=${rank[$rdx]} \
                                 --n-xind ${n_xind[$nxdx]} ${n_xind[$nxdx]} ${n_xind[$nxdx]} ${n_xind[$nxdx]} --log-interval=1000 --lam1=${lam1[$l1dx]} \
                                 --lam2=${lam2[$l2dx]} --a=${a[$adx]} --test-dataset-load-path=$SCRIPT_DIR"/dataset/Allen_Cahen_Test1000000.npz" \
                                 --dataset-load-path=$dataset_path"/Allen_CahenC"${n_train_col[$ndx]}"_B"$((((${n_train_bound[$ndx]} + 4 * 2 - 1)) / ((4 * 2)) * ((4 * 2))))".npz" \
                                 --random-sampling --seed=10859 --analysis
                                count=$(( count + 1 ))
                            done
                        done
                    done
                done
            done
        done
    done
done

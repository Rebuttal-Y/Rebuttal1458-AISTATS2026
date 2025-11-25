#!/bin/bash

round() {
  printf "%.${2}f" "${1}"
}

CURRENT_FOLDER=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $SCRIPT_DIR

declare -i count=0

n_train_col=( 16000 )
n_train_bound=( 3200 )
n_xind=( 140 )
ks=( 5 )

jitt=( 21.4 )
ls=( -0.3 )

rank=( 7 )

lam1=( 1000000000.0 )
lam2=( 14950.0 )

beta=( 6.0 )

Order=( 6 )


store_path=$SCRIPT_DIR"/result/Data_DarcyFlow6D_TR_ALS_NT"
store=$store_path
if [ ! -d $store ]; then
    mkdir -p $store
fi

dataset_path=$SCRIPT_DIR"/dataset/Dim6"


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
                            for bdx in "${!beta[@]}"
                            do
                                store_directory=$store_path"/Result_beta"$(round ${beta[$bdx]} 0)"_C"${n_train_col[$nxdx]}"_B"$((((${n_train_bound[$nxdx]} + 6 * 2 - 1)) / ((6 * 2)) * ((6 * 2))))
                                if [ ! -d $store_directory ]; then
                                    mkdir -p $store_directory
                                fi
                                python SolveDarcyFlow6D_TR.py --epochs=1000000 --early-stop=5000 --log-store-path=$store_directory"/config"$count \
                                 --kernel-s ${ks[$ksdx]} ${ks[$ksdx]} ${ks[$ksdx]} ${ks[$ksdx]} ${ks[$ksdx]} ${ks[$ksdx]} \
                                 --jitter ${jitt[$jdx]} ${jitt[$jdx]} ${jitt[$jdx]} ${jitt[$jdx]} ${jitt[$jdx]} ${jitt[$jdx]} \
                                 --log-lsx ${ls[$lsdx]} ${ls[$lsdx]} ${ls[$lsdx]} ${ls[$lsdx]} ${ls[$lsdx]} ${ls[$lsdx]} --n-train-batch=16000 \
                                 --rank ${rank[$rdx]} ${rank[$rdx]} ${rank[$rdx]} ${rank[$rdx]} ${rank[$rdx]} ${rank[$rdx]} --n-xind ${n_xind[$nxdx]} ${n_xind[$nxdx]} ${n_xind[$nxdx]} ${n_xind[$nxdx]} ${n_xind[$nxdx]} ${n_xind[$nxdx]} \
                                 --log-interval=100 --lam1=${lam1[$l1dx]} --lam2=${lam2[$l2dx]} --test-dataset-load-path=$dataset_path"/DarcyFlow_Houman_Test1000000.npz" \
                                 --dataset-load-path=$dataset_path"/Beta"$(round ${beta[$bdx]} 0)"_0/DarcyFlow_HoumanC"${n_train_col[$nxdx]}"_B"$((((${n_train_bound[$nxdx]} + 6 * 2 - 1)) / ((6 * 2)) * ((6 * 2))))"_Trace.npz" \
                                 --Newton-M --seed=12058 --analysis
                                count=$(( count + 1 ))
                            done
                        done
                    done
                done
            done
        done
    done
done

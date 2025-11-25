import argparse

from dataProviderL import DataGen
from model_running import model_run
from model_CP import GP_ALS_6D
from utilities import *




def test(args):
    data_gen = DataGen(args)
    write_log(args)
    gp = GP_ALS_6D(args, data_gen)

    model_run(gp, data_gen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GP Solver for DarcyFlow Houman Using ALS on CP Decomposition')
    parser.add_argument('--n-test', type=int, default=1000, metavar='N',
                        help='number of test points selected for evaluation(default: 500)')
    parser.add_argument('--test-batch', type=int, default=1000, metavar='N',
                        help='testing batch during testing phase (default: 10000)')
    parser.add_argument('--n-train-boundary', type=int, default=280, metavar='N',
                        help='number of boundary points for training (default: 10000)')
    parser.add_argument('--n-train-collocation', type=int, default=1000, metavar='N',
                        help='number of collocation points for training (default: 10000)')
    parser.add_argument('--n-train-batch', type=int, default=48000, metavar='N',
                        help='training batch in calculating Phi2 (default: 32000)')
    parser.add_argument('--x-range', type=str, default="0.0,1.0", metavar='TUPLE',
                        help='position range for the experiment')


    parser.add_argument('--beta', type=float, default=1.0, metavar='N',
                        help='coefficient controlling the frequency of solution(default: 1.0)')
    parser.add_argument('--n-xind', nargs='+', default=[700, 40, 700, 40, 700, 40], type=int,
                        help='num of inducing points by each dim ')

    
    parser.add_argument('--jitter', nargs='+', default=[23.0, 13.1, 13.1, 13.1, 13.1, 13.1], type=float, 
                        help='Positive Definite for Kernel (default: 23.0)')
    parser.add_argument('--log-lsx', nargs='+', default=[-4.5, -1.5, -4.5, -1.5, -4.5, -1.5], type=float,
                        help='log landscale per dim (default: -4.5)')
    parser.add_argument('--kernel-s', nargs='+', default=[1, 3, 1, 3, 1, 3], type=int, 
                        help='kernel selection: 0: 0.5, 1: 1.5, 2: 2.5, 3: 3.5, 4: 4.5, 5: inf')

    parser.add_argument('--rank', type=int, default=10, metavar='N',
                        help='CP rank per dimension (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='maximum round to do ALS CP (default: 100)')
    parser.add_argument('--lam1', type=float, default=100000.0, metavar='L1', 
                        help='Coefficient Weight for lam1 (default: 100000)')
    parser.add_argument('--lam2', type=float, default=100.0, metavar='L2', 
                        help='Coefficient Weight for lam2 (default: 100)')

    parser.add_argument('--stop-criteria', type=float, default=0.000000001, metavar='IC',
                        help='iteration stop criteria (default: 1e-6)')
    parser.add_argument('--log-store-path', type=str, default="./result/test", metavar='FILE',
                        help='path to store the training record and test result')
    parser.add_argument('--dataset-load-path', type=str, default="./dataset/Dim6/Beta6/DarcyFlow_HoumanC1000_B204_Trace.npz", metavar='FILE',
                        help='path to load the training dataset')
    parser.add_argument('--test-dataset-load-path', type=str, default="./dataset/Dim6/Beta6/DarcyFlow_Houman_Test1000000.npz", metavar='FILE',
                        help='path to load the test dataset')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--early-stop', type=int, default=5000, metavar='N',
                        help='number of epochs traced for update')
    parser.add_argument('--random-sampling', action="store_true",
                        help='choose to use random sampling for boundary and collocation points gen')
    parser.add_argument('--CPU-PDE', action="store_true",
                        help='Whether or not putting Matrix Calulation on CPU device')
    parser.add_argument('--analysis', action="store_true",
                        help='Whether or not providing analysis for better tuning')
    parser.add_argument('--Newton-M', action="store_true",
                        help='whether or not using Newton Method to linearize PDE')
    parser.add_argument('--n-order', type=int, default=6, metavar='N',
                        help='number of order in the equation (default: 6)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')

    args = parser.parse_args()
    np.random.seed(args.seed)
    tl.set_backend('jax')
    
    test(args)

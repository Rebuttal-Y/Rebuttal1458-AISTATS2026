import argparse

from dataProviderL import DataGen
from model_running import model_run
from model import GP_ALS
from utilities import *

#use Jax_env
#Implementation of TGP Solver with Piccard's iteration for solving the AllenCahen
#u_xx + u_yy + u**3 - u = f(x, y) with ground truth u = sin(2*pi*a*x)cos(2*pi*a*y) + sin(2*pi*x)cos(2*pi*y) with a = 15 and 20


def test(args):
    data_gen = DataGen(args)
    write_log(args)
    gp = GP_ALS(args, data_gen)


    model_run(gp, data_gen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GP Solver for AllenCahen Using ALS on CP Decomposition')
    parser.add_argument('--n-xtest', nargs='+', default=[100, 100], type=int,
                        help='num of test points per dimension (default: 100)')
    parser.add_argument('--n-train-boundary', type=int, default=280, metavar='N',
                        help='number of boundary points for training (default: 10000)')
    parser.add_argument('--n-train-collocation', type=int, default=1000, metavar='N',
                        help='number of collocation points for training (default: 10000)')
    parser.add_argument('--n-train-batch', type=int, default=40000, metavar='N',
                        help='training batch in calculating Phi2 (default: 20000)')
    parser.add_argument('--x-range', type=str, default="0.0,1.0", metavar='TUPLE',
                        help='position range for the experiment')


    parser.add_argument('--a', type=float, default=15.0, metavar='N',
                        help='coefficient in ground truth (default: 15.0)')
    parser.add_argument('--n-xtrain', nargs='+', default=[100, 100], type=int,
                        help='num of collocation points per dim ')
    parser.add_argument('--n-xind', nargs='+', default=[700, 40], type=int,
                        help='num of inducing points by each dim ')
    
    parser.add_argument('--jitter', nargs='+', default=[23.0, 13.1], type=float, 
                        help='Positive Definite for Kernel (default: 23.0)')
    parser.add_argument('--log-lsx', nargs='+', default=[-4.5, -1.5], type=float,
                        help='log landscale per dim (default: -4.5)')
    parser.add_argument('--kernel-s', nargs='+', default=[1, 3], type=int, 
                        help='kernel selection: 0: 0.5, 1: 1.5, 2: 2.5, 3: 3.5, 4: 4.5, 5: inf')

    parser.add_argument('--rank', type=int, default=10, metavar='N',
                        help='CP rank per dimension (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='maximum round to do ALS CP (default: 100)')
    parser.add_argument('--lam1', type=float, default=100000.0, metavar='L1', 
                        help='Coefficient Weight for lam1 (default: 100000)')
    parser.add_argument('--lam2', type=float, default=100.0, metavar='L2', 
                        help='Coefficient Weight for lam2 (default: 100)')

    parser.add_argument('--stop-criteria', type=float, default=0.000001, metavar='IC',
                        help='iteration stop criteria (default: 1e-6)')
    parser.add_argument('--log-store-path', type=str, default="./result/test", metavar='FILE',
                        help='path to store the training record and test result')

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
    parser.add_argument('--n-order', type=int, default=2, metavar='N',
                        help='number of order in the equation (default: 2)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')

    args = parser.parse_args()
    np.random.seed(args.seed)
    tl.set_backend('jax')
    
    test(args)

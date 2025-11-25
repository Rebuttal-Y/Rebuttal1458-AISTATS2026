from scipy.stats.qmc import LatinHypercube
import scipy.sparse
from scipy.sparse import diags
from scipy.sparse import identity
from itertools import product
from utilities import *

def solve_Eikonal(N, epsilon):
    hg = np.array(1.0/(N+1))
    x_grid = (np.arange(1,N+1,1))*hg
    # x_grid = (np.arange(0, N, 1))*hg
    a1 = np.ones((N, N+1))
    a2 = np.ones((N+1, N))

    # diagonal element of A
    a_diag = np.reshape(a1[:,:N]+a1[:,1:]+a2[:N,:]+a2[1:,:], (1,-1))
    
    # off-diagonals
    a_super1 = np.reshape(np.append(a1[:,1:N], np.zeros((N,1)), axis = 1), (1,-1))
    a_super2 = np.reshape(a2[1:N,:], (1,-1))
    
    A = diags([[-a_super2[np.newaxis, :]], [-a_super1[np.newaxis, :]], [a_diag], [-a_super1[np.newaxis, :]], [-a_super2[np.newaxis, :]]], [-N,-1,0,1,N], shape=(N**2, N**2), format = 'csr')
    XX, YY = np.meshgrid(x_grid, x_grid)
    f = np.zeros((N, N))
    f[0,:] = f[0,:] + epsilon**2 / (hg**2)
    f[N-1,:] = f[N-1,:] + epsilon**2 / (hg**2)
    f[:, 0] = f[:, 0] + epsilon**2 / (hg**2)
    f[:, N-1] = f[:, N-1] + epsilon**2 / (hg**2)
    fv = f.flatten()
    fv = fv[:, np.newaxis]
    
    mtx = identity(N**2)+(epsilon**2)*A/(hg**2)
    sol_v = scipy.sparse.linalg.spsolve(mtx, fv)
    # sol_v, exitCode = scipy.sparse.linalg.cg(mtx, fv)
    # print(exitCode)
    sol_u = -epsilon*np.log(sol_v)
    sol_u = np.reshape(sol_u, (N,N))
    return XX, YY, sol_u


class DataGen:

    def __init__(self, args):

        self.args = args
        self.x_range = [float(x) for x in self.args.x_range.split(',')]
        
        self.x_c, self.y_c = self.collocation_gen()
        
        self.x_b = self.boundary_gen()
        self.args.n_train_boundary = self.x_b.shape[0]
        
        self.X_test, self.Y_test = self._test_dataset_gen()

        self.YNorm2_test = np.linalg.norm(self.Y_test.reshape(-1), 2)
        self.YSUM_ABS = np.sum(np.abs(self.Y_test.reshape(-1)))
        
    def collocation_gen(self):
        if self.args.random_sampling:
            sampler = LatinHypercube(self.args.n_order, optimization='random-cd')
            X_samples = sampler.random(self.args.n_train_collocation)
            x1_sampling = self.x_range[0] + X_samples[:, 0:1] * (self.x_range[-1] - self.x_range[0])
            x2_sampling = self.x_range[0] + X_samples[:, 1:] * (self.x_range[-1] - self.x_range[0])
        else:
            x1_point_train = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtrain[0])
            x2_point_train = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtrain[-1])
            xx2, xx1 = np.meshgrid(x2_point_train, x1_point_train)
            x1_sampling, x2_sampling = xx1.reshape(-1, 1), xx2.reshape(-1, 1)
            self.args.n_train_collocation = self.args.n_xtrain[0] * self.args.n_xtrain[-1]
        X_train_col = np.concatenate((x1_sampling, x2_sampling), axis=-1)
        F_PDE = np.ones(self.args.n_train_collocation, dtype=np.float64)
        return X_train_col, F_PDE.reshape(-1)


    def boundary_gen(self):
        # X_lower_bound = np.array(self.x_range[0:1])
        # X_upper_bound = np.array(self.x_range[1:])
        X_bound = np.array(self.x_range)
        if self.args.random_sampling:
            # sampler = LatinHypercube(self.args.n_order-1, optimization='random-cd')
            
            # sampling y and get x boundary:
            # X_samples = sampler.random(self.args.n_xtrain[-1] - 1)
            X_samples = np.random.rand(self.args.n_xtrain[-1] - 1)
            # X_train_bound = np.array(list(product(X_bound, X_samples.reshape(-1))))
            X_train_bound = np.array(list(product(X_bound, X_samples.reshape(-1))))
            # X_samples = sampler.random(self.args.n_xtrain[-1] - 1)
            
            # X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_upper_bound, X_samples.reshape(-1))))), axis=0)
            
            # sampling x and get y boundary:
            # X_samples = sampler.random(self.args.n_xtrain[0] - 1) 
            X_samples = np.random.rand(self.args.n_xtrain[0] - 1)
            X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_samples.reshape(-1), X_bound)))), axis=0)
            # X_samples = sampler.random(self.args.n_xtrain[0] - 1) 
            # X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_samples.reshape(-1), X_upper_bound)))), axis=0)
        else:
            x1_point_train = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtrain[0])
            x2_point_train = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtrain[-1])
            
            # get x boundary:
            X_train_bound = np.array(list(product(X_bound, x2_point_train)))
        
            # get y boundary:
            X_train_bound = np.concatenate((X_train_bound, np.array(list(product(x1_point_train[1:-1], X_bound)))), axis=0)
           
        return X_train_bound
    
    def _test_dataset_gen(self):

        p1, p2, Y_test = solve_Eikonal(self.args.n_xtest, self.args.epsilon)

        X_test = np.concatenate((p1.reshape(-1, 1), p2.reshape(-1, 1)), axis=1)
        return X_test, Y_test.reshape(-1)

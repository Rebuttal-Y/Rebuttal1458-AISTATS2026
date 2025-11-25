from scipy.stats.qmc import LatinHypercube
from itertools import product
from utilities import *

gamma = 1.0
m = 3

def u_AllenC(a, x, y):
    return jnp.sin(2.0*jnp.pi*a*x) * jnp.cos(2.0*jnp.pi*a*y) + jnp.sin(2.0*jnp.pi*x) * jnp.cos(2.0*jnp.pi*y)

def pde_force_gen(a, x, y):
    duxx = -((2.0*jnp.pi)**2) * (jnp.sin(2.0*jnp.pi*a*x) * jnp.cos(2.0*jnp.pi*a*y) * (a**2) + jnp.sin(2.0*jnp.pi*x) * jnp.cos(2.0*jnp.pi*y))
    u_val = jnp.sin(2.0*jnp.pi*a*x) * jnp.cos(2.0*jnp.pi*a*y) + jnp.sin(2.0*jnp.pi*x) * jnp.cos(2.0*jnp.pi*y)
    return 2 * duxx + gamma * ((u_val**m) - u_val)


class DataGen:

    def __init__(self, args):

        self.args = args
        self.x_range = [float(x) for x in self.args.x_range.split(',')]

        self.x_c, self.y_c = self.collocation_gen()
        
        self.x_b, self.y_b = self.boundary_gen()
        self.args.n_train_boundary = self.x_b.shape[0]
        self.X_test, self.Y_test = self.test_data_gen()

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
        F_PDE = vmap(pde_force_gen, (None, 0, 0))(self.args.a, x1_sampling, x2_sampling)
        return X_train_col, F_PDE.reshape(-1)


    def boundary_gen(self):
        X_lower_bound = np.array(self.x_range[0:1])
        X_upper_bound = np.array(self.x_range[1:])
        X_bound = np.array(self.x_range)
        if self.args.random_sampling:
            sampler = LatinHypercube(self.args.n_order-1, optimization='random-cd')
            
            # sampling y and get x boundary:
            X_samples = sampler.random(self.args.n_xtrain[-1] - 1)
            # X_train_bound = np.array(list(product(X_bound, X_samples.reshape(-1))))
            X_train_bound = np.array(list(product(X_bound, X_samples.reshape(-1))))
            # X_samples = sampler.random(self.args.n_xtrain[-1] - 1)
            
            # X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_upper_bound, X_samples.reshape(-1))))), axis=0)
            
            # sampling x and get y boundary:
            X_samples = sampler.random(self.args.n_xtrain[0] - 1) 
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
        Y_train_bound = vmap(u_AllenC, (None, 0, 0))(self.args.a, X_train_bound[:, 0], X_train_bound[:, 1])
           
        return X_train_bound, Y_train_bound

    def test_data_gen(self):
        x1_point_test = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtest[0])
        x2_point_test = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtest[-1])
        X_test = np.array(list(product(x1_point_test, x2_point_test)))
        Y_test = vmap(u_AllenC, (None, 0, 0))(self.args.a, X_test[:, 0], X_test[:, 1])

        return X_test, Y_test

from scipy.stats.qmc import LatinHypercube
from itertools import product
from utilities import *


@jit
def u_elliptic_zhitong(x, y):
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) + 4.0 * jnp.sin(4.0 * jnp.pi * x) * jnp.sin(4.0 * jnp.pi * y)

@jit
def pde_force_gen(x, y):
    uval = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) + 4.0 * jnp.sin(4.0 * jnp.pi * x) * jnp.sin(4.0 * jnp.pi * y)
    dudxx = - (jnp.pi**2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) - 64.0 * (jnp.pi**2) * jnp.sin(4.0 * jnp.pi * x) * jnp.sin(4.0 * jnp.pi * y)
    return uval**3 - 2 * dudxx


class DataGen:

    def __init__(self, args):

        self.args = args
        self.x_range = [float(x) for x in self.args.x_range.split(',')]

        self.x_c, self.y_c = self.collocation_gen()
        
        self.x_b = self.boundary_gen()
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
            x1_point_train = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtrain[0] + 2)[1:-1]
            x2_point_train = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtrain[-1] + 2)[1:-1]
            xx2, xx1 = np.meshgrid(x2_point_train, x1_point_train)
            x1_sampling, x2_sampling = xx1.reshape(-1, 1), xx2.reshape(-1, 1)
            self.args.n_train_collocation = self.args.n_xtrain[0] * self.args.n_xtrain[-1]
        X_train_col = np.concatenate((x1_sampling, x2_sampling), axis=-1)
        F_PDE = vmap(pde_force_gen, (0, 0))(x1_sampling, x2_sampling)
        return X_train_col, F_PDE.reshape(-1)


    def boundary_gen(self):
        X_lower_bound = np.array(self.x_range[0:1])
        X_upper_bound = np.array(self.x_range[1:])
        X_bound = np.array(self.x_range)
        if self.args.random_sampling:
            sampler = LatinHypercube(self.args.n_order-1, optimization='random-cd')
            
            # sampling y and get x boundary:
            X_samples = sampler.random(self.args.n_xtrain[-1] - 1)
            X_train_bound = np.array(list(product(X_lower_bound, X_samples.reshape(-1))))
            X_samples = sampler.random(self.args.n_xtrain[-1] - 1)
            X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_upper_bound, X_samples.reshape(-1))))), axis=0)
            
            # sampling x and get y boundary:
            X_samples = sampler.random(self.args.n_xtrain[0] - 1) 
            X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_samples.reshape(-1), X_lower_bound)))), axis=0)
            X_samples = sampler.random(self.args.n_xtrain[0] - 1) 
            X_train_bound = np.concatenate((X_train_bound, np.array(list(product(X_samples.reshape(-1), X_upper_bound)))), axis=0)
        else:
            x1_point_train = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtrain[0])
            x2_point_train = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtrain[-1])
            
            # get x boundary:
            X_train_bound = np.array(list(product(X_bound, x2_point_train)))
        
            # get y boundary:
            X_train_bound = np.concatenate((X_train_bound, np.array(list(product(x1_point_train[1:-1], X_bound)))), axis=0)
           
        return X_train_bound

    def test_data_gen(self):
        x1_point_test = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtest[0])
        x2_point_test = np.linspace(self.x_range[0], self.x_range[-1], self.args.n_xtest[-1])
        X_test = np.array(list(product(x1_point_test, x2_point_test)))
        Y_test = vmap(u_elliptic_zhitong, (0, 0))(X_test[:, 0], X_test[:, 1])

        return X_test, Y_test

from scipy.stats.qmc import LatinHypercube
from utilities import *

def u_Burgers(args, G, w, t, x):
    temp = x - np.sqrt(4 * args.nu * t) * G
    val1 = (
        w[None, ...]
        * np.sin(np.pi * temp)
        * np.exp(-np.cos(np.pi * temp) / (2 * np.pi * args.nu))
    )
    val2 = w[None, ...] * np.exp(-np.cos(np.pi * temp) / (2 * np.pi * args.nu))
    return -np.sum(val1, axis=-1) / np.sum(val2, axis=-1)

def prepare_coef(args):
    [Gauss_pts, weights] = np.polynomial.hermite.hermgauss(args.n_Gpt)
    return Gauss_pts, weights

def u_Burgers_init(x):
    return -np.sin(np.pi * x)

class DataGen:

    def __init__(self, args):

        self.args = args

        self.x_range = [[float(x) for x in args.x1_range.split(',')], [float(x) for x in args.x2_range.split(',')]]
        self.x_c = self.collocation_gen()

        self.x_b, self.y_b = self.boundary_gen()
        self.args.n_train_boundary = self.x_b.shape[0]
        Gauss_pts, weights = prepare_coef(args)
        self.X_test, self.Y_test = self.test_data_gen(Gauss_pts, weights)

        self.YNorm2_test = np.linalg.norm(self.Y_test.reshape(-1), 2)
        self.YSUM_ABS = np.sum(np.abs(self.Y_test.reshape(-1)))

    def collocation_gen(self):
        if self.args.random_sampling:
            sampler = LatinHypercube(self.args.n_order, optimization='random-cd')
            X_samples = sampler.random(self.args.n_train_collocation)
            x1_sampling = self.x_range[0][0] + X_samples[:, 0:1] * (self.x_range[0][-1] - self.x_range[0][0])
            x2_sampling = self.x_range[-1][0] + X_samples[:, 1:] * (self.x_range[-1][-1] - self.x_range[-1][0])
        else:
            x1_point_train = np.linspace(self.x_range[0][0], self.x_range[0][-1], self.args.n_xtrain[0] + 2)[1:-1]
            x2_point_train = np.linspace(self.x_range[-1][0], self.x_range[-1][-1], self.args.n_xtrain[-1] + 1)[1:]
            xx2, xx1 = np.meshgrid(x2_point_train, x1_point_train)
            x1_sampling, x2_sampling = xx1.reshape(-1, 1), xx2.reshape(-1, 1)
            self.args.n_train_collocation = self.args.n_xtrain[0] * self.args.n_xtrain[-1]
        return np.concatenate((x1_sampling, x2_sampling), axis=-1)


    def boundary_gen(self):
        if self.args.random_sampling:
            sampler = LatinHypercube(self.args.n_order-1, optimization='random-cd')
            X_train_bound = np.empty(shape=(0, self.args.n_order))

            # sampling time and get pos boundary:
            X_samples = sampler.random(self.args.n_xtrain[-1] - 1)
            x_sampling = self.x_range[-1][0] + X_samples * (self.x_range[-1][-1] - self.x_range[-1][0])
            lower_bound = np.concatenate((np.ones((self.args.n_xtrain[-1] - 1, 1)) * self.x_range[0][0], x_sampling), axis=-1)
            upper_bound = np.concatenate((np.ones((self.args.n_xtrain[-1] - 1, 1)) * self.x_range[0][-1], x_sampling), axis=-1)
            X_train_bound = np.concatenate((X_train_bound, lower_bound, upper_bound), axis=0)
            n_total = X_train_bound.shape[0]

            Y_train_bound = np.zeros(n_total, dtype=np.float64)

            # sampling pos and get time boundary-init:
            X_samples = sampler.random(self.args.n_xtrain[0])
            x_sampling = self.x_range[0][0] + X_samples * (self.x_range[0][-1] - self.x_range[0][0])
            lower_bound = np.concatenate((x_sampling, np.ones((self.args.n_xtrain[0], 1)) * self.x_range[-1][0]), axis=-1)
            X_train_bound = np.concatenate((X_train_bound, lower_bound), axis=0)
            Y_train_bound = np.concatenate((Y_train_bound, u_Burgers_init(x_sampling).reshape(-1)), axis=0)
        else:
            x1_point_train = np.linspace(self.x_range[0][0], self.x_range[0][-1], self.args.n_xtrain[0])
            x2_point_train = np.linspace(self.x_range[-1][0], self.x_range[-1][-1], self.args.n_xtrain[-1])

            # init situation when t = 0.0
            X_train_bound = np.concatenate((x1_point_train[1:-1][..., None], np.ones((self.args.n_xtrain[0] - 2, 1)) * self.x_range[-1][0]), axis=-1)
            Y_train_bound = u_Burgers_init(X_train_bound[:, 0])

            # pos boundary situation when x = -1.0, 1.0
            lower_bound = np.concatenate((np.ones((self.args.n_xtrain[-1], 1)) * self.x_range[0][0], x2_point_train[..., None]), axis=-1)
            upper_bound = np.concatenate((np.ones((self.args.n_xtrain[-1], 1)) * self.x_range[0][-1], x2_point_train[..., None]), axis=-1)
            X_train_bound = np.concatenate((X_train_bound, lower_bound, upper_bound), axis=0)
            Y_train_bound = np.concatenate((Y_train_bound, np.zeros(self.args.n_xtrain[-1] * 2)), axis=0)
        return X_train_bound, Y_train_bound

    def test_data_gen(self, Gauss_pts, weights):
        x1_point_test = np.linspace(self.x_range[0][0], self.x_range[0][-1], self.args.n_xtest[0])
        x2_point_test = np.linspace(self.x_range[-1][0], self.x_range[-1][-1], self.args.n_xtest[-1])

        # time init t = 0.0:
        X_test = np.concatenate((x1_point_test[1:-1][..., None], np.ones((self.args.n_xtest[0] - 2, 1)) * self.x_range[-1][0]), axis=-1)
        Y_test = u_Burgers_init(X_test[:, 0])

        # pos boundary: -1.0, 1.0
        lower_bound = np.concatenate((np.ones((self.args.n_xtest[-1], 1)) * self.x_range[0][0], x2_point_test[..., None]), axis=-1)
        upper_bound = np.concatenate((np.ones((self.args.n_xtest[-1], 1)) * self.x_range[0][-1], x2_point_test[..., None]), axis=-1)
        Y_test = np.concatenate((Y_test, np.zeros(self.args.n_xtest[-1] * 2, dtype=np.float64)), axis=0)
        X_test = np.concatenate((X_test, lower_bound, upper_bound), axis=0)

        X_test_remaining = np.empty(shape=(0, self.args.n_order))
        for idx_x1 in range(1, self.args.n_xtest[0]-1):
            X_test_remaining = np.concatenate((X_test_remaining, np.concatenate((np.ones((self.args.n_xtest[-1] - 1, 1)) * x1_point_test[idx_x1], \
            x2_point_test[1:][..., None]), axis=-1)), axis=0)

        Y_test = np.concatenate((Y_test, u_Burgers(self.args, Gauss_pts, weights, X_test_remaining[:, 1:], X_test_remaining[:, 0:1])), axis=0)
        X_test = np.concatenate((X_test, X_test_remaining), axis=0)
        return X_test, Y_test

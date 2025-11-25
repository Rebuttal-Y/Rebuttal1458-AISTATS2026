from scipy.stats import qmc
from utilities import *
import math

class DataGen:

    def __init__(self, args):
        self.args = args
        
        self.x_range = [float(x) for x in self.args.x_range.split(',')]

        self.x_train_collocation, self.F_train_PDE = None, None
        self.x_train_boundary, self.y_train_boundary = None, None
        
        self.X_test, self.Y_test = None, None
        
        self._load_dataset()
        self.YNorm2_test = np.linalg.norm(self.Y_test.reshape(-1), 2)
        self.YSUM_ABS = np.sum(np.abs(self.Y_test.reshape(-1)))

        self._adjust_argument()
        
    def _load_dataset(self):
        dataset = np.load(self.args.dataset_load_path)
        self.args.beta = float(dataset['beta'])
        self.x_train_collocation, self.F_train_PDE = dataset['XTrain_Col'], dataset['FTrain_PDE_Trace'].reshape(-1)
        self.x_train_boundary, self.y_train_boundary = dataset['XTrain_Boundary'], dataset['YTrain_Boundary'].reshape(-1)
        dataset = np.load(self.args.test_dataset_load_path)
        self.X_test, self.Y_test = dataset['XTest'], dataset['YTest']
        
    def _adjust_argument(self):
        self.args.n_order = self.X_test.shape[-1]
        self.args.n_train_collocation = self.x_train_collocation.shape[0]
        self.args.n_train_boundary = self.x_train_boundary.shape[0]
        self.args.n_test = self.X_test.shape[0]


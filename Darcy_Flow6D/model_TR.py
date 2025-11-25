from kernels import *


class GP_ALS_6D(object):
    #X_b, boundary collocation points, Nb x d  
    #y_b: boundary values, Nb
    #X_c: all collocation points, including both boundary and interior points: Nc x d 
    #y_c: the PDE value corresponding to X_c
    #rank: the number of latent functions in each mode 
    def __init__(self, args, data_gen):
        #let us assume the sampling locations are the same across input and output examples
        self.args = args
        self.data_gen = data_gen

        self.X_inducing = [np.linspace(self.data_gen.x_range[0], self.data_gen.x_range[-1], num=self.args.n_xind[i]) \
            for i in range(self.args.n_order)] # inducing points per dim
        self.jitter = [np.exp(-self.args.jitter[i]) for i in range(self.args.n_order)] #jitter for the kernel matrix
        self.ker_func = []
        for i in range(self.args.n_order):
            if self.args.kernel_s[i] == 0:
                self.ker_func.append(Matern12_kernel_1d())
            elif self.args.kernel_s[i] == 1:
                self.ker_func.append(Matern32_kernel_1d())
            elif self.args.kernel_s[i] == 2:
                self.ker_func.append(Matern52_kernel_1d())
            elif self.args.kernel_s == 3:
                self.ker_func.append(Matern72_kernel_1d())
            elif self.args.kernel_s == 4:
                self.ker_func.append(Matern92_kernel_1d())
            else:
                self.ker_func.append(Gaussian_kernel_1d())
        
        self.ker_params = [{'log-ls': args.log_lsx[i]} for i in range(self.args.n_order)] #kernel parameters, log length scale
        
        #need to update alternatingly
        # self.H = [np.random.rand(self.args.n_xind[i], self.args.rank) for i in range(self.args.n_order)] #latent function values, M x R
        self.H = []
        pre_rank = self.args.rank[0]
        for i in range(1, len(self.args.rank)):
            self.H.append(np.random.rand(pre_rank, self.args.n_xind[i-1], self.args.rank[i]))
            pre_rank = self.args.rank[i]
        self.H.append(np.random.rand(pre_rank, self.args.n_xind[-1], self.args.rank[0]))
            
        self.X_b = self.data_gen.x_train_boundary
        self.y_b = self.data_gen.y_train_boundary
        self.X_c = self.data_gen.x_train_collocation
        self.y_c = self.data_gen.F_train_PDE

        self._prepare()

    def _prepare(self):
        #M x M kernel matrix in each dimension
        K = [kernel_matrix(self.X_inducing[j], self.ker_func[j].kappa, self.ker_params[j], self.jitter[j]) for j in range(self.args.n_order)] #
        Kinv =[jnp.linalg.solve(K[j], jnp.eye(K[j].shape[0])) for j in range(self.args.n_order)] #M x M, inverse of the kernel matrix

        #boundary points
        KCrossBoundary = [cross_kernel(self.X_b[:, j], self.X_inducing[j], self.ker_func[j].kappa, self.ker_params[j]) for j in range(self.args.n_order)] #Nb x M
        KInvCrossBoundary = [jnp.linalg.solve(K[j], KCrossBoundary[j].T) for j in range(self.args.n_order)] # M x Nb

        #factor fucntions for u_1
        KCrossCol_U1 = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].D_x1_kappa, self.ker_params[0]), 
                        cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                        cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                        cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                        cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                        cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M
        KInvCrossCol_U1 = [jnp.linalg.solve(K[j], KCrossCol_U1[j].T) for j in range(self.args.n_order)] #M x Nc

        #factor fucntions for u_11
        KCrossCol_U11 = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].DD_x1_kappa, self.ker_params[0]), 
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                         cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                         cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M
        KInvCrossCol_U11 = [jnp.linalg.solve(K[j], KCrossCol_U11[j].T) for j in range(self.args.n_order)] #M x Nc

        #factor fucntions for u_2
        KCrossCol_U2 = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                        cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].D_x1_kappa, self.ker_params[1]),
                        cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                        cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                        cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                        cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M 
        KInvCrossCol_U2 = [jnp.linalg.solve(K[j], KCrossCol_U2[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_22
        KCrossCol_U22 = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].DD_x1_kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                         cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                         cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M 
        KInvCrossCol_U22 = [jnp.linalg.solve(K[j], KCrossCol_U22[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_3
        KCrossCol_U3 = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                        cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                        cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].D_x1_kappa, self.ker_params[2]),
                        cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                        cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                        cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M 
        KInvCrossCol_U3 = [jnp.linalg.solve(K[j], KCrossCol_U3[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_33
        KCrossCol_U33 = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].DD_x1_kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                         cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                         cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M 
        KInvCrossCol_U33 = [jnp.linalg.solve(K[j], KCrossCol_U33[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_4
        KCrossCol_U4 = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                        cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                        cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                        cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].D_x1_kappa, self.ker_params[3]),
                        cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                        cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M 
        KInvCrossCol_U4 = [jnp.linalg.solve(K[j], KCrossCol_U4[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_44
        KCrossCol_U44 = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].DD_x1_kappa, self.ker_params[3]),
                         cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                         cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M 
        KInvCrossCol_U44 = [jnp.linalg.solve(K[j], KCrossCol_U44[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_5
        KCrossCol_U5 = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                        cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                        cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                        cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                        cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].D_x1_kappa, self.ker_params[4]),
                        cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M 
        KInvCrossCol_U5 = [jnp.linalg.solve(K[j], KCrossCol_U5[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_55
        KCrossCol_U55 = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                         cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].DD_x1_kappa, self.ker_params[4]),
                         cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M 
        KInvCrossCol_U55 = [jnp.linalg.solve(K[j], KCrossCol_U55[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_6
        KCrossCol_U6 = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                        cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                        cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                        cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                        cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                        cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].D_x1_kappa, self.ker_params[5])] #Nc x M 
        KInvCrossCol_U6 = [jnp.linalg.solve(K[j], KCrossCol_U6[j].T) for j in range(self.args.n_order)] 
        
        #factor fucntions for u_66
        KCrossCol_U66 = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                         cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                         cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].DD_x1_kappa, self.ker_params[5])] #Nc x M 
        KInvCrossCol_U66 = [jnp.linalg.solve(K[j], KCrossCol_U66[j].T) for j in range(self.args.n_order)] 

        #u
        KcrossCol_U = [cross_kernel(self.X_c[:, j], self.X_inducing[j], self.ker_func[j].kappa, self.ker_params[j]) for j in range(self.args.n_order)] #Nc x M
        KInvCrossCol_U = [jnp.linalg.solve(K[j], KcrossCol_U[j].T) for j in range(self.args.n_order)]


        #M x Nc
        self.K = K
        self.Kinv = Kinv
        self.KInvCrossBoundary = KInvCrossBoundary
        self.KInvCrossCol_U1 = KInvCrossCol_U1
        self.KInvCrossCol_U2 = KInvCrossCol_U2
        self.KInvCrossCol_U3 = KInvCrossCol_U3
        self.KInvCrossCol_U4 = KInvCrossCol_U4
        self.KInvCrossCol_U5 = KInvCrossCol_U5
        self.KInvCrossCol_U6 = KInvCrossCol_U6
        self.KInvCrossCol_U11 = KInvCrossCol_U11
        self.KInvCrossCol_U22 = KInvCrossCol_U22
        self.KInvCrossCol_U33 = KInvCrossCol_U33
        self.KInvCrossCol_U44 = KInvCrossCol_U44
        self.KInvCrossCol_U55 = KInvCrossCol_U55
        self.KInvCrossCol_U66 = KInvCrossCol_U66
        self.KInvCrossCol_U = KInvCrossCol_U
        self.A = jnp.array(A_coef(self.X_c))
        self.A_1 = jnp.array(dAxi(0, self.X_c))
        self.A_2 = jnp.array(dAxi(1, self.X_c))
        self.A_3 = jnp.array(dAxi(2, self.X_c))
        self.A_4 = jnp.array(dAxi(3, self.X_c))
        self.A_5 = jnp.array(dAxi(4, self.X_c))
        self.A_6 = jnp.array(dAxi(5, self.X_c))


    def update(self):
        n_col = self.X_c.shape[0]
        n_bound = self.X_b.shape[0]

        for j in range(self.args.n_order):

            left_bound = np.tile(np.eye(self.args.rank[0])[None, ...], (n_bound, 1, 1))
            left_u = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_u1 = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_u2 = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_u3 = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_u4 = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_u5 = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_u6 = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_u11 = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_u22 = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_u33 = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_u44 = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_u55 = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_u66 = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            for i in range(j):
                left_u = jnp.einsum("bac,cbd->bad", left_u, self.KInvCrossCol_U[i].T @ self.H[i])
                left_u1 = jnp.einsum("bac,cbd->bad", left_u1, self.KInvCrossCol_U1[i].T @ self.H[i])
                left_u2 = jnp.einsum("bac,cbd->bad", left_u2, self.KInvCrossCol_U2[i].T @ self.H[i])
                left_u3 = jnp.einsum("bac,cbd->bad", left_u3, self.KInvCrossCol_U3[i].T @ self.H[i])
                left_u4 = jnp.einsum("bac,cbd->bad", left_u4, self.KInvCrossCol_U4[i].T @ self.H[i])
                left_u5 = jnp.einsum("bac,cbd->bad", left_u5, self.KInvCrossCol_U5[i].T @ self.H[i])
                left_u6 = jnp.einsum("bac,cbd->bad", left_u6, self.KInvCrossCol_U6[i].T @ self.H[i])
                left_u11 = jnp.einsum("bac,cbd->bad", left_u11, self.KInvCrossCol_U11[i].T @ self.H[i])
                left_u22 = jnp.einsum("bac,cbd->bad", left_u22, self.KInvCrossCol_U22[i].T @ self.H[i])
                left_u33 = jnp.einsum("bac,cbd->bad", left_u33, self.KInvCrossCol_U33[i].T @ self.H[i])
                left_u44 = jnp.einsum("bac,cbd->bad", left_u44, self.KInvCrossCol_U44[i].T @ self.H[i])
                left_u55 = jnp.einsum("bac,cbd->bad", left_u55, self.KInvCrossCol_U55[i].T @ self.H[i])
                left_u66 = jnp.einsum("bac,cbd->bad", left_u66, self.KInvCrossCol_U66[i].T @ self.H[i])

                left_bound = jnp.einsum("bac,cbd->bad", left_bound, self.KInvCrossBoundary[i].T @ self.H[i])

            right_bound = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_bound))
            right_u = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_u1 = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_u2 = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_u3 = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_u4 = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_u5 = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_u6 = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_u11 = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_u22 = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_u33 = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_u44 = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_u55 = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_u66 = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            for k in reversed(range(j+1, self.args.n_order)):
                right_u = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U[k].T @ self.H[k], right_u)
                right_u1 = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U1[k].T @ self.H[k], right_u1)
                right_u2 = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U2[k].T @ self.H[k], right_u2)
                right_u3 = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U3[k].T @ self.H[k], right_u3)
                right_u4 = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U4[k].T @ self.H[k], right_u4)
                right_u5 = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U5[k].T @ self.H[k], right_u5)
                right_u6 = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U6[k].T @ self.H[k], right_u6)
                right_u11 = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U11[k].T @ self.H[k], right_u11)
                right_u22 = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U22[k].T @ self.H[k], right_u22)
                right_u33 = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U33[k].T @ self.H[k], right_u33)
                right_u44 = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U44[k].T @ self.H[k], right_u44)
                right_u55 = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U55[k].T @ self.H[k], right_u55)
                right_u66 = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U66[k].T @ self.H[k], right_u66)

                right_bound = jnp.einsum("abc,cdb->adb", self.KInvCrossBoundary[k].T @ self.H[k], right_bound)

            #u
            A0 = self.KInvCrossCol_U[j].T
            B0 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u, left_u), (0, 2, 1)).reshape(n_col, -1)
            coef = jnp.einsum("bac,cbd,deb->bae", left_u, A0 @ self.H[j], right_u)
            coef0 = jnp.einsum("bii->b", coef)
            coef1 = coef0*coef0
            coef2 = coef0*coef0*coef0
            #u1
            A1 = self.KInvCrossCol_U1[j].T
            B1 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u1, left_u1), (0, 2, 1)).reshape(n_col, -1)
            #u11
            A11 = self.KInvCrossCol_U11[j].T
            B11 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u11, left_u11), (0, 2, 1)).reshape(n_col, -1)
            #u2
            A2 = self.KInvCrossCol_U2[j].T
            B2 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u2, left_u2), (0, 2, 1)).reshape(n_col, -1)
            #u22
            A22 = self.KInvCrossCol_U22[j].T
            B22 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u22, left_u22), (0, 2, 1)).reshape(n_col, -1)
            #u3
            A3 = self.KInvCrossCol_U3[j].T
            B3 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u3, left_u3), (0, 2, 1)).reshape(n_col, -1)
            #u33
            A33 = self.KInvCrossCol_U33[j].T
            B33 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u33, left_u33), (0, 2, 1)).reshape(n_col, -1)
            #u4
            A4 = self.KInvCrossCol_U4[j].T
            B4 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u4, left_u4), (0, 2, 1)).reshape(n_col, -1)
            #u44
            A44 = self.KInvCrossCol_U44[j].T
            B44 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u44, left_u44), (0, 2, 1)).reshape(n_col, -1)
            #u5
            A5 = self.KInvCrossCol_U5[j].T
            B5 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u5, left_u5), (0, 2, 1)).reshape(n_col, -1)
            #u55
            A55 = self.KInvCrossCol_U55[j].T
            B55 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u55, left_u55), (0, 2, 1)).reshape(n_col, -1)
            #u6
            A6 = self.KInvCrossCol_U6[j].T
            B6 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u6, left_u6), (0, 2, 1)).reshape(n_col, -1)
            #u66
            A66 = self.KInvCrossCol_U66[j].T
            B66 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u66, left_u66), (0, 2, 1)).reshape(n_col, -1)

            Phi2 = jnp.zeros((0, A0.shape[-1], B0.shape[-1]))
            idx_start = 0

            while idx_start < n_col:
                idx_end = min(idx_start + self.args.n_train_batch, n_col)
                term1 = self.A[idx_start:idx_end, None] * (jnp.einsum('ij,ik->ijk', A11[idx_start:idx_end, ...], B11[idx_start:idx_end, ...]) + \
                                                           jnp.einsum('ij,ik->ijk', A22[idx_start:idx_end, ...], B22[idx_start:idx_end, ...]) + \
                                                           jnp.einsum('ij,ik->ijk', A33[idx_start:idx_end, ...], B33[idx_start:idx_end, ...]) + \
                                                           jnp.einsum('ij,ik->ijk', A44[idx_start:idx_end, ...], B44[idx_start:idx_end, ...]) + \
                                                           jnp.einsum('ij,ik->ijk', A55[idx_start:idx_end, ...], B55[idx_start:idx_end, ...]) + \
                                                           jnp.einsum('ij,ik->ijk', A66[idx_start:idx_end, ...], B66[idx_start:idx_end, ...]))
                term2 = self.A_1[idx_start:idx_end, None] * jnp.einsum('ij,ik->ijk', A1[idx_start:idx_end, ...], B1[idx_start:idx_end, ...]) + \
                        self.A_2[idx_start:idx_end, None] * jnp.einsum('ij,ik->ijk', A2[idx_start:idx_end, ...], B2[idx_start:idx_end, ...]) + \
                        self.A_3[idx_start:idx_end, None] * jnp.einsum('ij,ik->ijk', A3[idx_start:idx_end, ...], B3[idx_start:idx_end, ...]) + \
                        self.A_4[idx_start:idx_end, None] * jnp.einsum('ij,ik->ijk', A4[idx_start:idx_end, ...], B4[idx_start:idx_end, ...]) + \
                        self.A_5[idx_start:idx_end, None] * jnp.einsum('ij,ik->ijk', A5[idx_start:idx_end, ...], B5[idx_start:idx_end, ...]) + \
                        self.A_6[idx_start:idx_end, None] * jnp.einsum('ij,ik->ijk', A6[idx_start:idx_end, ...], B6[idx_start:idx_end, ...])
                if self.args.Newton_M:
                    Phi2 = jnp.concatenate((Phi2, - term1 - term2 + 3.0*coef1[idx_start:idx_end, None, None]*jnp.einsum('ij,ik->ijk', A0[idx_start:idx_end, ...], B0[idx_start:idx_end, ...])), axis=0)
                else:
                    Phi2 = Phi2 = jnp.concatenate((Phi2, - term1 - term2 + jnp.einsum('ij,ik->ijk', A0[idx_start:idx_end, ...], B0[idx_start:idx_end, ...])*coef1[idx_start:idx_end, None, None]), axis=0)
                idx_start = idx_end
            if self.args.CPU_PDE:
                Phi2 = jax.device_put(Phi2.reshape(n_col, -1), device=jax.devices("cpu")[0])
                Phi = self.args.lam2/self.args.lam1*jax.device_put(Phi2.T@Phi2, device=jax.devices("gpu")[0])
            else:
                Phi2 = Phi2.reshape(n_col, -1)
                Phi = self.args.lam2/self.args.lam1*(Phi2.T@Phi2)

            #boundary condition
            Ab = self.KInvCrossBoundary[j].T
            Bb = jnp.transpose(jnp.einsum("acb,bcd->bad", right_bound, left_bound), (0, 2, 1)).reshape(n_bound, -1)
            
            Phi1 = jnp.einsum('ij,ik->ijk', Ab, Bb)
            Phi1 = Phi1.reshape(n_bound, -1)
            Phi += Phi1.T@Phi1

            rank_left, n_inducing, rank_right = self.H[j].shape
            Phi += 1/self.args.lam1*jnp.kron(self.Kinv[j], jnp.eye(rank_left * rank_right))
            if self.args.Newton_M:
                eta = Phi1.T@self.y_b + self.args.lam2/self.args.lam1*Phi2.T@(self.y_c + 2.0*coef2)
            else:
                eta = Phi1.T@self.y_b + self.args.lam2/self.args.lam1*Phi2.T@self.y_c

            HVec = jnp.linalg.solve(Phi, eta)
            self.H[j] = jnp.transpose(HVec.reshape(n_inducing, rank_left, rank_right), (1, 0, 2)) #update

    def pred(self, Xte):
        KCrossTest = [cross_kernel(Xte[:, j], self.X_inducing[j], self.ker_func[j].kappa, self.ker_params[j]) for j in range(self.args.n_order)]
        KInvInducing = [jnp.linalg.solve(self.K[j], self.H[j]) for j in range(self.args.n_order)]
        FuncVals = [KCrossTest[j] @ KInvInducing[j] for j in range(self.args.n_order)]

        pred = FuncVals[0]
        for j in range(1, self.args.n_order):
            pred = jnp.einsum("abc,cbd->abd", pred, FuncVals[j])
        # pred = pred.reshape(Xte.shape[0], -1)
        pred = jnp.einsum("ibi->b", pred)
        return pred

    def solution_approx(self, x1, x2, x3, x4, x5, x6):
        KCross = [cross_kernel(x1, self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]), 
                  cross_kernel(x2, self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]), 
                  cross_kernel(x3, self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]), 
                  cross_kernel(x4, self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                  cross_kernel(x5, self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                  cross_kernel(x6, self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #N x M
        KInvInducing = [jnp.linalg.solve(self.K[j], self.H[j]) for j in range(self.args.n_order)] #M x R
        res = KCross[0] @ KInvInducing[0]
        for i in range(1, self.args.n_order):
            res = jnp.einsum("abc,cbd->abd", res, KCross[i] @ KInvInducing[i])
        # res = jnp.sum(res)
        res = jnp.trace(res[:, 0, :])
        return res 

    def get_loss(self):
        boundary_pred = vmap(self.solution_approx, (0, 0, 0, 0, 0, 0))(self.X_b[:, 0], self.X_b[:, 1], self.X_b[:, 2], self.X_b[:, 3], self.X_b[:, 4], self.X_b[:, 5])
        boundary_loss = jnp.square(boundary_pred - self.y_b).mean()
        u1 = grad(self.solution_approx, 0)
        u2 = grad(self.solution_approx, 1)
        u3 = grad(self.solution_approx, 2)
        u4 = grad(self.solution_approx, 3)
        u5 = grad(self.solution_approx, 4)
        u6 = grad(self.solution_approx, 5)
        u11 = grad(grad(self.solution_approx, 0), 0)
        u22 = grad(grad(self.solution_approx, 1), 1)
        u33 = grad(grad(self.solution_approx, 2), 2)
        u44 = grad(grad(self.solution_approx, 3), 3)
        u55 = grad(grad(self.solution_approx, 4), 4)
        u66 = grad(grad(self.solution_approx, 5), 5)
        u_pred = vmap(self.solution_approx, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        u1_pred = vmap(u1, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        u2_pred = vmap(u2, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        u3_pred = vmap(u3, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        u4_pred = vmap(u4, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        u5_pred = vmap(u5, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        u6_pred = vmap(u6, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        u11_pred = vmap(u11, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        u22_pred = vmap(u22, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        u33_pred = vmap(u33, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        u44_pred = vmap(u44, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        u55_pred = vmap(u55, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        u66_pred = vmap(u66, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        term1 = (u11_pred + u22_pred + u33_pred + u44_pred + u55_pred + u66_pred) * self.A.reshape(-1)
        term2 = self.A_1.reshape(-1) * u1_pred + self.A_2.reshape(-1) * u2_pred + self.A_3.reshape(-1) * u3_pred + self.A_4.reshape(-1) * u4_pred + \
            self.A_5.reshape(-1) * u5_pred + self.A_6.reshape(-1) * u6_pred
        g_pred = u_pred*u_pred*u_pred - term1 - term2
        residual = jnp.square(g_pred - self.y_c).mean() #collocation loss
        loss = boundary_loss + residual #total loss
        return loss, boundary_loss, residual

    def get_residual(self):
        
        U1 = self.KInvCrossCol_U1[0].T@self.H[0]
        U2 = self.KInvCrossCol_U2[0].T@self.H[0]
        U3 = self.KInvCrossCol_U3[0].T@self.H[0]
        U4 = self.KInvCrossCol_U4[0].T@self.H[0]
        U5 = self.KInvCrossCol_U5[0].T@self.H[0]
        U6 = self.KInvCrossCol_U6[0].T@self.H[0]
        U11 = self.KInvCrossCol_U11[0].T@self.H[0]
        U22 = self.KInvCrossCol_U22[0].T@self.H[0]
        U33 = self.KInvCrossCol_U33[0].T@self.H[0]
        U44 = self.KInvCrossCol_U44[0].T@self.H[0]
        U55 = self.KInvCrossCol_U55[0].T@self.H[0]
        U66 = self.KInvCrossCol_U66[0].T@self.H[0]
        U = self.KInvCrossCol_U[0].T@self.H[0]
        for i in range(1, self.args.n_order):
            U1 = jnp.einsum("abc,cbd->abd", U1, self.KInvCrossCol_U1[i].T@self.H[i])
            U2 = jnp.einsum("abc,cbd->abd", U2, self.KInvCrossCol_U2[i].T@self.H[i])
            U3 = jnp.einsum("abc,cbd->abd", U3, self.KInvCrossCol_U3[i].T@self.H[i])
            U4 = jnp.einsum("abc,cbd->abd", U4, self.KInvCrossCol_U4[i].T@self.H[i])
            U5 = jnp.einsum("abc,cbd->abd", U5, self.KInvCrossCol_U5[i].T@self.H[i])
            U6 = jnp.einsum("abc,cbd->abd", U6, self.KInvCrossCol_U6[i].T@self.H[i])
            U11 = jnp.einsum("abc,cbd->abd", U11, self.KInvCrossCol_U11[i].T@self.H[i])
            U22 = jnp.einsum("abc,cbd->abd", U22, self.KInvCrossCol_U22[i].T@self.H[i])
            U33 = jnp.einsum("abc,cbd->abd", U33, self.KInvCrossCol_U33[i].T@self.H[i])
            U44 = jnp.einsum("abc,cbd->abd", U44, self.KInvCrossCol_U44[i].T@self.H[i])
            U55 = jnp.einsum("abc,cbd->abd", U55, self.KInvCrossCol_U55[i].T@self.H[i])
            U66 = jnp.einsum("abc,cbd->abd", U66, self.KInvCrossCol_U66[i].T@self.H[i])
            U = jnp.einsum("abc,cbd->abd", U, self.KInvCrossCol_U[i].T@self.H[i])


        U1 = jnp.einsum("ibi->b", U1)
        U2 = jnp.einsum("ibi->b", U2)
        U3 = jnp.einsum("ibi->b", U3)
        U4 = jnp.einsum("ibi->b", U4)
        U5 = jnp.einsum("ibi->b", U5)
        U6 = jnp.einsum("ibi->b", U6)
        U11 = jnp.einsum("ibi->b", U11)
        U22 = jnp.einsum("ibi->b", U22)
        U33 = jnp.einsum("ibi->b", U33)
        U44 = jnp.einsum("ibi->b", U44)
        U55 = jnp.einsum("ibi->b", U55)
        U66 = jnp.einsum("ibi->b", U66)
        U = jnp.einsum("ibi->b", U)

        term1 = (U11 + U22 + U33 + U44 + U55 + U66) * self.A.reshape(-1)
        term2 = self.A_1.reshape(-1) * U1 + self.A_2.reshape(-1) * U2 + self.A_3.reshape(-1) * U3 + self.A_4.reshape(-1) * U4 + \
            self.A_5.reshape(-1) * U5 + self.A_6.reshape(-1) * U6

        g_pred = U*U*U - term1 - term2
        residual = jnp.square(g_pred.reshape(-1) - self.y_c).mean()
        return residual 
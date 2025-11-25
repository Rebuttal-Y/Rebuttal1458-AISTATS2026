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
        self.H = [np.random.rand(self.args.n_xind[i], self.args.rank) for i in range(self.args.n_order)] #latent function values, M x R
        
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

        for j in range(self.args.n_order):
            #u
            A0 = self.KInvCrossCol_U[j].T #Nc x M
            B0 = math.prod([self.KInvCrossCol_U[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            coef0 = jnp.sum((A0 @ self.H[j]) * B0, 1) #Nc
            coef1 = coef0*coef0
            coef2 = coef0*coef0*coef0
            #u_1
            A1 = self.KInvCrossCol_U1[j].T #Nc x M
            B1 = math.prod([self.KInvCrossCol_U1[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_11
            A11 = self.KInvCrossCol_U11[j].T #Nc x M
            B11 = math.prod([self.KInvCrossCol_U11[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_2
            A2 = self.KInvCrossCol_U2[j].T #Nc x M
            B2 = math.prod([self.KInvCrossCol_U2[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_22
            A22 = self.KInvCrossCol_U22[j].T #Nc x M
            B22 = math.prod([self.KInvCrossCol_U22[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_3
            A3 = self.KInvCrossCol_U3[j].T #Nc x M
            B3 = math.prod([self.KInvCrossCol_U3[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_33
            A33 = self.KInvCrossCol_U33[j].T #Nc x M
            B33 = math.prod([self.KInvCrossCol_U33[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_4
            A4 = self.KInvCrossCol_U4[j].T #Nc x M
            B4 = math.prod([self.KInvCrossCol_U4[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_44
            A44 = self.KInvCrossCol_U44[j].T #Nc x M
            B44 = math.prod([self.KInvCrossCol_U44[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_5
            A5 = self.KInvCrossCol_U5[j].T #Nc x M
            B5 = math.prod([self.KInvCrossCol_U5[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_55
            A55 = self.KInvCrossCol_U55[j].T #Nc x M
            B55 = math.prod([self.KInvCrossCol_U55[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R 
            #u_6
            A6 = self.KInvCrossCol_U6[j].T #Nc x M
            B6 = math.prod([self.KInvCrossCol_U6[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_66
            A66 = self.KInvCrossCol_U66[j].T #Nc x M
            B66 = math.prod([self.KInvCrossCol_U66[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R 

            Phi2 = jnp.zeros((0, A0.shape[-1], B0.shape[-1]))
            idx_start = 0
            n_total = A0.shape[0]
            while idx_start < n_total:
                idx_end = min(idx_start + self.args.n_train_batch, n_total)
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
                    Phi2 = jnp.concatenate((Phi2, - term1 - term2 + jnp.einsum('ij,ik->ijk', A0[idx_start:idx_end, ...], B0[idx_start:idx_end, ...])*coef1[idx_start:idx_end, None, None]), axis=0)
                idx_start = idx_end
            if self.args.CPU_PDE:
                Phi2 = jax.device_put(Phi2.reshape(n_total, -1), device=jax.devices("cpu")[0]) #Nc x (M*R)
                Phi = self.args.lam2/self.args.lam1*jax.device_put(Phi2.T@Phi2, device=jax.devices("gpu")[0])
            else:
                Phi2 = Phi2.reshape(n_total, -1) #Nc x (M*R)
                Phi = self.args.lam2/self.args.lam1*(Phi2.T@Phi2)
                
            #boundary condition
            Ab = self.KInvCrossBoundary[j].T #Nb x M
            Bb = math.prod([self.KInvCrossBoundary[i].T@self.H[i] for i in range(self.args.n_order) if i != j]) #Nb x R

            Phi1 = jnp.einsum('ij,ik->ijk', Ab, Bb) #Nb x M x R
            Phi1 = Phi1.reshape(Phi1.shape[0], -1) #Nb x (M*R)
            Phi += Phi1.T@Phi1

            Phi += 1/self.args.lam1*jnp.kron(self.Kinv[j], jnp.eye(self.args.rank))
            if self.args.Newton_M:
                eta = Phi1.T@self.y_b + self.args.lam2/self.args.lam1*Phi2.T@(self.y_c + 2.0*coef2) # (M*R) 
            else:
                eta = Phi1.T@self.y_b + self.args.lam2/self.args.lam1*Phi2.T@self.y_c # (M*R) 
            
            HVec = jnp.linalg.solve(Phi, eta) #MR, 
            self.H[j] = HVec.reshape(-1, self.args.rank) #update   


    def pred(self, Xte):
        KCrossTest = [cross_kernel(Xte[:, j], self.X_inducing[j], self.ker_func[j].kappa, self.ker_params[j]) for j in range(self.args.n_order)]
        KInvInducing = [jnp.linalg.solve(self.K[j], self.H[j]) for j in range(self.args.n_order)] #M x R
        FuncVals = [KCrossTest[j] @ KInvInducing[j] for j in range(self.args.n_order)] #N_test x R

        pred = FuncVals[0]
        for j in range(1, self.args.n_order):
            pred = pred * FuncVals[j]
        pred = jnp.sum(pred, axis=1) #N_test
        pred = pred.reshape(Xte.shape[0], -1)
        return pred

    def solution_approx(self, x1, x2, x3, x4, x5, x6):
        KCross = [cross_kernel(x1, self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]), 
                  cross_kernel(x2, self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]), 
                  cross_kernel(x3, self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]), 
                  cross_kernel(x4, self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                  cross_kernel(x5, self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                  cross_kernel(x6, self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #N x M
        KInvInducing = [jnp.linalg.solve(self.K[j], self.H[j]) for j in range(self.args.n_order)] #M x R
        F = math.prod([KCross[i] @ KInvInducing[i] for i in range(self.args.n_order)]) #N x R
        res = jnp.sum(F) #N
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

        U1 = math.prod([self.KInvCrossCol_U1[i].T@self.H[i] for i in range(self.args.n_order)])
        U1 = jnp.sum(U1, 1)
        U2 = math.prod([self.KInvCrossCol_U2[i].T@self.H[i] for i in range(self.args.n_order)])
        U2 = jnp.sum(U2, 1)
        U3 = math.prod([self.KInvCrossCol_U3[i].T@self.H[i] for i in range(self.args.n_order)])
        U3 = jnp.sum(U3, 1)
        U4 = math.prod([self.KInvCrossCol_U4[i].T@self.H[i] for i in range(self.args.n_order)])
        U4 = jnp.sum(U4, 1)
        U5 = math.prod([self.KInvCrossCol_U5[i].T@self.H[i] for i in range(self.args.n_order)])
        U5 = jnp.sum(U5, 1)
        U6 = math.prod([self.KInvCrossCol_U6[i].T@self.H[i] for i in range(self.args.n_order)])
        U6 = jnp.sum(U6, 1)

        U11 = math.prod([self.KInvCrossCol_U11[i].T@self.H[i] for i in range(self.args.n_order)])
        U11 = jnp.sum(U11, 1)
        U22 = math.prod([self.KInvCrossCol_U22[i].T@self.H[i] for i in range(self.args.n_order)])
        U22 = jnp.sum(U22, 1)
        U33 = math.prod([self.KInvCrossCol_U33[i].T@self.H[i] for i in range(self.args.n_order)])
        U33 = jnp.sum(U33, 1)
        U44 = math.prod([self.KInvCrossCol_U44[i].T@self.H[i] for i in range(self.args.n_order)])
        U44 = jnp.sum(U44, 1)
        U55 = math.prod([self.KInvCrossCol_U55[i].T@self.H[i] for i in range(self.args.n_order)])
        U55 = jnp.sum(U55, 1)
        U66 = math.prod([self.KInvCrossCol_U66[i].T@self.H[i] for i in range(self.args.n_order)])
        U66 = jnp.sum(U66, 1)
        U = math.prod([self.KInvCrossCol_U[i].T@self.H[i] for i in range(self.args.n_order)]) #Nc x R
        U = jnp.sum(U, 1) #Nc

        term1 = (U11 + U22 + U33 + U44 + U55 + U66) * self.A.reshape(-1)
        term2 = self.A_1.reshape(-1) * U1 + self.A_2.reshape(-1) * U2 + self.A_3.reshape(-1) * U3 + self.A_4.reshape(-1) * U4 + \
            self.A_5.reshape(-1) * U5 + self.A_6.reshape(-1) * U6

        g_pred = U*U*U - term1 - term2
        residual = jnp.square(g_pred - self.y_c).mean()
        return residual 
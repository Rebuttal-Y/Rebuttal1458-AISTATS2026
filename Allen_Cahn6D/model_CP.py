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

        #factor fucntions for u_xx
        KCrossCol_Uxx = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].DD_x1_kappa, self.ker_params[0]), 
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                         cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                         cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M
        KInvCrossCol_Uxx = [jnp.linalg.solve(K[j], KCrossCol_Uxx[j].T) for j in range(self.args.n_order)] #M x Nc

        #factor fucntions for u_yy
        KCrossCol_Uyy = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].DD_x1_kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                         cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                         cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M
        KInvCrossCol_Uyy = [jnp.linalg.solve(K[j], KCrossCol_Uyy[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_zz
        KCrossCol_Uzz = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].DD_x1_kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                         cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                         cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M
        KInvCrossCol_Uzz = [jnp.linalg.solve(K[j], KCrossCol_Uzz[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_ww
        KCrossCol_Uww = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].DD_x1_kappa, self.ker_params[3]),
                         cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                         cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M
        KInvCrossCol_Uww = [jnp.linalg.solve(K[j], KCrossCol_Uww[j].T) for j in range(self.args.n_order)]

        #factor fucntions for u_vv
        KCrossCol_Uvv = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                         cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].DD_x1_kappa, self.ker_params[4]),
                         cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].kappa, self.ker_params[5])] #Nc x M
        KInvCrossCol_Uvv = [jnp.linalg.solve(K[j], KCrossCol_Uvv[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_ss
        KCrossCol_Uss = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3]),
                         cross_kernel(self.X_c[:, 4], self.X_inducing[4], self.ker_func[4].kappa, self.ker_params[4]),
                         cross_kernel(self.X_c[:, 5], self.X_inducing[5], self.ker_func[5].DD_x1_kappa, self.ker_params[5])] #Nc x M
        KInvCrossCol_Uss = [jnp.linalg.solve(K[j], KCrossCol_Uss[j].T) for j in range(self.args.n_order)] 
        
        #u
        KcrossCol_U = [cross_kernel(self.X_c[:, j], self.X_inducing[j], self.ker_func[j].kappa, self.ker_params[j]) for j in range(self.args.n_order)] #Nc x M
        KInvCrossCol_U = [jnp.linalg.solve(K[j], KcrossCol_U[j].T) for j in range(self.args.n_order)]
        #M x Nc
        self.K = K
        self.Kinv = Kinv
        self.KInvCrossBoundary = KInvCrossBoundary
        self.KInvCrossCol_Uxx = KInvCrossCol_Uxx
        self.KInvCrossCol_Uyy = KInvCrossCol_Uyy
        self.KInvCrossCol_Uzz = KInvCrossCol_Uzz
        self.KInvCrossCol_Uww = KInvCrossCol_Uww
        self.KInvCrossCol_Uvv = KInvCrossCol_Uvv
        self.KInvCrossCol_Uss = KInvCrossCol_Uss
        self.KInvCrossCol_U = KInvCrossCol_U


    def update(self):

        for j in range(self.args.n_order):
            #u
            A0 = self.KInvCrossCol_U[j].T #Nc x M
            B0 = math.prod([self.KInvCrossCol_U[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            coef0 = jnp.sum((A0 @ self.H[j]) * B0, 1) #Nc
            coef1 = coef0*coef0
            coef2 = coef0*coef0*coef0
            #u_xx
            A1 = self.KInvCrossCol_Uxx[j].T #Nc x M
            B1 = math.prod([self.KInvCrossCol_Uxx[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_yy   
            A2 = self.KInvCrossCol_Uyy[j].T #Nc x M
            B2 = math.prod([self.KInvCrossCol_Uyy[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_zz
            A3 = self.KInvCrossCol_Uzz[j].T #Nc x M
            B3 = math.prod([self.KInvCrossCol_Uzz[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_ww
            A4 = self.KInvCrossCol_Uww[j].T #Nc x M
            B4 = math.prod([self.KInvCrossCol_Uww[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_vv
            A5 = self.KInvCrossCol_Uvv[j].T #Nc x M
            B5 = math.prod([self.KInvCrossCol_Uvv[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R
            #u_ss
            A6 = self.KInvCrossCol_Uss[j].T #Nc x M
            B6 = math.prod([self.KInvCrossCol_Uss[i].T @ self.H[i] for i in range(self.args.n_order) if i != j]) #Nc x R

            Phi2 = jnp.zeros((0, A0.shape[-1], B0.shape[-1]))
            idx_start = 0
            n_total = A0.shape[0]
            while idx_start < n_total:
                idx_end = min(idx_start + self.args.n_train_batch, n_total)
                if self.args.Newton_M:
                    Phi2 = jnp.concatenate((Phi2, jnp.einsum('ij,ik->ijk', A1[idx_start:idx_end, ...], B1[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A2[idx_start:idx_end, ...], B2[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A3[idx_start:idx_end, ...], B3[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A4[idx_start:idx_end, ...], B4[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A5[idx_start:idx_end, ...], B5[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A6[idx_start:idx_end, ...], B6[idx_start:idx_end, ...]) + \
                        (3.0*coef1[idx_start:idx_end, None, None] - 1.0) * jnp.einsum('ij,ik->ijk', A0[idx_start:idx_end, ...], B0[idx_start:idx_end, ...])), axis=0)
                else:
                    Phi2 = jnp.concatenate((Phi2, jnp.einsum('ij,ik->ijk', A1[idx_start:idx_end, ...], B1[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A2[idx_start:idx_end, ...], B2[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A3[idx_start:idx_end, ...], B3[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A4[idx_start:idx_end, ...], B4[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A5[idx_start:idx_end, ...], B5[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A6[idx_start:idx_end, ...], B6[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A0[idx_start:idx_end, ...], B0[idx_start:idx_end, ...])*(coef1[idx_start:idx_end, None, None] - 1.0)), axis=0)
                idx_start = idx_end
            if self.args.CPU_PDE:
                Phi2 = jax.device_put(Phi2.reshape(n_total, -1), device=jax.devices("cpu")[0]) #Nc x (M*R)
                Phi = self.args.lam2/self.args.lam1*jax.device_put(Phi2.T@Phi2, device=jax.devices("gpu")[0])
            else:
                Phi2 = Phi2.reshape(n_total, -1) #Nc x (M*R)
                Phi = self.args.lam2/self.args.lam1*(Phi2.T@Phi2)
                
            #boundary condition
            A = self.KInvCrossBoundary[j].T #Nb x M
            B = math.prod([self.KInvCrossBoundary[i].T@self.H[i] for i in range(self.args.n_order) if i != j]) #Nb x R

            Phi1 = jnp.einsum('ij,ik->ijk', A, B) #Nb x M x R
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
        boundary_pred = vmap(self.solution_approx, (0, 0, 0, 0, 0, 0))(self.X_b[:, 0], self.X_b[:, 1], self.X_b[:, 2], self.X_b[:, 3], \
        self.X_b[:, 4], self.X_b[:, 5])
        boundary_loss = jnp.square(boundary_pred - self.y_b).mean()
        uxx = grad(grad(self.solution_approx, 0), 0)
        uyy = grad(grad(self.solution_approx, 1), 1)
        uzz = grad(grad(self.solution_approx, 2), 2)
        uww = grad(grad(self.solution_approx, 3), 3)
        uvv = grad(grad(self.solution_approx, 4), 4)
        uss = grad(grad(self.solution_approx, 5), 5)
        u_pred = vmap(self.solution_approx, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        uxx_pred = vmap(uxx, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        uyy_pred = vmap(uyy, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        uzz_pred = vmap(uzz, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        uww_pred = vmap(uww, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        uvv_pred = vmap(uvv, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        uss_pred = vmap(uss, (0, 0, 0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3], self.X_c[:, 4], self.X_c[:, 5]) #N_c
        
        g_pred = uxx_pred + uyy_pred + uzz_pred + uww_pred + uvv_pred + uss_pred + u_pred*u_pred*u_pred - u_pred #N_c
        residual = jnp.square(g_pred - self.y_c).mean() #collocation loss
        loss = boundary_loss + residual #total loss
        return loss, boundary_loss, residual

    def get_residual(self):

        Uxx = math.prod([self.KInvCrossCol_Uxx[i].T@self.H[i] for i in range(self.args.n_order)])
        Uxx = jnp.sum(Uxx, 1)
        Uyy = math.prod([self.KInvCrossCol_Uyy[i].T@self.H[i] for i in range(self.args.n_order)])
        Uyy = jnp.sum(Uyy, 1)
        Uzz = math.prod([self.KInvCrossCol_Uzz[i].T@self.H[i] for i in range(self.args.n_order)])
        Uzz = jnp.sum(Uzz, 1)
        Uww = math.prod([self.KInvCrossCol_Uww[i].T@self.H[i] for i in range(self.args.n_order)])
        Uww = jnp.sum(Uww, 1)
        Uvv = math.prod([self.KInvCrossCol_Uvv[i].T@self.H[i] for i in range(self.args.n_order)])
        Uvv = jnp.sum(Uvv, 1)
        Uss = math.prod([self.KInvCrossCol_Uss[i].T@self.H[i] for i in range(self.args.n_order)])
        Uss = jnp.sum(Uss, 1)

        U = math.prod([self.KInvCrossCol_U[i].T@self.H[i] for i in range(self.args.n_order)]) #Nc x R
        U = jnp.sum(U, 1) #Nc
        g_pred = Uxx + Uyy + Uzz + Uww + Uvv + Uss + U*U*U - U #Nc
        residual = jnp.square(g_pred - self.y_c).mean()
        return residual 
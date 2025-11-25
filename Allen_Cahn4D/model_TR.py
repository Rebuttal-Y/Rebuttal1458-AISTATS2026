from kernels import *


class GP_ALS_4D(object):
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

        #factor fucntions for u_xx
        KCrossCol_Uxx = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].DD_x1_kappa, self.ker_params[0]), 
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3])] #Nc x M
        KInvCrossCol_Uxx = [jnp.linalg.solve(K[j], KCrossCol_Uxx[j].T) for j in range(self.args.n_order)] #M x Nc

        #factor fucntions for u_yy
        KCrossCol_Uyy = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].DD_x1_kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3])] #Nc x M 
        KInvCrossCol_Uyy = [jnp.linalg.solve(K[j], KCrossCol_Uyy[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_zz
        KCrossCol_Uzz = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].DD_x1_kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3])] #Nc x M 
        KInvCrossCol_Uzz = [jnp.linalg.solve(K[j], KCrossCol_Uzz[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_ww
        KCrossCol_Uww = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]),
                         cross_kernel(self.X_c[:, 2], self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]),
                         cross_kernel(self.X_c[:, 3], self.X_inducing[3], self.ker_func[3].DD_x1_kappa, self.ker_params[3])] #Nc x M 
        KInvCrossCol_Uww = [jnp.linalg.solve(K[j], KCrossCol_Uww[j].T) for j in range(self.args.n_order)] 
        
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
        self.KInvCrossCol_U = KInvCrossCol_U


    def update(self):
        n_col = self.X_c.shape[0]
        n_bound = self.X_b.shape[0]

        for j in range(self.args.n_order):

            left_bound = np.tile(np.eye(self.args.rank[0])[None, ...], (n_bound, 1, 1))
            left_u = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_uxx = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_uyy = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_uzz = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            left_uww = np.tile(np.eye(self.args.rank[0])[None, ...], (n_col, 1, 1))
            for i in range(j):
                left_u = jnp.einsum("bac,cbd->bad", left_u, self.KInvCrossCol_U[i].T @ self.H[i])
                left_uxx = jnp.einsum("bac,cbd->bad", left_uxx, self.KInvCrossCol_Uxx[i].T @ self.H[i])
                left_uyy = jnp.einsum("bac,cbd->bad", left_uyy, self.KInvCrossCol_Uyy[i].T @ self.H[i])
                left_uzz = jnp.einsum("bac,cbd->bad", left_uzz, self.KInvCrossCol_Uzz[i].T @ self.H[i])
                left_uww = jnp.einsum("bac,cbd->bad", left_uww, self.KInvCrossCol_Uww[i].T @ self.H[i])

                left_bound = jnp.einsum("bac,cbd->bad", left_bound, self.KInvCrossBoundary[i].T @ self.H[i])

            right_bound = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_bound))
            right_u = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_uxx = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_uyy = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_uzz = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            right_uww = np.tile(np.eye(self.args.rank[0])[..., None], (1, 1, n_col))
            for k in reversed(range(j+1, self.args.n_order)):
                right_u = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_U[k].T @ self.H[k], right_u)
                right_uxx = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_Uxx[k].T @ self.H[k], right_uxx)
                right_uyy = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_Uyy[k].T @ self.H[k], right_uyy)
                right_uzz = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_Uzz[k].T @ self.H[k], right_uzz)
                right_uww = jnp.einsum("abc,cdb->adb", self.KInvCrossCol_Uww[k].T @ self.H[k], right_uww)

                right_bound = jnp.einsum("abc,cdb->adb", self.KInvCrossBoundary[k].T @ self.H[k], right_bound)

            #u
            A0 = self.KInvCrossCol_U[j].T
            B0 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_u, left_u), (0, 2, 1)).reshape(n_col, -1)
            coef = jnp.einsum("bac,cbd,deb->bae", left_u, A0 @ self.H[j], right_u)
            coef0 = jnp.einsum("bii->b", coef)
            coef1 = coef0*coef0
            coef2 = coef0*coef0*coef0
            #uxx
            A1 = self.KInvCrossCol_Uxx[j].T
            B1 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_uxx, left_uxx), (0, 2, 1)).reshape(n_col, -1)
            #uyy
            A2 = self.KInvCrossCol_Uyy[j].T
            B2 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_uyy, left_uyy), (0, 2, 1)).reshape(n_col, -1)
            #uzz
            A3 = self.KInvCrossCol_Uzz[j].T
            B3 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_uzz, left_uzz), (0, 2, 1)).reshape(n_col, -1)
            #uww
            A4 = self.KInvCrossCol_Uww[j].T
            B4 = jnp.transpose(jnp.einsum("acb,bcd->bad", right_uww, left_uww), (0, 2, 1)).reshape(n_col, -1)

            Phi2 = jnp.zeros((0, A0.shape[-1], B0.shape[-1]))
            idx_start = 0

            while idx_start < n_col:
                idx_end = min(idx_start + self.args.n_train_batch, n_col)
                if self.args.Newton_M:
                    Phi2 = jnp.concatenate((Phi2, jnp.einsum('ij,ik->ijk', A1[idx_start:idx_end, ...], B1[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A2[idx_start:idx_end, ...], B2[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A3[idx_start:idx_end, ...], B3[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A4[idx_start:idx_end, ...], B4[idx_start:idx_end, ...]) + \
                        (3.0*coef1[idx_start:idx_end, None, None] - 1.0) * jnp.einsum('ij,ik->ijk', A0[idx_start:idx_end, ...], B0[idx_start:idx_end, ...])), axis=0)
                else:
                    Phi2 = jnp.concatenate((Phi2, jnp.einsum('ij,ik->ijk', A1[idx_start:idx_end, ...], B1[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A2[idx_start:idx_end, ...], B2[idx_start:idx_end, ...]) + jnp.einsum('ij,ik->ijk', A3[idx_start:idx_end, ...], B3[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A4[idx_start:idx_end, ...], B4[idx_start:idx_end, ...]) + jnp.einsum('ij,ik->ijk', A0[idx_start:idx_end, ...], \
                            B0[idx_start:idx_end, ...])*(coef1[idx_start:idx_end, None, None] - 1.0)), axis=0)
                idx_start = idx_end
            if self.args.CPU_PDE:
                Phi2 = jax.device_put(Phi2.reshape(n_col, -1), device=jax.devices("cpu")[0])
                Phi = self.args.lam2/self.args.lam1*jax.device_put(Phi2.T@Phi2, device=jax.devices("gpu")[0])
            else:
                Phi2 = Phi2.reshape(n_col, -1)
                Phi = self.args.lam2/self.args.lam1*(Phi2.T@Phi2)

            #boundary condition
            A = self.KInvCrossBoundary[j].T
            B = jnp.transpose(jnp.einsum("acb,bcd->bad", right_bound, left_bound), (0, 2, 1)).reshape(n_bound, -1)
            
            Phi1 = jnp.einsum('ij,ik->ijk', A, B)
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

    def solution_approx(self, x1, x2, x3, x4):
        KCross = [cross_kernel(x1, self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]), 
                  cross_kernel(x2, self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1]), 
                  cross_kernel(x3, self.X_inducing[2], self.ker_func[2].kappa, self.ker_params[2]), 
                  cross_kernel(x4, self.X_inducing[3], self.ker_func[3].kappa, self.ker_params[3])] #N x M
        KInvInducing = [jnp.linalg.solve(self.K[j], self.H[j]) for j in range(self.args.n_order)] #M x R
        res = KCross[0] @ KInvInducing[0]
        for i in range(1, self.args.n_order):
            res = jnp.einsum("abc,cbd->abd", res, KCross[i] @ KInvInducing[i])
        # res = jnp.sum(res)
        res = jnp.trace(res[:, 0, :])
        return res 

    def get_loss(self):
        boundary_pred = vmap(self.solution_approx, (0, 0, 0, 0))(self.X_b[:, 0], self.X_b[:, 1], self.X_b[:, 2], self.X_b[:, 3])
        boundary_loss = jnp.square(boundary_pred - self.y_b).mean()
        uxx = grad(grad(self.solution_approx, 0), 0)
        uyy = grad(grad(self.solution_approx, 1), 1)
        uzz = grad(grad(self.solution_approx, 2), 2)
        uww = grad(grad(self.solution_approx, 3), 3)
        u_pred = vmap(self.solution_approx, (0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3]) #N_c
        uxx_pred = vmap(uxx, (0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3]) #N_c
        uyy_pred = vmap(uyy, (0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3]) #N_c
        uzz_pred = vmap(uzz, (0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3]) #N_c
        uww_pred = vmap(uww, (0, 0, 0, 0))(self.X_c[:, 0], self.X_c[:, 1], self.X_c[:, 2], self.X_c[:, 3]) #N_c
        g_pred = uxx_pred + uyy_pred + uzz_pred + uww_pred + u_pred*u_pred*u_pred - u_pred #N_c
        residual = jnp.square(g_pred - self.y_c).mean() #collocation loss
        loss = boundary_loss + residual #total loss
        return loss, boundary_loss, residual

    def get_residual(self):
        
        Uxx = self.KInvCrossCol_Uxx[0].T@self.H[0]
        Uyy = self.KInvCrossCol_Uyy[0].T@self.H[0]
        Uzz = self.KInvCrossCol_Uzz[0].T@self.H[0]
        Uww = self.KInvCrossCol_Uww[0].T@self.H[0]
        U = self.KInvCrossCol_U[0].T@self.H[0]
        for i in range(1, self.args.n_order):
            Uxx = jnp.einsum("abc,cbd->abd", Uxx, self.KInvCrossCol_Uxx[i].T@self.H[i])
            Uyy = jnp.einsum("abc,cbd->abd", Uyy, self.KInvCrossCol_Uyy[i].T@self.H[i])
            Uzz = jnp.einsum("abc,cbd->abd", Uzz, self.KInvCrossCol_Uzz[i].T@self.H[i])
            Uww = jnp.einsum("abc,cbd->abd", Uww, self.KInvCrossCol_Uww[i].T@self.H[i])
            U = jnp.einsum("abc,cbd->abd", U, self.KInvCrossCol_U[i].T@self.H[i])

        Uxx = jnp.einsum("ibi->b", Uxx)
        Uyy = jnp.einsum("ibi->b", Uyy)
        Uzz = jnp.einsum("ibi->b", Uzz)
        Uww = jnp.einsum("ibi->b", Uww)
        U = jnp.einsum("ibi->b", U)

        g_pred = Uxx + Uyy + Uzz + Uww + U*U*U - U #Nc
        residual = jnp.square(g_pred.reshape(-1) - self.y_c).mean()
        return residual 
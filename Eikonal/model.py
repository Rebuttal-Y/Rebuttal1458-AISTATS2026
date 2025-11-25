from kernels import *


class GP_ALS(object):
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
        
        self.X_b = self.data_gen.x_b
        self.X_c = self.data_gen.x_c
        self.y_c = self.data_gen.y_c

        self._prepare()

    def _prepare(self):
        #M x M kernel matrix in each dimension
        K = [kernel_matrix(self.X_inducing[j], self.ker_func[j].kappa, self.ker_params[j], self.jitter[j]) for j in range(self.args.n_order)] #
        Kinv =[jnp.linalg.solve(K[j], jnp.eye(K[j].shape[0])) for j in range(self.args.n_order)] #M x M, inverse of the kernel matrix

        #boundary points
        KCrossBoundary = [cross_kernel(self.X_b[:, j], self.X_inducing[j], self.ker_func[j].kappa, self.ker_params[j]) for j in range(self.args.n_order)] #Nb x M
        KInvCrossBoundary = [jnp.linalg.solve(K[j], KCrossBoundary[j].T) for j in range(self.args.n_order)] # M x Nb
        
        #factor fucntions for u_x
        KCrossCol_Ux = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].D_x1_kappa, self.ker_params[0]), 
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1])] #Nc x M
        KInvCrossCol_Ux = [jnp.linalg.solve(K[j], KCrossCol_Ux[j].T) for j in range(self.args.n_order)] #M x Nc

        #factor fucntions for u_xx
        KCrossCol_Uxx = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].DD_x1_kappa, self.ker_params[0]), 
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1])] #Nc x M
        KInvCrossCol_Uxx = [jnp.linalg.solve(K[j], KCrossCol_Uxx[j].T) for j in range(self.args.n_order)] #M x Nc
        
        #factor fucntions for u_y
        KCrossCol_Uy = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].D_x1_kappa, self.ker_params[1])] #Nc x M 
        KInvCrossCol_Uy = [jnp.linalg.solve(K[j], KCrossCol_Uy[j].T) for j in range(self.args.n_order)] 

        #factor fucntions for u_yy
        KCrossCol_Uyy = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].DD_x1_kappa, self.ker_params[1])] #Nc x M 
        KInvCrossCol_Uyy = [jnp.linalg.solve(K[j], KCrossCol_Uyy[j].T) for j in range(self.args.n_order)] 
        
        #M x Nc
        self.K = K
        self.Kinv = Kinv
        self.KInvCrossBoundary = KInvCrossBoundary
        self.KInvCrossCol_Ux = KInvCrossCol_Ux
        self.KInvCrossCol_Uxx = KInvCrossCol_Uxx
        self.KInvCrossCol_Uy = KInvCrossCol_Uy
        self.KInvCrossCol_Uyy = KInvCrossCol_Uyy



    def update(self):

        for j in range(2):
            o = 1 - j #the other dimension
            #update H[j]
            #equation part
            #u_x
            A0 = self.KInvCrossCol_Ux[j].T #Nc x M
            B0 = self.KInvCrossCol_Ux[o].T @ self.H[o] #Nc x R
            coef0 = jnp.sum((A0 @ self.H[j]) * B0, 1)
            #u_y
            A1 = self.KInvCrossCol_Uy[j].T #Nc x M
            B1 = self.KInvCrossCol_Uy[o].T @ self.H[o] #Nc x R
            coef1 = jnp.sum((A1 @ self.H[j]) * B1, 1)
            #u_xx
            A2 = self.KInvCrossCol_Uxx[j].T #Nc x M
            B2 = self.KInvCrossCol_Uxx[o].T @ self.H[o] #Nc x R
            #u_yy   
            A3 = self.KInvCrossCol_Uyy[j].T #Nc x M
            B3 = self.KInvCrossCol_Uyy[o].T @ self.H[o] #Nc x R
            
            
            Phi2 = jnp.zeros((0, A0.shape[-1], B0.shape[-1]))
            idx_start = 0
            n_total = A0.shape[0]
            while idx_start < n_total:
                idx_end = min(idx_start + self.args.n_train_batch, n_total)
                u_x = jnp.einsum('ij,ik->ijk', A0[idx_start:idx_end, ...], B0[idx_start:idx_end, ...])
                u_y = jnp.einsum('ij,ik->ijk', A1[idx_start:idx_end, ...], B1[idx_start:idx_end, ...])
                u_xx = jnp.einsum('ij,ik->ijk', A2[idx_start:idx_end, ...], B2[idx_start:idx_end, ...])
                u_yy = jnp.einsum('ij,ik->ijk', A3[idx_start:idx_end, ...], B3[idx_start:idx_end, ...])
                if self.args.Newton_M:
                    # Phi2 = jnp.concatenate((Phi2, 2.0*coef0[idx_start:idx_end, None, None]*u_x - coef0[idx_start:idx_end, None, None]*coef0[idx_start:idx_end, None, None] + \
                    #     2.0*coef1[idx_start:idx_end, None, None]*u_y - coef1[idx_start:idx_end, None, None]*coef1[idx_start:idx_end, None, None] - \
                    #         self.args.epsilon * (u_xx + u_yy)), axis=0)
                    Phi2 = jnp.concatenate((Phi2, 2.0*coef0[idx_start:idx_end, None, None]*u_x + 2.0*coef1[idx_start:idx_end, None, None]*u_y - \
                            self.args.epsilon * (u_xx + u_yy)), axis=0)
                else:
                    Phi2 = jnp.concatenate((Phi2, u_x*coef0[idx_start:idx_end, None, None] + u_y*coef1[idx_start:idx_end, None, None] - self.args.epsilon * (u_xx + u_yy)), axis=0)

                # Phi2 = jnp.concatenate((Phi2, jnp.einsum('ij,ik->ijk', A0[idx_start:idx_end, ...], B0[idx_start:idx_end, ...])*coef0[idx_start:idx_end, None, None] + \
                #     jnp.einsum('ij,ik->ijk', A1[idx_start:idx_end, ...], B1[idx_start:idx_end, ...])*coef1[idx_start:idx_end, None, None] - \
                #         self.args.epsilon * (jnp.einsum('ij,ik->ijk', A2[idx_start:idx_end, ...], B2[idx_start:idx_end, ...]) + jnp.einsum('ij,ik->ijk', \
                #             A3[idx_start:idx_end, ...], B3[idx_start:idx_end, ...]))), axis=0)
                
                idx_start = idx_end
            if self.args.CPU_PDE:
                Phi2 = jax.device_put(Phi2.reshape(n_total, -1), device=jax.devices("cpu")[0]) #Nc x (M*R)
                Phi = self.args.lam2/self.args.lam1*jax.device_put(Phi2.T@Phi2, device=jax.devices("gpu")[0])
            else:
                Phi2 = Phi2.reshape(n_total, -1) #Nc x (M*R)
                Phi = self.args.lam2/self.args.lam1*(Phi2.T@Phi2)
                
            #boundary condition
            A = self.KInvCrossBoundary[j].T #Nb x M
            B = self.KInvCrossBoundary[o].T@self.H[o] #Nb x R
            Phi1 = jnp.einsum('ij,ik->ijk', A, B) #Nb x M x R
            Phi1 = Phi1.reshape(Phi1.shape[0], -1) #Nb x (M*R)
            Phi += Phi1.T@Phi1

            Phi += 1/self.args.lam1*jnp.kron(self.Kinv[j], jnp.eye(self.args.rank))
            if self.args.Newton_M:
                eta = self.args.lam2/self.args.lam1*Phi2.T@(self.y_c + coef1*coef1 + coef0*coef0) # (M*R)
            else:
                eta = self.args.lam2/self.args.lam1*Phi2.T@self.y_c # (M*R) 
            
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

    def solution_approx(self, x1, x2):
        KCross = [cross_kernel(x1, self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]), cross_kernel(x2, self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1])] #N x M
        KInvInducing = [jnp.linalg.solve(self.K[j], self.H[j]) for j in range(self.args.n_order)] #M x R
        F1 = KCross[0] @ KInvInducing[0] #N x R
        F2 = KCross[1] @ KInvInducing[1]
        F = F1 * F2 #N x R
        res = jnp.sum(F) #N
        return res 

    def get_loss(self):
        boundary_pred = vmap(self.solution_approx, (0, 0))(self.X_b[:, 0], self.X_b[:, 1])
        boundary_loss = jnp.square(boundary_pred).mean()
        ux = grad(self.solution_approx, 0)
        uy = grad(self.solution_approx, 1)
        uxx = grad(grad(self.solution_approx, 0), 0)
        uyy = grad(grad(self.solution_approx, 1), 1)
        ux_pred = vmap(ux, (0, 0))(self.X_c[:, 0], self.X_c[:, 1]) #N_c
        uy_pred = vmap(uy, (0, 0))(self.X_c[:, 0], self.X_c[:, 1]) #N_c
        uxx_pred = vmap(uxx, (0, 0))(self.X_c[:, 0], self.X_c[:, 1]) #N_c
        uyy_pred = vmap(uyy, (0, 0))(self.X_c[:, 0], self.X_c[:, 1]) #N_c
        g_pred = ux_pred*ux_pred + uy_pred*uy_pred - self.args.epsilon*(uxx_pred + uyy_pred) #N_c
        residual = jnp.square(g_pred - self.y_c).mean() #collocation loss
        loss = boundary_loss + residual #total loss
        return loss, boundary_loss, residual

    def get_residual(self):
        Ux = (self.KInvCrossCol_Ux[0].T@self.H[0])*(self.KInvCrossCol_Ux[1].T@self.H[1])
        Ux = jnp.sum(Ux, 1)
        Uy = (self.KInvCrossCol_Uy[0].T@self.H[0])*(self.KInvCrossCol_Uy[1].T@self.H[1])
        Uy = jnp.sum(Uy, 1)
        Uxx = (self.KInvCrossCol_Uxx[0].T@self.H[0])*(self.KInvCrossCol_Uxx[1].T@self.H[1]) 
        Uxx = jnp.sum(Uxx, 1)
        Uyy = (self.KInvCrossCol_Uyy[0].T@self.H[0])*(self.KInvCrossCol_Uyy[1].T@self.H[1])
        Uyy = jnp.sum(Uyy, 1)
        g_pred = Ux*Ux + Uy*Uy - self.args.epsilon*(Uxx + Uyy) #Nc
        residual = jnp.square(g_pred - self.y_c).mean()
        return residual 
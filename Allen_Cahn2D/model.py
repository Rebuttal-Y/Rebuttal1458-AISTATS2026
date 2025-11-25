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
        self.y_b = self.data_gen.y_b
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

        #factor fucntions for u_xx
        KCrossCol_Uxx = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].DD_x1_kappa, self.ker_params[0]), 
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].kappa, self.ker_params[1])] #Nc x M
        KInvCrossCol_Uxx = [jnp.linalg.solve(K[j], KCrossCol_Uxx[j].T) for j in range(self.args.n_order)] #M x Nc

        #factor fucntions for u_yy
        KCrossCol_Uyy = [cross_kernel(self.X_c[:, 0], self.X_inducing[0], self.ker_func[0].kappa, self.ker_params[0]),
                         cross_kernel(self.X_c[:, 1], self.X_inducing[1], self.ker_func[1].DD_x1_kappa, self.ker_params[1])] #Nc x M 
        KInvCrossCol_Uyy = [jnp.linalg.solve(K[j], KCrossCol_Uyy[j].T) for j in range(self.args.n_order)] 
        
        #u
        KcrossCol_U = [cross_kernel(self.X_c[:, j], self.X_inducing[j], self.ker_func[j].kappa, self.ker_params[j]) for j in range(self.args.n_order)] #Nc x M
        KInvCrossCol_U = [jnp.linalg.solve(K[j], KcrossCol_U[j].T) for j in range(self.args.n_order)]
        #M x Nc
        self.K = K
        self.Kinv = Kinv
        self.KInvCrossBoundary = KInvCrossBoundary
        self.KInvCrossCol_Uxx = KInvCrossCol_Uxx
        self.KInvCrossCol_Uyy = KInvCrossCol_Uyy
        self.KInvCrossCol_U = KInvCrossCol_U


    def update(self):

        for j in range(2):
            o = 1 - j #the other dimension
            #update H[j]
            #equation part
            #u
            A0 = self.KInvCrossCol_U[j].T #Nc x M
            B0 = self.KInvCrossCol_U[o].T @ self.H[o] #Nc x R
            coef0 = jnp.sum((A0 @ self.H[j]) * B0, 1) #Nc
            coef1 = coef0*coef0
            coef2 = coef0*coef0*coef0
            #u_xx
            A1 = self.KInvCrossCol_Uxx[j].T #Nc x M
            B1 = self.KInvCrossCol_Uxx[o].T @ self.H[o] #Nc x R 
            #u_yy   
            A2 = self.KInvCrossCol_Uyy[j].T #Nc x M
            B2 = self.KInvCrossCol_Uyy[o].T @ self.H[o] #Nc x R
            
            
            Phi2 = jnp.zeros((0, A0.shape[-1], B0.shape[-1]))
            idx_start = 0
            n_total = A0.shape[0]
            while idx_start < n_total:
                idx_end = min(idx_start + self.args.n_train_batch, n_total)
                if self.args.Newton_M:
                    Phi2 = jnp.concatenate((Phi2, jnp.einsum('ij,ik->ijk', A1[idx_start:idx_end, ...], B1[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A2[idx_start:idx_end, ...], B2[idx_start:idx_end, ...]) + jnp.einsum('ij,ik->ijk', A0[idx_start:idx_end, ...], \
                            B0[idx_start:idx_end, ...])*(3.0*coef1[idx_start:idx_end, None, None] - 1.0) - 2.0*coef2[idx_start:idx_end, None, None]), axis=0)
                else:
                    Phi2 = jnp.concatenate((Phi2, jnp.einsum('ij,ik->ijk', A1[idx_start:idx_end, ...], B1[idx_start:idx_end, ...]) + \
                        jnp.einsum('ij,ik->ijk', A2[idx_start:idx_end, ...], B2[idx_start:idx_end, ...]) + jnp.einsum('ij,ik->ijk', A0[idx_start:idx_end, ...], \
                            B0[idx_start:idx_end, ...])*(coef1[idx_start:idx_end, None, None] - 1.0)), axis=0)
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
        boundary_loss = jnp.square(boundary_pred - self.y_b).mean()
        uxx = grad(grad(self.solution_approx, 0), 0)
        uyy = grad(grad(self.solution_approx, 1), 1)
        u_pred = vmap(self.solution_approx, (0, 0))(self.X_c[:, 0], self.X_c[:, 1]) #N_c
        uxx_pred = vmap(uxx, (0, 0))(self.X_c[:, 0], self.X_c[:, 1]) #N_c
        uyy_pred = vmap(uyy, (0, 0))(self.X_c[:, 0], self.X_c[:, 1]) #N_c
        g_pred = uxx_pred + uyy_pred + u_pred*(u_pred*u_pred - 1.0) #N_c
        residual = jnp.square(g_pred - self.y_c).mean() #collocation loss
        loss = boundary_loss + residual #total loss
        return loss, boundary_loss, residual

    def get_residual(self):
        #u_xx
        Uxx = (self.KInvCrossCol_Uxx[0].T@self.H[0])*(self.KInvCrossCol_Uxx[1].T@self.H[1]) 
        Uxx = jnp.sum(Uxx, 1)
        Uyy = (self.KInvCrossCol_Uyy[0].T@self.H[0])*(self.KInvCrossCol_Uyy[1].T@self.H[1])
        Uyy = jnp.sum(Uyy, 1)
        U = (self.KInvCrossCol_U[0].T@self.H[0])*(self.KInvCrossCol_U[1].T@self.H[1]) #Nc x R
        U = jnp.sum(U, 1) #Nc
        g_pred = Uxx + Uyy + U*(U*U - 1.0) #Nc
        residual = jnp.square(g_pred - self.y_c).mean()
        return residual 
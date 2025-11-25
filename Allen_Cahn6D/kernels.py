from utilities import *


class Gaussian_kernel_1d(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, para):
        return (jnp.exp(- 0.5*(x1 - y1)**2 / jnp.exp(2*para['log-ls']))).sum()

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras):  # cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val
    
    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras):  # cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val
    

class SM_kernel_u_1d(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, paras):
        return (jnp.exp(paras['log-w'])*jnp.exp(-(x1-y1)**2*jnp.exp(paras['log-ls']))*jnp.cos(2*jnp.pi*(x1-y1)*paras['freq'])).sum()

    


    
class RQ_kernel_1d(object):

    def __init__(self):
        self.alpha=1.0
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, para):
        return (jnp.power(1 + 0.5*(x1 - y1)**2 / jnp.exp(2*para['log-ls']), -1*self.alpha)).sum()



class Matern12_kernel_1d(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, para):
        d = jnp.abs(x1-y1)/jnp.exp(para['log-ls'])
        return jnp.exp(-d)

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras):  # cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val
    
    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras):  # cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val




class Matern32_kernel_1d(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, para):
        d = jnp.abs(x1-y1)/jnp.exp(para['log-ls'])
        return (1 + jnp.sqrt(3)*d)*jnp.exp(-jnp.sqrt(3)*d)

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras):  # cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val
    
    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras):  # cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val

class Matern52_kernel_1d(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, para):
        d = jnp.abs(x1-y1)/jnp.exp(para['log-ls'])
        return (1 + jnp.sqrt(5)*d + 5.0/3*d*d)*jnp.exp(-jnp.sqrt(5)*d)

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras):  # cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val
    
    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras):  # cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val

class Matern72_kernel_1d(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, para):
        d = jnp.abs(x1-y1)/jnp.exp(para['log-ls']) * jnp.sqrt(7)
        return (1.0 + d + d**2 / 5.0*2 + d**3 / 15.0)*jnp.exp(-d)

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras):  # cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val
    
    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras):  # cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val

class Matern92_kernel_1d(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, para):
        d = jnp.abs(x1-y1)/jnp.exp(para['log-ls']) * 3.0
        return (1.0 + d + (d**2)*(3.0/7.0) + (d**3)*(2.0/21) + (1.0/105)*(d**4))*jnp.exp(-d)

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, paras):  # cov(f'(x1), f(y1))
        val = grad(self.kappa, 0)(x1, y1, paras)
        return val
    
    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, paras):  # cov(f''(x1), f(y1))
        val = grad(grad(self.kappa, 0), 0)(x1, y1, paras)
        return val
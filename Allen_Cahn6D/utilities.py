import numpy as np

import tensorly as tl
import optax
import jax
import jax.numpy as jnp
from jax import vmap, grad, jit
from functools import partial

jax.config.update("jax_enable_x64", True)

import math

import configparser


def write_log(args):
    config = configparser.ConfigParser()

    config['DEFAULT'] = vars(args)
    with open(args.log_store_path + ".ini", 'w+') as configfile:
        config.write(configfile)

def write_records(args, rmse, mae, relativeL1, relativeL2, best_test_rmse, best_test_mae, best_test_relativeL1, best_test_relativeL2, best_test_epoch_rmse, best_test_epoch_mae, \
    best_test_epoch_relativeL1, best_test_epoch_relativeL2, epoch):
    log = "Training at epoch: [" + str(epoch) + "]\n"
    log += "test RMSE-(" + str(rmse) + "), test L2 Relative mean error-(" + str(relativeL2) + "), test MAE error-(" + str(mae) + "), test L1 Relative error-(" + str(relativeL1) + ")\n"
    log += "best test RMSE: (" + str(best_test_rmse) + ") at epoch-[" + str(best_test_epoch_rmse) + "]\n"
    log += "best test L1 relative error: (" + str(best_test_relativeL1) + ") at epoch-[" + str(best_test_epoch_relativeL1) + "]\n"
    log += "best test L2 relative mean error: (" + str(best_test_relativeL2) + ") at epoch-[" + str(best_test_epoch_relativeL2) + "]\n"
    log += "best test MAE: (" + str(best_test_mae) + ") at epoch-[" + str(best_test_epoch_mae) + "]\n"
    with open(args.log_store_path + ".ini", 'a') as configfile:
        configfile.write(log)
        configfile.write("*************************************************************************\n\n")
        
        
@partial(jit, static_argnums=(2, ))
def cross_kernel(x1, x2, ker_func, ker_params):
    x1_p = jnp.tile(x1.flatten(), (x2.size, 1)).T
    x2_p = jnp.tile(x2.flatten(), (x1.size, 1)).T
    x1_p = x1_p.flatten()
    x2_p = x2_p.transpose().flatten()
    Kmn = vmap(ker_func, (0, 0, None))(x1_p.flatten(), x2_p.flatten(), ker_params).reshape(x1.size, x2.size)
    return Kmn
    

@partial(jit, static_argnums=(1, ))
def kernel_matrix(x1, ker_func, ker_params, jitter):
    K = cross_kernel(x1, x1, ker_func, ker_params)
    K = K + jitter * jnp.eye(x1.size)
    return K
    
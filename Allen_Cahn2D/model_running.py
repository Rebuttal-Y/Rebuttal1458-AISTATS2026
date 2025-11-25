from utilities import *


def do_analysis(model, data_gen, change, iter, err):
    pred_tr = model.pred(jnp.array(data_gen.x_b)).reshape(-1)
    ground_truth = jnp.array(data_gen.y_b).reshape(-1)
    assert pred_tr.size == ground_truth.size, 'pred and ground_truth should have the same number of locations'
    err_tr  = jnp.sqrt(jnp.sum(jnp.square(pred_tr - ground_truth)) / jnp.sum(jnp.square(ground_truth)))
    loss, loss_b, loss_c = model.get_loss() #scalar
    loss_c2 = model.get_residual() #scalar
    print('Iteration %d, change = %g, relative L2 err = %g, err_b = %g, loss=%g, b_loss = %g, residual=%g, res2=%g'%(iter, change, err, err_tr, loss, loss_b, loss_c, loss_c2))

def test(model, data_gen):
    pred = model.pred(jnp.array(data_gen.X_test)).reshape(-1)
    ground_truth = jnp.array(data_gen.Y_test).reshape(-1)
    assert pred.size == ground_truth.size, 'pred and ground_truth should have the same number of locations'
    diff = pred - ground_truth
    diff_square = jnp.square(diff)
    diff_abs = jnp.abs(diff)
    relativeL2 = jnp.sqrt(jnp.sum(diff_square))/data_gen.YNorm2_test
    relativeL1 = jnp.sum(diff_abs)/data_gen.YSUM_ABS
    rmse = jnp.sqrt(jnp.mean(diff_square))
    mae = jnp.mean(diff_abs)
    return rmse, mae, relativeL2, relativeL1


def train(model):
    model.update() #update H


def model_run(model, data_gen):

    best_test_epoch_rmse, best_test_epoch_mae, best_test_epoch_relativeL2, best_test_epoch_relativeL1 = -1, -1, -1, -1
    best_test_rmse, best_test_mae, best_test_relativeL2, best_test_relativeL1 = math.inf, math.inf, math.inf, math.inf
    start_epoch = 0
    count = 0

    while start_epoch < model.args.epochs:
        old_H = [model.H[j].copy() for j in range(model.args.n_order)] #M x R
        train(model)
        rmse, mae, relativeL2, relativeL1 = test(model, data_gen)
        count += 1
        if rmse < best_test_rmse:
            best_test_rmse = rmse
            best_test_epoch_rmse = start_epoch
            count = 0

        if mae < best_test_mae:
            best_test_mae = mae
            best_test_epoch_mae = start_epoch
            
        if relativeL2 < best_test_relativeL2:
            best_test_relativeL2 = relativeL2
            best_test_epoch_relativeL2 = start_epoch
            count = 0

        if relativeL1 < best_test_relativeL1:
            best_test_relativeL1 = relativeL1
            best_test_epoch_relativeL1 = start_epoch

        if count == model.args.early_stop or start_epoch  == (model.args.epochs - 1):
            write_records(data_gen.args, rmse, mae, relativeL1, relativeL2, best_test_rmse, best_test_mae, best_test_relativeL1, best_test_relativeL2, best_test_epoch_rmse, best_test_epoch_mae, best_test_epoch_relativeL1, best_test_epoch_relativeL2, start_epoch)
            break

        change = jnp.linalg.norm(model.H[0] - old_H[0])/jnp.linalg.norm(old_H[0]) + jnp.linalg.norm(model.H[1] - old_H[1])/jnp.linalg.norm(old_H[1]) #scalar
        if change < model.args.stop_criteria:
            write_records(data_gen.args, rmse, mae, relativeL1, relativeL2, best_test_rmse, best_test_mae, best_test_relativeL1, best_test_relativeL2, best_test_epoch_rmse, best_test_epoch_mae, best_test_epoch_relativeL1, best_test_epoch_relativeL2, start_epoch)
            print('Converged after %d iterations'%(start_epoch))
            break

        if model.args.analysis:
            do_analysis(model, data_gen, change, start_epoch, relativeL2)


        if start_epoch % model.args.log_interval == 0:
            write_records(data_gen.args, rmse, mae, relativeL1, relativeL2, best_test_rmse, best_test_mae, best_test_relativeL1, best_test_relativeL2, best_test_epoch_rmse, best_test_epoch_mae, best_test_epoch_relativeL1, best_test_epoch_relativeL2, start_epoch)

        start_epoch += 1


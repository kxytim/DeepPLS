import torch as th


def _get_first_singular_vectors_power_method(X, Y, max_iter=500, tol=1e-06):
    eps = th.finfo(th.float64).eps
    try:
        y_score = next(col for col in Y.T if th.any(th.abs(col) > eps))
    except StopIteration as e:
        raise StopIteration("Y residual is constant") from e

    x_weights_old = 100
    for i in range(max_iter):
        x_weights = th.matmul(X.T, y_score) / th.matmul(y_score, y_score)
        x_weights /= th.sqrt(th.matmul(x_weights, x_weights)) + eps
        x_score = th.matmul(X, x_weights)
        y_weights = th.matmul(Y.T, x_score) / th.matmul(x_score.T, x_score)
        y_score = th.matmul(Y, y_weights) / (th.matmul(y_weights, y_weights) + eps)
        x_weights_diff = x_weights - x_weights_old
        if th.matmul(x_weights_diff, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        x_weights_old = x_weights
    n_iter = i + 1
    return x_weights, y_weights, n_iter


def _get_first_singular_vectors_svd(X, Y):
    C = th.matmul(X.T, Y)
    U, _, Vt = th.linalg.svd(C)
    return U[:, 0], Vt[0, :]


def _svd_flip_1d(u, v):
    biggest_abs_val_idx = th.argmax(th.abs(u))
    sign = th.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign

from sklearn.kernel_approximation import Nystroem
import torch as th
from utils import _get_first_singular_vectors_power_method, _get_first_singular_vectors_svd, _svd_flip_1d


class PLS(object):
    """Pytorch implementation of the basic PLS algorithm.

    Implemented with reference to the 'scikit-learn PLSRegression' model.
    'https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression'

    Parameters
    ----------
    n_components : int. Dimension of latent variables in the PLS algorithm.

    solver : {'iter', 'svd'}. Solver type of the PLS algorithm.

    max_iter : int. The maximum number of iterations of the power method.
        Only effective when solver == 'iter'.

    tol : float. The tolerance used as convergence criteria in the power method.
        Only effective when solver == 'iter'.
    """

    def __init__(self, n_components, solver, max_iter=500, tol=1e-06):
        self.n_components = n_components
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol

    def _fit(self, X, Y):
        X = X.clone()
        Y = Y.clone()
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        n_components = self.n_components

        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        Xk, Yk, self._x_mean, self._y_mean = X, Y, x_mean, y_mean

        self.x_weights_ = th.zeros((p, n_components))
        self.y_weights_ = th.zeros((q, n_components))
        self._x_scores = th.zeros((n, n_components))
        self._y_scores = th.zeros((n, n_components))
        self.x_loadings_ = th.zeros((p, n_components))
        self.y_loadings_ = th.zeros((q, n_components))
        self.n_iter_ = []

        Y_eps = th.finfo(th.float64).eps
        for k in range(n_components):
            Yk_mask = th.all(th.abs(Yk) < 10 * Y_eps, dim=0)
            Yk[:, Yk_mask] = 0.0
            if self.solver == 'iter':
                x_weights, y_weights, n_iter_ = _get_first_singular_vectors_power_method(
                    Xk, Yk, max_iter=self.max_iter, tol=self.tol
                )
                self.n_iter_.append(n_iter_)
            elif self.solver == 'svd':
                x_weights, y_weights = _get_first_singular_vectors_svd(Xk, Yk)
            else:
                raise NameError('PLS solver not supported')

            _svd_flip_1d(x_weights, y_weights)
            x_scores = th.matmul(Xk, x_weights)
            y_ss = th.matmul(y_weights, y_weights)
            y_scores = th.matmul(Yk, y_weights) / y_ss
            x_loadings = th.matmul(x_scores, Xk) / th.matmul(x_scores, x_scores)
            Xk -= th.einsum('i,j->ij', x_scores, x_loadings)
            y_loadings = th.matmul(x_scores, Yk) / th.matmul(x_scores, x_scores)
            Yk -= th.einsum('i,j->ij', x_scores, y_loadings)

            self.x_weights_[:, k] = x_weights
            self.y_weights_[:, k] = y_weights
            self._x_scores[:, k] = x_scores
            self._y_scores[:, k] = y_scores
            self.x_loadings_[:, k] = x_loadings
            self.y_loadings_[:, k] = y_loadings
            self.x_scores_ = self._x_scores
            self.y_scores_ = self._y_scores

        self.x_rotations_ = th.matmul(
            self.x_weights_,
            th.pinverse(th.matmul(self.x_loadings_.T, self.x_weights_)))
        self.y_rotations_ = th.matmul(
            self.y_weights_, th.pinverse(th.matmul(self.y_loadings_.T, self.y_weights_)))

        self.coef_ = th.matmul(self.x_rotations_, self.y_loadings_.T)
        return self

    def fit(self, X, Y):
        self._fit(X, Y)
        return self

    def predict(self, X):
        X = X.clone()
        X -= self._x_mean
        Y_pred = th.matmul(X, self.coef_)
        return Y_pred + self._y_mean

    def transform(self, X):
        X = X.clone()
        X -= self._x_mean
        x_scores = th.matmul(X, self.x_rotations_)
        return x_scores


class DeepPLS(object):
    """Deep PLS and generalized deep PLS models.

    Please see the article 'Deep PLS: A Lightweight Deep Learning Model for Interpretable and Efficient Data Analytics,
    https://dx.doi.org/10.1109/TNNLS.2022.3154090' for more details.

    Parameters
    ----------
    lv_dimensions : list of ints. Dimension of latent variables in each PLS layer.

    pls_solver : {'iter', 'svd'}. Solver type of the PLS algorithm.

    use_nonlinear_mapping : bool. Whether to use nonlinear mapping or not.

    mapping_dimensions : list of ints. Dimension of nonlinear features in each nonlinear mapping layer.
        Only effective when use_nonlinear_mapping == True.

    nys_gamma_values : list of floats. Gamma values of Nystroem function in each nonlinear mapping layer.
        Only effective when use_nonlinear_mapping == True.

    stack_previous_lv1 : bool. Whether to stack the first latent variable of the previous PLS layer
        into the current nonlinear features. See the right column of Page 8 in the original article.
        Only effective when use_nonlinear_mapping == True.
    """

    def __init__(
            self,
            lv_dimensions: list,
            pls_solver: str,
            use_nonlinear_mapping: bool,
            mapping_dimensions: list,
            nys_gamma_values: list,
            stack_previous_lv1: bool

    ):
        self.lv_dimensions = lv_dimensions
        self.n_layers = len(self.lv_dimensions)
        self.pls_solver = pls_solver
        self.latent_variables = []
        self.pls_funcs = []
        self.use_nonlinear_mapping = use_nonlinear_mapping
        self.mapping_dimensions = mapping_dimensions
        self.nys_gamma_values = nys_gamma_values
        self.mapping_funcs = []
        self.stack_previous_lv1 = stack_previous_lv1

        if self.use_nonlinear_mapping:
            assert len(self.lv_dimensions) == len(self.mapping_dimensions)
            assert len(self.mapping_dimensions) == len(self.nys_gamma_values)

    def _fit(self, X, Y):
        for layer_index in range(self.n_layers):
            if self.use_nonlinear_mapping:
                nys_func = Nystroem(kernel='rbf',
                                    gamma=self.nys_gamma_values[layer_index],
                                    n_components=self.mapping_dimensions[layer_index],
                                    n_jobs=-1)
                X_backup = X.clone()
                X = nys_func.fit_transform(X)
                self.mapping_funcs.append(nys_func)
                X = th.tensor(X)
                if self.stack_previous_lv1 and layer_index > 0:
                    lv1_previous_layer = X_backup[:, [0]]
                    X = th.hstack((lv1_previous_layer, X))

            pls = PLS(n_components=self.lv_dimensions[layer_index], solver=self.pls_solver)
            pls.fit(X, Y)
            self.pls_funcs.append(pls)

            latent_variables = pls.x_scores_
            self.latent_variables.append(latent_variables)
            X = latent_variables

    def fit(self, X, Y):
        self._fit(X, Y)
        return self

    def predict(self, test_X):
        Y_pred = None
        for layer_index in range(self.n_layers):
            if self.use_nonlinear_mapping:
                test_X_backup = test_X.clone()
                test_X = self.mapping_funcs[layer_index].transform(test_X)
                test_X = th.tensor(test_X)
                if self.stack_previous_lv1 and layer_index > 0:
                    lv1_previous_layer = test_X_backup[:, [0]]
                    test_X = th.hstack((lv1_previous_layer, test_X))

            if layer_index + 1 == self.n_layers:
                Y_pred = self.pls_funcs[layer_index].predict(test_X)
            test_X = self.pls_funcs[layer_index].transform(test_X)

        return Y_pred

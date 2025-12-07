"""Robust Model Structure Selection (RMSS).

This module implements the RMSS algorithm described in the paper attached in
``RMSS.md``. The method selects model terms using an overall mean absolute
error (OMAE) criterion computed over resampled sub-datasets (leave-one-out by
default). It follows the same interface conventions as other model structure
selection classes (e.g., :class:`~sysidentpy.model_structure_selection.FROLS`),
reusing estimators, basis functions and prediction utilities already provided
by SysIdentPy.

Key points
----------
- Supports all parameter estimators and basis functions available to OFR-based
  classes.
- Uses leave-one-out resampling to score candidate regressors with OMAE (or
  alternative error measures inspired by the paper).
- Keeps output attributes (``final_model``, ``theta``, ``pivv``) compatible
  with equation formatter utilities.

References
----------
- Gu, Y., & Wei, H.-L. "A Robust Model Structure Selection Method for Small
  Sample Size and Multiple Datasets Problems."
"""

from __future__ import annotations

import copy
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

from ..basis_function import Fourier, Polynomial
from ..utils.information_matrix import build_lagged_matrix
from ..utils.check_arrays import num_features
from .ofr_base import OFRBase, apress, get_min_info_value
from ..parameter_estimation.estimators import (
    LeastSquares,
    RidgeRegression,
    RecursiveLeastSquares,
    TotalLeastSquares,
    LeastMeanSquareMixedNorm,
    LeastMeanSquares,
    LeastMeanSquaresFourth,
    LeastMeanSquaresLeaky,
    LeastMeanSquaresNormalizedLeaky,
    LeastMeanSquaresNormalizedSignRegressor,
    LeastMeanSquaresNormalizedSignSign,
    LeastMeanSquaresSignError,
    LeastMeanSquaresSignSign,
    AffineLeastMeanSquares,
    NormalizedLeastMeanSquares,
    NormalizedLeastMeanSquaresSignError,
    LeastMeanSquaresSignRegressor,
)

Estimators = Union[
    LeastSquares,
    RidgeRegression,
    RecursiveLeastSquares,
    TotalLeastSquares,
    LeastMeanSquareMixedNorm,
    LeastMeanSquares,
    LeastMeanSquaresFourth,
    LeastMeanSquaresLeaky,
    LeastMeanSquaresNormalizedLeaky,
    LeastMeanSquaresNormalizedSignRegressor,
    LeastMeanSquaresNormalizedSignSign,
    LeastMeanSquaresSignError,
    LeastMeanSquaresSignSign,
    AffineLeastMeanSquares,
    NormalizedLeastMeanSquares,
    NormalizedLeastMeanSquaresSignError,
    LeastMeanSquaresSignRegressor,
]


class RMSS(OFRBase):
    r"""Robust Model Structure Selection.

    The RMSS algorithm ranks candidate regressors using an overall error metric
    computed over resampled sub-datasets (leave-one-out, as suggested in the
    paper for small-sample problems). At each step it selects the regressor with
    the smallest aggregated error, orthogonalizes the remaining candidates, and
    repeats until the desired number of terms is reached.

    Parameters
    ----------
    ylag : int or list, default=2
        Maximum output lag.
    xlag : int or list, default=2
        Maximum input lag.
    elag : int or list, default=2
        Maximum residue lag (used when estimator requires it).
    order_selection : bool, default=True
        Whether to use information criteria to choose model size.
    info_criteria : {'aic','aicc','bic','fpe','lilc','apress'}, default='apress'
        Information criterion when ``order_selection`` is enabled.
    n_terms : int, optional
        Number of terms to select. Required when ``order_selection`` is False.
    n_info_values : int, default=15
        Maximum number of terms evaluated by the information criterion.
    estimator : Estimators, default=RecursiveLeastSquares()
        Parameter estimator.
    basis_function : Polynomial or Fourier, default=Polynomial()
        Basis function generator.
    model_type : {'NARMAX','NAR','NFIR'}, default='NARMAX'
        Model type.
    eps : float, default=np.finfo(np.float64).eps
        Numerical stability constant.
    alpha : float, default=0
        Regularization parameter (used when estimator is RidgeRegression).
    err_tol : float, optional
        Cumulative ERR/OMAE threshold to stop early.
    resampling : {'loo','bootstrap'}, default='loo'
        Resampling strategy. ``'loo'`` performs leave-one-out as proposed in the
        paper. ``'bootstrap'`` draws ``n_subsets`` bootstrap samples (with
        replacement) of size ``subset_size``.
    error_measure : {'mae','mse','phi3','rmse_ratio'}, default='mae'
        Aggregated error used to rank candidates. ``'mae'`` matches the OMAE in
        the paper. ``'phi3'`` matches the normalized MAE ratio of eq. (19).
        ``'smape'`` is kept as a backward-compatible alias for ``'phi3'``.
    average_theta : bool, default=True
        If True, estimate parameters on every sub-dataset and average the
        resulting coefficients. If False, uses the estimator once on the full
        data (aligned with OFRBase behaviour).
    apress_lambda : float, default=1.0
        Lambda factor used in APRESS (eq. 9). Only used when
        ``info_criteria='apress'``.
    n_subsets : int, optional
        Number of subsets to draw when ``resampling='bootstrap'``. Defaults to
        ``n_samples`` (one subset per leave-one-out equivalent) when not set.
    subset_size : int, optional
        Subset size when ``resampling='bootstrap'``. Defaults to ``n_samples - 1``
        to mimic the sensitivity study in the paper.
    random_state : int, optional
        Seed for bootstrap resampling.
    multi_resampling : bool, default=False
        When multiple datasets are provided, apply the chosen resampling
        strategy to each dataset before scoring candidates (keeps parity with
        the small-sample discussion in the RMSS paper).

    Notes
    -----
    - The implementation follows the same prediction and formatting interfaces
        as other SysIdentPy MSS classes to remain drop-in compatible with
        utilities such as ``equation_formatter``.
    - Setting ``average_theta=False`` skips the per-sub-dataset averaging in
        eq. (28) of the paper; keep it ``True`` for the canonical RMSS behaviour.
    """

    def __init__(
        self,
        *,
        ylag: Union[int, list] = 2,
        xlag: Union[int, list] = 2,
        elag: Union[int, list] = 2,
        order_selection: bool = True,
        info_criteria: str = "apress",
        n_terms: Union[int, None] = None,
        n_info_values: int = 15,
        estimator: Estimators = RecursiveLeastSquares(),
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        model_type: str = "NARMAX",
        eps: np.float64 = np.finfo(np.float64).eps,
        alpha: float = 0,
        err_tol: Optional[float] = None,
        resampling: str = "loo",
        error_measure: str = "mae",
        average_theta: bool = True,
        apress_lambda: float = 1.0,
        n_subsets: Optional[int] = None,
        subset_size: Optional[int] = None,
        random_state: Optional[int] = None,
        multi_resampling: bool = False,
    ):
        self.resampling = resampling
        self.error_measure = error_measure
        self.average_theta = average_theta
        self.n_subsets = n_subsets
        self.subset_size = subset_size
        self.random_state = random_state
        self.multi_resampling = multi_resampling
        self.omae_history: List[np.ndarray] = []
        self._reg_matrices: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []

        super().__init__(
            ylag=ylag,
            xlag=xlag,
            elag=elag,
            order_selection=order_selection,
            info_criteria=info_criteria,
            n_terms=n_terms,
            n_info_values=n_info_values,
            estimator=estimator,
            basis_function=basis_function,
            model_type=model_type,
            eps=eps,
            alpha=alpha,
            err_tol=err_tol,
            apress_lambda=apress_lambda,
        )

        self._validate_rmss_params()

    def _validate_rmss_params(self):
        if self.resampling not in ["loo", "bootstrap"]:
            raise ValueError(f"Unsupported resampling strategy: {self.resampling}")

        if self.error_measure == "smape":
            warnings.warn(
                "error_measure='smape' is deprecated; use 'phi3' (eq. 19) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.error_measure = "phi3"

        valid_measures = {"mae", "mse", "phi3", "rmse_ratio"}
        if self.error_measure not in valid_measures:
            raise ValueError(
                "error_measure must be one of 'mae', 'mse', 'phi3', 'rmse_ratio'. "
                f"Got {self.error_measure}"
            )

        if not isinstance(self.average_theta, bool):
            raise TypeError(
                f"average_theta must be a boolean value. Got {type(self.average_theta)}"
            )

        if self.resampling == "bootstrap":
            if self.n_subsets is not None and self.n_subsets < 1:
                raise ValueError("n_subsets must be a positive integer when provided.")
            if self.subset_size is not None and self.subset_size < 1:
                raise ValueError(
                    "subset_size must be a positive integer when provided."
                )
            if self.random_state is not None and not isinstance(self.random_state, int):
                raise TypeError("random_state must be an integer when provided.")

        if not isinstance(self.multi_resampling, bool):
            raise TypeError(
                f"multi_resampling must be a boolean value. Got {type(self.multi_resampling)}"
            )

    def _create_sub_datasets(
        self, reg_matrix: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate leave-one-out or bootstrap views for a single dataset."""
        if reg_matrix.shape[0] < 2:
            raise ValueError("Need at least two samples to perform RMSS resampling.")

        if self.resampling == "loo":
            n_samples, n_features = reg_matrix.shape
            psi_views = np.empty(
                (n_samples, n_samples - 1, n_features), dtype=np.float64
            )
            y_views = np.empty((n_samples, n_samples - 1), dtype=np.float64)

            for idx in range(n_samples):
                mask = np.ones(n_samples, dtype=bool)
                mask[idx] = False
                psi_views[idx] = reg_matrix[mask]
                y_views[idx] = target[mask, 0]

            return psi_views, y_views

        if self.resampling == "bootstrap":
            rng = np.random.default_rng(self.random_state)
            n_samples, n_features = reg_matrix.shape
            k_subsets = self.n_subsets or n_samples
            subset_size = self.subset_size or max(1, n_samples - 1)

            psi_views = np.empty((k_subsets, subset_size, n_features), dtype=np.float64)
            y_views = np.empty((k_subsets, subset_size), dtype=np.float64)

            for k in range(k_subsets):
                idx = rng.choice(n_samples, size=subset_size, replace=True)
                psi_views[k] = reg_matrix[idx]
                y_views[k] = target[idx, 0]

            return psi_views, y_views

        raise ValueError(f"Unsupported resampling strategy: {self.resampling}")

    def _overall_error(self, psi_views: np.ndarray, y_views: np.ndarray) -> np.ndarray:
        """Compute aggregated error for each candidate across sub-datasets."""
        # psi_views: (K, N', M), y_views: (K, N')
        numerators = np.einsum("knm,kn->km", psi_views, y_views)
        denominators = np.einsum("knm,knm->km", psi_views, psi_views)
        denominators = np.where(np.abs(denominators) < self.eps, self.eps, denominators)
        alphas = numerators / denominators

        preds = psi_views * alphas[:, None, :]
        errors = y_views[:, :, None] - preds

        if self.error_measure == "mae":
            metric = np.abs(errors).mean(axis=1)
        elif self.error_measure == "mse":
            metric = np.square(errors).mean(axis=1)
        elif self.error_measure == "phi3":
            numerator = np.abs(errors).sum(axis=1)
            denom = np.abs(y_views).sum(axis=1)[:, None] + np.abs(preds).sum(axis=1)
            denom = np.where(np.abs(denom) < self.eps, self.eps, denom)
            metric = numerator / denom
        else:  # rmse_ratio
            rmse = np.sqrt(np.square(errors).mean(axis=1))
            denom = np.sqrt(np.square(y_views).mean(axis=1))[:, None] + np.sqrt(
                np.square(preds).mean(axis=1)
            )
            denom = np.where(np.abs(denom) < self.eps, self.eps, denom)
            metric = rmse / denom

        return metric.mean(axis=0)

    def _overall_error_multi(
        self, psi_list: List[np.ndarray], y_list: List[np.ndarray]
    ) -> np.ndarray:
        """Compute aggregated error across multiple datasets using simple mean."""
        per_dataset = []
        for psi_k, y_k in zip(psi_list, y_list):
            if psi_k.ndim == 3:
                # Resampled views (K_k, N', M)
                metric = self._overall_error(psi_k, y_k)
            else:
                y_vec = y_k.reshape(-1)
                numerators = psi_k.T @ y_vec
                denominators = np.einsum("ij,ij->j", psi_k, psi_k)
                denominators = np.where(
                    np.abs(denominators) < self.eps, self.eps, denominators
                )
                alphas = numerators / denominators

                preds = psi_k * alphas[None, :]
                errors = y_vec[:, None] - preds

                if self.error_measure == "mae":
                    metric = np.abs(errors).mean(axis=0)
                elif self.error_measure == "mse":
                    metric = np.square(errors).mean(axis=0)
                elif self.error_measure == "phi3":
                    numerator = np.abs(errors).sum(axis=0)
                    denom = np.abs(y_vec).sum(axis=0) + np.abs(preds).sum(axis=0)
                    denom = np.where(np.abs(denom) < self.eps, self.eps, denom)
                    metric = numerator / denom
                else:  # rmse_ratio
                    rmse = np.sqrt(np.square(errors).mean(axis=0))
                    denom = np.sqrt(np.square(y_k).mean(axis=0)) + np.sqrt(
                        np.square(preds).mean(axis=0)
                    )
                    denom = np.where(np.abs(denom) < self.eps, self.eps, denom)
                    metric = rmse / denom

            per_dataset.append(metric)

        stacked = np.stack(per_dataset, axis=0)
        return np.mean(stacked, axis=0)

    def _orthogonalize_remaining_views(
        self, psi_views: np.ndarray, selected_q: np.ndarray
    ) -> np.ndarray:
        """Orthogonalize remaining candidates for resampled views (K, N', M)."""
        denom = np.einsum("kn,kn->k", selected_q, selected_q)
        denom = np.where(np.abs(denom) < self.eps, self.eps, denom)

        projection = np.einsum("kn,knm->km", selected_q, psi_views)
        coeff = projection / denom[:, None]
        return psi_views - selected_q[:, :, None] * coeff[:, None, :]

    def _orthogonalize_remaining(
        self, psi_views: np.ndarray, selected_q: np.ndarray
    ) -> np.ndarray:
        """Orthogonalize remaining candidates against the selected vector."""
        denom = np.einsum("kn,kn->k", selected_q, selected_q)
        denom = np.where(np.abs(denom) < self.eps, self.eps, denom)

        projection = np.einsum("kn,knm->km", selected_q, psi_views)
        coeff = projection / denom[:, None]
        psi_views = psi_views - selected_q[:, :, None] * coeff[:, None, :]
        return psi_views

    def _orthogonalize_remaining_multi(
        self, psi_list: List[np.ndarray], selected_q_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Orthogonalize remaining candidates for multiple datasets."""
        updated = []
        for psi_k, q_k in zip(psi_list, selected_q_list):
            denom = np.dot(q_k, q_k)
            denom = self.eps if np.abs(denom) < self.eps else denom
            projection = psi_k.T @ q_k
            coeff = projection / denom
            updated.append(psi_k - np.outer(q_k, coeff))
        return updated

    def _prepare_datasets(
        self,
        X: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]],
        y: Union[np.ndarray, List[np.ndarray]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Build regressor matrices/targets for single or multiple datasets."""
        if isinstance(y, (list, tuple)):
            y_list = list(y)
            if X is None or isinstance(X, np.ndarray):
                X_list = [X for _ in y_list]
            else:
                X_list = list(X)
            if len(X_list) != len(y_list):
                raise ValueError("X and y lists must have the same length.")

            reg_matrices: List[np.ndarray] = []
            targets: List[np.ndarray] = []
            self.n_inputs = None

            for Xi, yi in zip(X_list, y_list):
                lagged_data = build_lagged_matrix(
                    Xi, yi, self.xlag, self.ylag, self.model_type
                )
                reg_matrix = self.basis_function.fit(
                    lagged_data,
                    self.max_lag,
                    self.ylag,
                    self.xlag,
                    self.model_type,
                    predefined_regressors=None,
                )

                target = self._default_estimation_target(yi)
                reg_matrices.append(reg_matrix)
                targets.append(target)

                if self.n_inputs is None:
                    self.n_inputs = num_features(Xi) if Xi is not None else 1
                elif self.n_inputs != (num_features(Xi) if Xi is not None else 1):
                    raise ValueError(
                        "All datasets must share the same input dimension."
                    )

            n_features = {rm.shape[1] for rm in reg_matrices}
            if len(n_features) != 1:
                raise ValueError("All datasets must produce the same regressor space.")

            return reg_matrices, targets

        # Single dataset path
        lagged_data = build_lagged_matrix(X, y, self.xlag, self.ylag, self.model_type)
        reg_matrix = self.basis_function.fit(
            lagged_data,
            self.max_lag,
            self.ylag,
            self.xlag,
            self.model_type,
            predefined_regressors=None,
        )

        if X is not None:
            self.n_inputs = num_features(X)
        else:
            self.n_inputs = 1

        target = self._default_estimation_target(y)
        return [reg_matrix], [target]

    def run_mss_algorithm(
        self,
        psi: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
        process_term_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform RMSS selection over single or multiple datasets."""
        self.omae_history = []

        if not isinstance(psi, list):
            reg_matrices = [psi]
            targets = [y]
        else:
            reg_matrices = psi
            targets = y if isinstance(y, list) else [y]

        if len(reg_matrices) == 1:
            psi = reg_matrices[0]
            target = targets[0]
            psi_views, y_views = self._create_sub_datasets(psi, target)

            available_indices = np.arange(psi.shape[1])
            selected_indices = []
            err_trace = []

            current_views = psi_views
            for _ in range(min(process_term_number, psi.shape[1])):
                omae = self._overall_error(current_views, y_views)
                self.omae_history.append(omae)
                best_local_idx = int(np.argmin(omae))
                selected_indices.append(int(available_indices[best_local_idx]))
                err_trace.append(float(omae[best_local_idx]))

                if (self.err_tol is not None) and (np.sum(err_trace) >= self.err_tol):
                    break

                selected_q = current_views[:, :, best_local_idx]
                available_indices = np.delete(available_indices, best_local_idx)
                current_views = np.delete(current_views, best_local_idx, axis=2)

                if current_views.shape[2] == 0:
                    break
                current_views = self._orthogonalize_remaining(current_views, selected_q)

            piv = np.array(selected_indices, dtype=int)
            err = np.array(err_trace, dtype=float)
            psi_selected = psi[:, piv] if piv.size else psi[:, :0]
            return err, piv, psi_selected, target

        # Multiple independent datasets path
        psi_list: List[np.ndarray] = []
        target_list: List[np.ndarray] = []
        for rm, tgt in zip(reg_matrices, targets):
            if self.multi_resampling:
                views, yv = self._create_sub_datasets(rm, tgt)
                psi_list.append(views)
                target_list.append(yv)
            else:
                psi_list.append(rm.copy())
                target_list.append(tgt)

        available_indices = np.arange(psi_list[0].shape[-1])
        selected_indices: List[int] = []
        err_trace: List[float] = []

        current_views = psi_list
        for _ in range(min(process_term_number, psi_list[0].shape[-1])):
            omae = self._overall_error_multi(current_views, target_list)
            self.omae_history.append(omae)
            best_local_idx = int(np.argmin(omae))
            selected_indices.append(int(available_indices[best_local_idx]))
            err_trace.append(float(omae[best_local_idx]))

            if (self.err_tol is not None) and (np.sum(err_trace) >= self.err_tol):
                break

            selected_q_list = []
            updated_views = []

            for view in current_views:
                if view.ndim == 3:
                    selected_q_list.append(view[:, :, best_local_idx])
                    view_reduced = np.delete(view, best_local_idx, axis=2)
                    updated_views.append(view_reduced)
                else:
                    selected_q_list.append(view[:, best_local_idx])
                    updated_views.append(np.delete(view, best_local_idx, axis=1))

            available_indices = np.delete(available_indices, best_local_idx)

            if updated_views[0].shape[-1] == 0:
                break

            new_views = []
            for view, q in zip(updated_views, selected_q_list):
                if view.ndim == 3:
                    new_views.append(self._orthogonalize_remaining_views(view, q))
                else:
                    new_views.append(
                        self._orthogonalize_remaining_multi([view], [q])[0]
                    )

            current_views = new_views

        piv = np.array(selected_indices, dtype=int)
        err = np.array(err_trace, dtype=float)
        psi_selected = reg_matrices[0][:, piv] if piv.size else reg_matrices[0][:, :0]
        return err, piv, psi_selected, targets[0]

    def _estimate_theta(
        self,
        reg_matrices: List[np.ndarray],
        targets: List[np.ndarray],
        piv: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        piv = self.pivv if piv is None else piv
        if piv is None or len(piv) == 0:
            return np.empty((0, 1))

        def _select_columns(mat: np.ndarray) -> np.ndarray:
            return mat[:, piv]

        if len(reg_matrices) == 1:
            psi = _select_columns(reg_matrices[0])
            target = targets[0]
            if not self.average_theta:
                warnings.warn(
                    "average_theta=False skips the per-subset averaging in eq.(28) "
                    "of the RMSS paper; use True to match the reference method.",
                    UserWarning,
                    stacklevel=2
                )
                theta = self.estimator.optimize(psi, target)
            else:
                psi_views, y_views = self._create_sub_datasets(psi, target)
                thetas = []
                for k in range(psi_views.shape[0]):
                    est_copy = copy.deepcopy(self.estimator)
                    theta_k = est_copy.optimize(psi_views[k], y_views[k].reshape(-1, 1))
                    thetas.append(theta_k.reshape(-1, 1))
                theta = np.mean(np.stack(thetas, axis=2), axis=2)

            if getattr(self.estimator, "unbiased", False) is True:
                theta = self.estimator.unbiased_estimator(
                    psi,
                    target,
                    theta,
                    self.elag,
                    self.max_lag,
                    self.estimator,
                    self.basis_function,
                    self.estimator.uiter,
                )
            return theta

        # Multiple datasets: average parameters across datasets (eq. 28)
        if getattr(self.estimator, "unbiased", False) is True:
            warnings.warn(
                "Unbiased correction is not applied when fitting multiple datasets "
                "with RMSS; results may differ from single-dataset unbiased fits.",
                UserWarning,
                stacklevel=2
            )
        thetas = []
        for reg_matrix, target in zip(reg_matrices, targets):
            psi_sel = _select_columns(reg_matrix)
            est_copy = copy.deepcopy(self.estimator)
            theta_k = est_copy.optimize(psi_sel, target)
            thetas.append(theta_k.reshape(-1, 1))

        theta = np.mean(np.stack(thetas, axis=2), axis=2)
        return theta

    def information_criterion(
        self,
        x: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """Compute information criteria using robust parameter estimation."""
        reg_matrices = x if isinstance(x, list) else [x]
        targets = y if isinstance(y, list) else [y]

        if (
            self.n_info_values is not None
            and self.n_info_values > reg_matrices[0].shape[1]
        ):
            self.n_info_values = reg_matrices[0].shape[1]

        output_vector = np.zeros(self.n_info_values)
        output_vector[:] = np.nan

        for i in range(self.n_info_values):
            n_theta = i + 1
            _, piv, _, _ = self.run_mss_algorithm(reg_matrices, targets, n_theta)

            tmp_theta = self._estimate_theta(reg_matrices, targets, piv)

            if len(reg_matrices) == 1:
                psi_sel = reg_matrices[0][:, piv]
                target_sel = targets[0]
                tmp_yhat = np.dot(psi_sel, tmp_theta)
                tmp_residual = target_sel - tmp_yhat

                if self.info_criteria == "apress":
                    mse = np.mean(np.square(tmp_residual))
                    output_vector[i] = apress(
                        n_theta, target_sel.shape[0], mse, self.apress_lambda
                    )
                else:
                    e_var = np.var(tmp_residual, ddof=1)
                    output_vector[i] = self.info_criteria_function(
                        n_theta, target_sel.shape[0], e_var
                    )
            else:
                per_dataset_vals = []
                for rm, tgt in zip(reg_matrices, targets):
                    psi_sel = rm[:, piv]
                    yhat = np.dot(psi_sel, tmp_theta)
                    residual = tgt - yhat

                    if self.info_criteria == "apress":
                        mse = np.mean(np.square(residual))
                        val = apress(n_theta, tgt.shape[0], mse, self.apress_lambda)
                    else:
                        e_var = np.var(residual, ddof=1)
                        val = self.info_criteria_function(n_theta, tgt.shape[0], e_var)
                    per_dataset_vals.append(val)

                output_vector[i] = float(np.mean(per_dataset_vals))

            if i == self.n_info_values - 1:
                self.pivv = piv

        return output_vector

    def fit(
        self, *, X: Optional[np.ndarray] = None, y: Union[np.ndarray, List[np.ndarray]]
    ):
        if y is None:
            raise ValueError("y cannot be None")

        self.max_lag = self._get_max_lag()

        reg_matrices, targets = self._prepare_datasets(X, y)
        self._reg_matrices = reg_matrices
        self._targets = targets

        self.regressor_code = self.regressor_space(self.n_inputs)

        if self.order_selection is True:
            self.info_values = self.information_criterion(reg_matrices, targets)

        if self.n_terms is None and self.order_selection is True:
            if self.info_criteria == "apress":
                model_length = int(np.nanargmin(self.info_values)) + 1
            else:
                model_length = get_min_info_value(self.info_values)
            self.n_terms = model_length
        elif self.n_terms is None and self.order_selection is not True:
            raise ValueError(
                "If order_selection is False, you must define n_terms value."
            )
        else:
            model_length = self.n_terms

        mss_result = self.run_mss_algorithm(reg_matrices, targets, model_length)
        self.err, self.pivv, _psi, _estimation_target = self._unpack_mss_output(
            mss_result, targets[0]
        )

        model_length = min(model_length, len(self.pivv))
        self.n_terms = model_length

        tmp_piv = self.pivv[0:model_length]
        repetition = len(reg_matrices[0])
        if isinstance(self.basis_function, Polynomial):
            self.final_model = self.regressor_code[tmp_piv, :].copy()
        else:
            self.regressor_code = np.sort(
                np.tile(self.regressor_code[1:, :], (repetition, 1)),
                axis=0,
            )
            self.final_model = self.regressor_code[tmp_piv, :].copy()

        self.theta = self._estimate_theta(self._reg_matrices, self._targets)
        return self

    def predict(
        self,
        *,
        X: Optional[np.ndarray] = None,
        y: np.ndarray,
        steps_ahead: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
    ) -> np.ndarray:
        return super().predict(
            X=X, y=y, steps_ahead=steps_ahead, forecast_horizon=forecast_horizon
        )

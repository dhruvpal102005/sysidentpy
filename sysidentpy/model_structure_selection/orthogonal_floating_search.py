"""Orthogonal floating search algorithms (OSF, OIF, OOS/O2S).

The algorithms implemented here follow the Orthogonal Floating Search
framework described in the attached OFSA manuscript. They adapt
well-known floating feature selection strategies to the NARX model
structure selection problem by combining: (i) orthogonal projections of
candidate regressors and (ii) the classical Error Reduction Ratio (ERR)
criterion. All classes are drop-in compatible with the existing SysIdentPy
API (estimators, basis functions, equation formatter, etc.).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union
from itertools import combinations

import numpy as np

from ..basis_function import Fourier, Polynomial
from ..narmax_base import house, rowhouse
from ..parameter_estimation.estimators import RecursiveLeastSquares
from .ofr_base import Estimators, OFRBase, _compute_err_slice


class _OrthogonalFloatingBase(OFRBase):
    """Shared helpers for the Orthogonal Floating Search family."""

    def __init__(
        self,
        *,
        ylag: Union[int, List[int]] = 2,
        xlag: Union[int, List[int]] = 2,
        elag: Union[int, List[int]] = 2,
        order_selection: bool = True,
        info_criteria: str = "aic",
        n_terms: Optional[int] = None,
        n_info_values: int = 15,
        estimator: Estimators = RecursiveLeastSquares(),
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        model_type: str = "NARMAX",
        eps: float = np.finfo(np.float64).eps,
        alpha: float = 0.0,
        err_tol: Optional[float] = None,
        apress_lambda: float = 1.0,
    ):
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

    # --------- low-level ERR utilities ---------
    def _subset_err(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        squared_y: float,
    ) -> Tuple[float, np.ndarray]:
        """Compute ERR score for a given subset (Eq. 4–5).

        The ERR of each selected regressor is computed in an orthonormal
        basis obtained via QR. The returned score is the sum of ERRs,
        matching J(·) in Definition 5.
        """

        if not subset:
            return 0.0, np.array([])

        psi_sel = psi[:, subset]
        # Reduced QR gives orthonormal columns (||w_i||=1), aligning with
        # the Gram-Schmidt-based ERR definition without explicit scaling.
        q, _ = np.linalg.qr(psi_sel, mode="reduced")
        g = q.T @ target
        denom = squared_y if squared_y > 0 else np.finfo(np.float64).eps
        err_vals = np.square(g).flatten() / denom
        score = float(np.sum(err_vals))
        return score, err_vals

    def _compute_squared_y(self, target: np.ndarray) -> float:
        """Compute ||y||^2 with numerical floor to avoid division by zero."""

        squared_y = float(np.dot(target.T, target).item())
        return squared_y if squared_y > self.eps else float(self.eps)

    # --------- local search building blocks ---------
    def _best_addition(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        available: List[int],
        squared_y: float,
    ) -> Tuple[Optional[int], float]:
        """Pick the most significant term (Definition 1)."""

        best_idx: Optional[int] = None
        best_score = -np.inf
        for idx in available:
            score, _ = self._subset_err(psi, target, subset + [idx], squared_y)
            if score > best_score or (
                np.isclose(score, best_score) and (best_idx is None or idx < best_idx)
            ):
                best_score = score
                best_idx = idx
        return best_idx, best_score

    def _best_removal(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        squared_y: float,
    ) -> Tuple[int, float]:
        """Pick the least significant term (Definition 2)."""

        best_idx = subset[0]
        best_score = -np.inf
        for idx in subset:
            remaining = [t for t in subset if t != idx]
            score, _ = self._subset_err(psi, target, remaining, squared_y)
            if score > best_score or (np.isclose(score, best_score) and idx < best_idx):
                best_score = score
                best_idx = idx
        return best_idx, best_score

    def _most_significant_terms(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        available: List[int],
        count: int,
        squared_y: float,
    ) -> List[int]:
        """Exhaustive selection of the most significant ``count``-subset (Definition 3)."""

        k = min(count, len(available))
        if k <= 0:
            return []

        best_score = -np.inf
        best_combo: Optional[Tuple[int, ...]] = None

        for combo in combinations(available, k):
            score, _ = self._subset_err(psi, target, subset + list(combo), squared_y)
            if score > best_score or (
                np.isclose(score, best_score)
                and (best_combo is None or combo < best_combo)
            ):
                best_score = score
                best_combo = combo

        return list(best_combo) if best_combo is not None else []

    def _least_significant_terms(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        count: int,
        squared_y: float,
    ) -> List[int]:
        """Exhaustive removal of the least significant ``count``-subset (Definition 4)."""

        k = min(count, len(subset))
        if k <= 0:
            return []

        best_score = -np.inf
        best_combo: Optional[Tuple[int, ...]] = None

        for combo in combinations(subset, k):
            candidate_subset = [t for t in subset if t not in combo]
            score, _ = self._subset_err(psi, target, candidate_subset, squared_y)
            if score > best_score or (
                np.isclose(score, best_score)
                and (best_combo is None or combo < best_combo)
            ):
                best_score = score
                best_combo = combo

        return list(best_combo) if best_combo is not None else []

    def _select_most_significant_subset(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        base_subset: List[int],
        available: List[int],
        count: int,
        squared_y: float,
    ) -> List[int]:
        """Floating forward search (OSF-style) to pick ``count`` terms.

        This mirrors the bottom-up floating strategy but constrains the search
        to adding terms on top of a fixed ``base_subset``. Backtracking only
        removes terms that were added in this routine, never those in
        ``base_subset``.
        """

        if count <= 0 or not available:
            return []

        _, base_score_arr = self._subset_err(psi, target, base_subset, squared_y)
        base_score = float(np.sum(base_score_arr))

        best_by_size = {0: (base_score, [])}
        selected: List[int] = []
        current_score = base_score
        last_added: Optional[int] = None

        def backtrack(
            selected_terms: List[int],
            score: float,
            last_added_term: Optional[int],
        ) -> Tuple[List[int], float]:
            flag_first_removal = 1
            while (
                len(base_subset) + len(selected_terms) > 2 and len(selected_terms) > 0
            ):
                full_subset = base_subset + selected_terms
                ls_idx, ls_score = self._best_removal(
                    psi, target, full_subset, squared_y
                )
                # Do not remove any term that belongs to the fixed base subset.
                if ls_idx in base_subset:
                    break

                prev_best_score = best_by_size.get(
                    len(selected_terms) - 1, (-np.inf, [])
                )[0]
                if (flag_first_removal == 1 and ls_idx == last_added_term) or (
                    ls_score <= prev_best_score
                ):
                    break

                selected_terms = [t for t in selected_terms if t != ls_idx]
                score = ls_score
                if ls_score > prev_best_score:
                    best_by_size[len(selected_terms)] = (
                        ls_score,
                        selected_terms.copy(),
                    )

                flag_first_removal = 0

            return selected_terms, score

        while len(selected) < count and len(available) > 0:
            ms_idx, _ = self._best_addition(
                psi, target, base_subset + selected, available, squared_y
            )
            if ms_idx is None:
                break

            candidate_selected = selected + [ms_idx]
            candidate_score, _ = self._subset_err(
                psi, target, base_subset + candidate_selected, squared_y
            )

            stored_score, stored_subset = best_by_size.get(
                len(candidate_selected), (-np.inf, [])
            )

            if candidate_score > stored_score:
                selected = candidate_selected
                current_score = candidate_score
                best_by_size[len(selected)] = (current_score, selected.copy())
                last_added = ms_idx
            else:
                selected = stored_subset.copy()
                current_score = stored_score
                last_added = ms_idx

            available = [a for a in available if a not in selected]
            selected, current_score = backtrack(selected, current_score, last_added)

        return selected

    def _select_least_significant_subset(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        base_subset: List[int],
        count: int,
        squared_y: float,
    ) -> List[int]:
        """Sequential backward floating-style removal of ``count`` terms.

        Uses repeated evaluation of the removal that maximizes the resulting
        criterion. Backtracking avoids undoing the first removal when it is the
        most recent change, mirroring the OSF safeguard.
        """

        if count <= 0 or len(base_subset) == 0:
            return []

        best_by_size = {
            len(base_subset): self._subset_err(psi, target, base_subset, squared_y)
        }
        removed: List[int] = []
        current_score = best_by_size[len(base_subset)][0]
        last_removed: Optional[int] = None

        while len(removed) < count and (len(base_subset) - len(removed)) > 0:
            working_subset = [t for t in base_subset if t not in removed]
            ls_idx, _ = self._best_removal(psi, target, working_subset, squared_y)
            candidate_removed = removed + [ls_idx]
            candidate_subset = [t for t in base_subset if t not in candidate_removed]
            candidate_score, _ = self._subset_err(
                psi, target, candidate_subset, squared_y
            )

            stored_score, stored_removed = best_by_size.get(
                len(candidate_subset), (-np.inf, [])
            )

            if candidate_score > stored_score:
                removed = candidate_removed
                current_score = candidate_score
                best_by_size[len(candidate_subset)] = (current_score, removed.copy())
                last_removed = ls_idx
            else:
                removed = stored_removed.copy()
                current_score = stored_score
                last_removed = ls_idx

            # Backtrack: avoid immediately re-removing the last removed term.
            flag_first_removal = 1
            while len(base_subset) - len(removed) > 2:
                working_subset = [t for t in base_subset if t not in removed]
                next_idx, next_score = self._best_removal(
                    psi, target, working_subset, squared_y
                )
                prev_best_score = best_by_size.get(
                    len(working_subset) - 1, (-np.inf, [])
                )[0]
                if (flag_first_removal == 1 and next_idx == last_removed) or (
                    next_score <= prev_best_score
                ):
                    break

                removed.append(next_idx)
                current_score = next_score
                best_by_size[len(working_subset) - 1] = (
                    current_score,
                    removed.copy(),
                )
                flag_first_removal = 0

            if len(removed) >= count:
                break

        return removed

    def _backtrack(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        current_score: float,
        best_by_size: dict,
        squared_y: float,
        last_added: Optional[int],
    ) -> Tuple[List[int], float]:
        """Adaptive backward step used by OSF/OIF."""

        flag_first_removal = 1
        while len(subset) > 2:
            ls_idx, ls_score = self._best_removal(psi, target, subset, squared_y)
            prev_best_score = best_by_size.get(len(subset) - 1, (-np.inf, []))[0]
            if (flag_first_removal == 1 and ls_idx == last_added) or (
                ls_score <= prev_best_score
            ):
                break

            subset = [t for t in subset if t != ls_idx]
            current_score = ls_score
            if ls_score > prev_best_score:
                best_by_size[len(subset)] = (ls_score, subset.copy())

            flag_first_removal = 0

        return subset, current_score


class OSF(_OrthogonalFloatingBase):
    """Orthogonal Sequential Floating search (paper Section 3.2)."""

    def run_mss_algorithm(
        self, psi: np.ndarray, y: np.ndarray, process_term_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target = self._default_estimation_target(y)
        squared_y = self._compute_squared_y(target)

        total_terms = psi.shape[1]
        all_indices = list(range(total_terms))

        best_by_size = {0: (0.0, [])}
        subset: List[int] = []
        current_score = 0.0
        last_added: Optional[int] = None

        while len(subset) < process_term_number and len(subset) < total_terms:
            available = [idx for idx in all_indices if idx not in subset]
            if not available:
                break

            ms_idx, _ = self._best_addition(psi, target, subset, available, squared_y)
            if ms_idx is None:
                break

            candidate_subset = subset + [ms_idx]
            candidate_score, _ = self._subset_err(
                psi, target, candidate_subset, squared_y
            )

            stored_score, stored_subset = best_by_size.get(
                len(candidate_subset), (-np.inf, [])
            )

            if candidate_score > stored_score:
                subset = candidate_subset
                current_score = candidate_score
                best_by_size[len(subset)] = (current_score, subset.copy())
                last_added = ms_idx
            else:
                subset = stored_subset.copy()
                current_score = stored_score
                last_added = ms_idx

            subset, current_score = self._backtrack(
                psi, target, subset, current_score, best_by_size, squared_y, last_added
            )

        _, err_vals = self._subset_err(psi, target, subset, squared_y)
        piv = np.array(subset, dtype=int)
        psi_selected = psi[:, piv] if len(piv) else psi[:, :0]
        return err_vals, piv, psi_selected, target


class OIF(OSF):
    """Orthogonal Improved Floating search (paper Section 3.3)."""

    def run_mss_algorithm(
        self, psi: np.ndarray, y: np.ndarray, process_term_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target = self._default_estimation_target(y)
        squared_y = self._compute_squared_y(target)
        total_terms = psi.shape[1]
        all_indices = list(range(total_terms))

        best_by_size = {0: (0.0, [])}
        subset: List[int] = []
        current_score = 0.0
        last_added: Optional[int] = None

        while len(subset) < process_term_number and len(subset) < total_terms:
            available = [idx for idx in all_indices if idx not in subset]
            if not available:
                break

            ms_idx, _ = self._best_addition(psi, target, subset, available, squared_y)
            if ms_idx is None:
                break

            candidate_subset = subset + [ms_idx]
            candidate_score, _ = self._subset_err(
                psi, target, candidate_subset, squared_y
            )

            stored_score, stored_subset = best_by_size.get(
                len(candidate_subset), (-np.inf, [])
            )

            if candidate_score > stored_score:
                subset = candidate_subset
                current_score = candidate_score
                best_by_size[len(subset)] = (current_score, subset.copy())
                last_added = ms_idx
            else:
                subset = stored_subset.copy()
                current_score = stored_score
                last_added = ms_idx

            subset, current_score = self._backtrack(
                psi, target, subset, current_score, best_by_size, squared_y, last_added
            )

            # Swapping step: replace one weak term by the most significant
            # available term (Definition 3 based swap).
            subset, current_score, swap_added = self._swap_step(
                psi,
                target,
                subset,
                current_score,
                best_by_size,
                all_indices,
                squared_y,
            )

            if swap_added is not None:
                subset, current_score = self._backtrack(
                    psi,
                    target,
                    subset,
                    current_score,
                    best_by_size,
                    squared_y,
                    swap_added,
                )

        _, err_vals = self._subset_err(psi, target, subset, squared_y)
        piv = np.array(subset, dtype=int)
        psi_selected = psi[:, piv] if len(piv) else psi[:, :0]
        return err_vals, piv, psi_selected, target

    def _swap_step(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        current_score: float,
        best_by_size: dict,
        all_indices: List[int],
        squared_y: float,
    ) -> Tuple[List[int], float, Optional[int]]:
        best_subset = subset.copy()
        best_score = current_score
        best_added: Optional[int] = None

        for idx in subset:
            reduced = [t for t in subset if t != idx]
            available = [i for i in all_indices if i not in reduced]
            ms_idx, _ = self._best_addition(psi, target, reduced, available, squared_y)
            if ms_idx is None or ms_idx in reduced:
                continue

            candidate_subset = reduced + [ms_idx]
            candidate_score, _ = self._subset_err(
                psi, target, candidate_subset, squared_y
            )

            if candidate_score > best_score:
                best_score = candidate_score
                best_subset = candidate_subset
                best_added = ms_idx

        if best_score > current_score:
            best_by_size[len(best_subset)] = (best_score, best_subset.copy())
            return best_subset, best_score, best_added

        return subset, current_score, None


class OOS(_OrthogonalFloatingBase):
    """Orthogonal Oscillating Search (paper Section 3.4).

    This class is named ``OOS`` (Orthogonal Oscillating Search) to avoid
    the caret notation ``O^2S`` in code. It corresponds to the same method
    described as O²S in the paper.
    """

    def __init__(
        self,
        *,
        ylag: Union[int, List[int]] = 2,
        xlag: Union[int, List[int]] = 2,
        elag: Union[int, List[int]] = 2,
        order_selection: bool = True,
        info_criteria: str = "aic",
        n_terms: Optional[int] = None,
        n_info_values: int = 15,
        estimator: Estimators = RecursiveLeastSquares(),
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        model_type: str = "NARMAX",
        eps: float = np.finfo(np.float64).eps,
        alpha: float = 0.0,
        err_tol: Optional[float] = None,
        apress_lambda: float = 1.0,
        max_search_depth: Optional[int] = None,
    ):
        self.max_search_depth = max_search_depth
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
        self._validate_oos_params()

    def _validate_oos_params(self) -> None:
        if self.max_search_depth is None:
            return
        if not isinstance(self.max_search_depth, int) or self.max_search_depth < 1:
            raise ValueError(
                f"max_search_depth must be integer and > zero. Got {self.max_search_depth}"
            )

    def _resolve_search_depth(self, process_term_number: int, total_terms: int) -> int:
        """Choose search depth per OFSA guideline (25% of the smaller side)."""

        if self.max_search_depth is not None:
            return self.max_search_depth

        smaller_side = max(
            1, min(process_term_number, max(0, total_terms - process_term_number))
        )
        depth = int(np.floor(0.25 * smaller_side))
        return max(1, depth)

    def run_mss_algorithm(
        self, psi: np.ndarray, y: np.ndarray, process_term_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target = self._default_estimation_target(y)
        squared_y = self._compute_squared_y(target)
        total_terms = psi.shape[1]
        all_indices = list(range(total_terms))

        # Resolve depth once we know xi and n, as suggested by the paper.
        max_depth = self._resolve_search_depth(process_term_number, total_terms)

        # Initial model: greedy inclusion of ``process_term_number`` terms.
        subset: List[int] = []
        available = all_indices.copy()
        for _ in range(min(process_term_number, total_terms)):
            ms_idx, _ = self._best_addition(psi, target, subset, available, squared_y)
            if ms_idx is None:
                break
            subset.append(ms_idx)
            available.remove(ms_idx)

        current_score, _ = self._subset_err(psi, target, subset, squared_y)

        o = 1
        # Flags f1/f2 track consecutive failures of down/up swings as in Alg. 2.
        f1 = 0
        f2 = 0

        while o <= max_depth and len(subset) > 0:
            improvement = False

            # Down swing: remove exactly o least significant, then add o most significant.
            if len(subset) >= o:
                down_subset = subset.copy()
                to_remove = self._select_least_significant_subset(
                    psi, target, down_subset, o, squared_y
                )

                if len(to_remove) == o:
                    down_subset = [t for t in down_subset if t not in to_remove]

                    available_down = [
                        idx for idx in all_indices if idx not in down_subset
                    ]
                    if len(available_down) >= o:
                        ms_terms = self._select_most_significant_subset(
                            psi,
                            target,
                            down_subset,
                            available_down,
                            o,
                            squared_y,
                        )
                    else:
                        ms_terms = []

                    if len(ms_terms) == o:
                        down_subset = down_subset + ms_terms
                        down_score, _ = self._subset_err(
                            psi, target, down_subset, squared_y
                        )

                        if down_score > current_score:
                            subset = down_subset
                            current_score = down_score
                            improvement = True
                            f1 = 0
                        else:
                            f1 = 1
                    else:
                        f1 = 1
                else:
                    f1 = 1
            else:
                f1 = 1

            # If both down and previous up swings failed, increase depth before the up swing.
            if f1 == 1 and f2 == 1:
                o += 1
                f1 = 0
                f2 = 0
                if o > max_depth:
                    break

            # Up swing: add o most significant, then remove o least significant.
            if len(subset) + o <= total_terms:
                up_subset = subset.copy()
                available_up = [idx for idx in all_indices if idx not in up_subset]
                if len(available_up) >= o:
                    ms_terms_up = self._select_most_significant_subset(
                        psi,
                        target,
                        up_subset,
                        available_up,
                        o,
                        squared_y,
                    )
                else:
                    ms_terms_up = []

                if len(ms_terms_up) == o:
                    up_subset = up_subset + ms_terms_up

                    to_remove_up = self._select_least_significant_subset(
                        psi, target, up_subset, o, squared_y
                    )
                    if len(to_remove_up) == o:
                        up_subset = [t for t in up_subset if t not in to_remove_up]
                        up_score, _ = self._subset_err(
                            psi, target, up_subset, squared_y
                        )

                        if up_score > current_score:
                            subset = up_subset
                            current_score = up_score
                            improvement = True
                            f2 = 0
                        else:
                            f2 = 1
                    else:
                        f2 = 1
                else:
                    f2 = 1
            else:
                f2 = 1

            if improvement:
                o = 1
                f1 = 0
                f2 = 0
            else:
                if f1 == 1 and f2 == 1:
                    o += 1
                    f1 = 0
                    f2 = 0
                else:
                    # Alternate swings but keep depth.
                    pass

        _, err_vals = self._subset_err(psi, target, subset, squared_y)
        piv = np.array(subset, dtype=int)
        psi_selected = psi[:, piv] if len(piv) else psi[:, :0]
        return err_vals, piv, psi_selected, target


# Alias matching the notation O²S used in the paper.
O2S = OOS

__all__ = ["OSF", "OIF", "OOS", "O2S"]

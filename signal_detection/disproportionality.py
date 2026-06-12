# -*- coding: utf-8 -*-
"""
signal_detection/disproportionality.py
=======================================
Day 7 target: Implement PRR, ROR, and EBGM calculations here.

Stubs only -- import-safe placeholder so the package is valid from Day 1.
"""


def compute_prr(n_drug_event: int, n_drug: int, n_event: int, n_total: int) -> float:
    """
    Proportional Reporting Ratio.
    PRR = (n_DE / n_D) / (n_E / n_total)

    Raises ValueError if inputs are zero (caller must handle sparse pairs).
    This is a stub -- full implementation on Day 7.
    """
    raise NotImplementedError("Implement on Day 7")


def compute_ror(n_drug_event: int, n_drug: int, n_event: int, n_total: int) -> float:
    """
    Reporting Odds Ratio.
    ROR = (n_DE * (n_total - n_D - n_E + n_DE)) / ((n_D - n_DE) * (n_E - n_DE))

    Stub -- full implementation on Day 7.
    """
    raise NotImplementedError("Implement on Day 7")


def compute_ebgm(n_drug_event: int, expected: float, prior_alpha: float = 0.5) -> float:
    """
    Empirical Bayes Geometric Mean (DuMouchel shrinkage).
    Shrinks extreme PRR values for sparse drug-event pairs toward the prior.

    Stub -- full implementation on Day 7 using the DuMouchel (1999) mixture model.
    """
    raise NotImplementedError("Implement on Day 7")

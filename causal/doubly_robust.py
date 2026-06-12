# -*- coding: utf-8 -*-
"""
causal/doubly_robust.py
=======================
Day 19 target: Doubly robust causal effect estimation for validated signal pairs.

Stubs only -- import-safe placeholder so the package is valid from Day 1.
"""


def estimate_ate(
    treatment, outcome, covariates,
    propensity_model=None, outcome_model=None
) -> dict:
    """
    Average Treatment Effect via LinearDRLearner (EconML).
    Returns {"ate": float, "ci_lower": float, "ci_upper": float}.

    Stub -- full implementation on Day 19.
    """
    raise NotImplementedError("Implement on Day 19")

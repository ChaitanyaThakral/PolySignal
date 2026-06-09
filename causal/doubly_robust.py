"""
causal/doubly_robust.py
=======================
Day 19 target: Doubly robust causal effect estimation for validated signal pairs.

Stubs only — import-safe placeholder so the package is valid from Day 1.

Key concept (from the Day 1 primer):
  Doubly robust = propensity score weighting + outcome regression combined.
  The estimator is consistent if EITHER the propensity model OR the outcome
  model is correctly specified — not both must be right simultaneously.
  This is what makes it more reliable than pure IPW or pure regression alone.
"""


def estimate_ate(
    treatment, outcome, covariates,
    propensity_model=None, outcome_model=None
) -> dict:
    """
    Average Treatment Effect via LinearDRLearner (EconML).
    Returns {'ate': float, 'ci_lower': float, 'ci_upper': float}.

    Stub — full implementation on Day 19.
    """
    raise NotImplementedError("Implement on Day 19")

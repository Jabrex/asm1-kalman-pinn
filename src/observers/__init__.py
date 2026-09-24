"""Model-based state estimators that share the PINN's physics (v1.1).

Nothing in this package reads ground truth: the static layer of
``tests/test_leakage.py`` scans it as part of the training path and, stricter
than for ``src/train``, forbids the truth names altogether (not even ``[0]``).
Initial information reaches the estimators only through anchor files.
"""

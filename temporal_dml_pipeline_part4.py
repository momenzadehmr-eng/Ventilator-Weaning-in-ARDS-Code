# ========================================================================
# SECTION 8 — TEMPORAL DOUBLE MACHINE LEARNING (CAUSAL FOREST)
# ========================================================================

class TemporalDMLEstimator:
    """
    Wrapper around econml.CausalForestDML for estimating
    history-conditioned associations (HCAs).
    """

    def __init__(self,
                 n_estimators=2000,
                 min_samples_leaf=10,
                 max_depth=None,
                 random_state=RANDOM_SEED):

        self.model = None
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state

    # --------------------------------------------------------------------
    def fit(self, Y, T, X, W=None):
        """
        Fit CausalForestDML.

        Args:
            Y: outcome (binary)
            T: treatment indicator (binary)
            X: heterogeneity features
            W: adjustment covariates (optional)
        """
        logger.info("Fitting Temporal DML (CausalForestDML)...")

        if W is None:
            W = X

        self.model = CausalForestDML(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state,
            discrete_treatment=True,
            n_crossfit_splits=3,
            verbose=0
        )

        self.model.fit(Y=Y, T=T, X=X, W=W)

        logger.info("Temporal DML fitting complete.")
        return self

    # --------------------------------------------------------------------
    def estimate_hcas(self, X):
        """
        Estimate individualized HCAs.

        Args:
            X: feature matrix

        Returns:
            tau_hat: array of HCAs
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        return self.model.effect(X)


# ========================================================================
# SECTION 9 — EVALUATION UTILITIES
# ========================================================================

def evaluate_predictions(y_true, y_pred):
    """
    Compute AUROC, AUPRC, and Brier score.
    """
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)

    logger.info(f"AUROC = {auroc:.3f}")
    logger.info(f"AUPRC = {auprc:.3f}")
    logger.info(f"Brier = {brier:.3f}")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier
    }


# ========================================================================
# SECTION 10 — SERIALIZATION HELPERS
# ========================================================================

def save_object(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to {path}")


def load_object(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded object from {path}")
    return obj


# ========================================================================
# SECTION 11 — MAIN GUARD
# ========================================================================

if __name__ == "__main__":
    logger.info("Temporal DML pipeline module loaded successfully.")
    logger.info("Use the classes programmatically to run the pipeline.")

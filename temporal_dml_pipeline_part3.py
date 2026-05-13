# ========================================================================
# SECTION 6 — ENSEMBLE NUISANCE PARAMETER ESTIMATOR
# ========================================================================

class EnsembleNuisanceEstimator:
    """
    Stacked ensemble for nuisance parameter estimation.
    Base learners:
        - XGBoost
        - Random Forest
        - LSTM
        - TCN
    Meta-learner:
        - Ridge regression
    """

    def __init__(self, model_type="classification", use_gpu=True):
        self.model_type = model_type
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        self.base_models = {
            "xgb": None,
            "rf": None,
            "lstm": None,
            "tcn": None
        }

        self.meta_model = None
        self.seq_len = None
        self.feature_dim = None

    # --------------------------------------------------------------------
    def _reshape_sequences(self, X):
        """
        Convert flat features → (samples, seq_len, feature_dim)
        """
        if len(X.shape) != 2:
            raise ValueError("X must be 2D for sequence reshaping.")

        if self.seq_len is None:
            self.seq_len = min(24, max(1, X.shape[1] // 10))
            self.feature_dim = X.shape[1] // self.seq_len

        X_trim = X[:, :self.seq_len * self.feature_dim]
        return X_trim.reshape(-1, self.seq_len, self.feature_dim)

    # --------------------------------------------------------------------
    def fit(self, X_train, y_train, X_val, y_val):
        logger.info("Training ensemble nuisance estimator...")

        # ---------------------------
        # 1) XGBoost
        # ---------------------------
        logger.info("Training XGBoost...")
        if self.model_type == "classification":
            self.base_models["xgb"] = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=50,
                random_state=RANDOM_SEED,
                tree_method="gpu_hist" if self.device == "cuda" else "hist",
                eval_metric="logloss"
            )
        else:
            self.base_models["xgb"] = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                random_state=RANDOM_SEED,
                tree_method="gpu_hist" if self.device == "cuda" else "hist"
            )

        self.base_models["xgb"].fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        # ---------------------------
        # 2) Random Forest
        # ---------------------------
        logger.info("Training Random Forest...")
        self.base_models["rf"] = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=50,
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        self.base_models["rf"].fit(X_train, y_train)

        # ---------------------------
        # 3) Temporal models (LSTM + TCN)
        # ---------------------------
        logger.info("Preparing sequences for LSTM/TCN...")
        X_train_seq = self._reshape_sequences(X_train)
        X_val_seq = self._reshape_sequences(X_val)

        train_ds = TemporalDataset(X_train_seq, y_train)
        val_ds = TemporalDataset(X_val_seq, y_val)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        input_dim = X_train_seq.shape[2]

        # LSTM
        logger.info("Training LSTM...")
        lstm = LSTMModel(input_dim=input_dim)
        lstm = train_neural_network(lstm, train_loader, val_loader, device=self.device)
        self.base_models["lstm"] = lstm

        # TCN
        logger.info("Training TCN...")
        tcn = TCNModel(input_dim=input_dim)
        tcn = train_neural_network(tcn, train_loader, val_loader, device=self.device)
        self.base_models["tcn"] = tcn

        # ---------------------------
        # 4) Meta-learner (Ridge)
        # ---------------------------
        logger.info("Training meta-learner...")

        preds = []

        # XGB
        if self.model_type == "classification":
            preds.append(self.base_models["xgb"].predict_proba(X_val)[:, 1])
        else:
            preds.append(self.base_models["xgb"].predict(X_val))

        # RF
        preds.append(self.base_models["rf"].predict_proba(X_val)[:, 1])

        # LSTM
        self.base_models["lstm"].eval()
        with torch.no_grad():
            Xv = torch.FloatTensor(X_val_seq).to(self.device)
            preds.append(self.base_models["lstm"](Xv).cpu().numpy())

        # TCN
        self.base_models["tcn"].eval()
        with torch.no_grad():
            Xv = torch.FloatTensor(X_val_seq).to(self.device)
            preds.append(self.base_models["tcn"](Xv).cpu().numpy())

        meta_X = np.vstack(preds).T
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_X, y_val)

        logger.info("Ensemble nuisance estimator training complete.")
        return self

    # --------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Predict ensemble probability.
        """
        preds = []

        # XGB
        preds.append(self.base_models["xgb"].predict_proba(X)[:, 1])

        # RF
        preds.append(self.base_models["rf"].predict_proba(X)[:, 1])

        # Temporal models
        X_seq = self._reshape_sequences(X)

        self.base_models["lstm"].eval()
        with torch.no_grad():
            preds.append(
                self.base_models["lstm"](torch.FloatTensor(X_seq).to(self.device)).cpu().numpy()
            )

        self.base_models["tcn"].eval()
        with torch.no_grad():
            preds.append(
                self.base_models["tcn"](torch.FloatTensor(X_seq).to(self.device)).cpu().numpy()
            )

        meta_X = np.vstack(preds).T
        out = self.meta_model.predict(meta_X)

        # Convert to probability
        return 1 / (1 + np.exp(-out))


# ========================================================================
# SECTION 7 — PROPENSITY SCORE ESTIMATOR
# ========================================================================

class PropensityEstimator:
    """
    Simple XGBoost-based propensity score estimator.
    Scores are truncated to [0.05, 0.95].
    """

    def __init__(self, lower=0.05, upper=0.95, use_gpu=True):
        self.lower = lower
        self.upper = upper
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = None

    # --------------------------------------------------------------------
    def fit(self, X, A):
        logger.info("Training propensity score model...")

        self.model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            random_state=RANDOM_SEED,
            tree_method="gpu_hist" if self.device == "cuda" else "hist",
            eval_metric="logloss"
        )

        self.model.fit(X, A)
        return self

    # --------------------------------------------------------------------
    def predict_proba(self, X):
        p = self.model.predict_proba(X)[:, 1]
        return np.clip(p, self.lower, self.upper)

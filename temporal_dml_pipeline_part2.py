# ========================================================================
# SECTION 2 — TEMPORAL FEATURE ENGINEERING
# ========================================================================

class TemporalFeatureEngineer:
    """
    Engineer temporal features from raw time-series data.
    Includes:
        - Resampling to fixed intervals
        - Rolling windows (6h, 24h)
        - Trend extraction
        - Weaning trial identification
    """

    def __init__(self,
                 short_window_hours: int = 6,
                 medium_window_hours: int = 24,
                 resample_minutes: int = 30):

        self.short_window = pd.Timedelta(hours=short_window_hours)
        self.medium_window = pd.Timedelta(hours=medium_window_hours)
        self.resample_freq = f"{resample_minutes}min"

        # Mapping itemid → variable names
        self.vital_map = {
            220045: "heart_rate",
            220179: "sbp_invasive",
            220050: "sbp_noninvasive",
            220180: "dbp_invasive",
            220051: "dbp_noninvasive",
            220052: "map_invasive",
            220181: "map_noninvasive",
            223761: "temperature_f",
            223762: "temperature_c",
            220210: "resp_rate"
        }

        self.vent_map = {
            220339: "peep",
            223835: "fio2_perc",
            220277: "fio2_set",
            224685: "tidal_volume",
            224684: "resp_rate_set",
            224686: "peak_pressure",
            224688: "plateau_pressure",
            224690: "minute_vent",
            224422: "pressure_support"
        }

    # --------------------------------------------------------------------
    def resample(self, df: pd.DataFrame, icustay_id: int) -> pd.DataFrame:
        """
        Resample each variable to fixed intervals with forward-fill.
        """
        df = df.copy()
        df["charttime"] = pd.to_datetime(df["charttime"])
        df = df.set_index("charttime")

        resampled = []

        for itemid in df["itemid"].unique():
            sub = df[df["itemid"] == itemid][["valuenum"]]
            r = sub.resample(self.resample_freq).mean().ffill(limit=4)

            name = self.vital_map.get(itemid) or self.vent_map.get(itemid, f"var_{itemid}")
            r.columns = [name]
            resampled.append(r)

        out = pd.concat(resampled, axis=1)
        out["icustay_id"] = icustay_id
        return out.reset_index()

    # --------------------------------------------------------------------
    def rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling 6h and 24h statistics + trend.
        """
        df = df.copy().set_index("charttime")

        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if "icustay_id" in numeric:
            numeric.remove("icustay_id")

        # 6-hour window
        for col in numeric:
            df[f"{col}_6h_mean"] = df[col].rolling(self.short_window, min_periods=3).mean()
            df[f"{col}_6h_std"] = df[col].rolling(self.short_window, min_periods=3).std()
            df[f"{col}_6h_min"] = df[col].rolling(self.short_window, min_periods=3).min()
            df[f"{col}_6h_max"] = df[col].rolling(self.short_window, min_periods=3).max()

        # 24-hour window
        for col in numeric:
            df[f"{col}_24h_mean"] = df[col].rolling(self.medium_window, min_periods=6).mean()
            df[f"{col}_24h_std"] = df[col].rolling(self.medium_window, min_periods=6).std()
            df[f"{col}_24h_trend"] = df[col].rolling(self.medium_window, min_periods=6).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 6 else np.nan
            )

        return df.reset_index()

    # --------------------------------------------------------------------
    def identify_weaning_trials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify weaning trial initiation events based on ventilator reductions.
        """
        df = df.copy()

        # FiO2 normalization
        if "fio2_perc" in df.columns:
            df["fio2"] = df["fio2_perc"] / 100.0
        elif "fio2_set" in df.columns:
            df["fio2"] = df["fio2_set"]

        df["peep_baseline"] = df["peep"].rolling("6h", min_periods=3).median()
        df["fio2_baseline"] = df["fio2"].rolling("6h", min_periods=3).median()

        df["peep_drop"] = df["peep_baseline"] - df["peep"]
        df["fio2_drop"] = df["fio2_baseline"] - df["fio2"]

        df["trial_raw"] = ((df["peep_drop"] >= 2) & (df["fio2_drop"] >= 0.10)).astype(int)
        df["trial_sustained"] = df["trial_raw"].rolling("2h", min_periods=2).sum() >= 2

        df["weaning_trial"] = (df["trial_raw"] & df["trial_sustained"]).astype(int)
        return df


# ========================================================================
# SECTION 3 — MISSING DATA HANDLING
# ========================================================================

class MissingDataHandler:
    """
    Hybrid missing data strategy:
        - <5% missing → median imputation
        - 5–40% missing → MICE
        - >40% missing → indicator + median
    """

    def __init__(self, mice_threshold: float = 0.40, max_iter: int = 10):
        self.mice_threshold = mice_threshold
        self.max_iter = max_iter
        self.median_imputer = None
        self.mice_imputer = None

    # --------------------------------------------------------------------
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        miss = X.isnull().mean()

        low = miss[miss < 0.05].index.tolist()
        mid = miss[(miss >= 0.05) & (miss < self.mice_threshold)].index.tolist()
        high = miss[miss >= self.mice_threshold].index.tolist()

        # Low missingness → median
        if low:
            self.median_imputer = SimpleImputer(strategy="median")
            X[low] = self.median_imputer.fit_transform(X[low])

        # Mid missingness → MICE
        if mid:
            self.mice_imputer = IterativeImputer(max_iter=self.max_iter, random_state=RANDOM_SEED)
            X[mid] = self.mice_imputer.fit_transform(X[mid])

        # High missingness → indicator + median
        for col in high:
            X[f"{col}_missing"] = X[col].isnull().astype(int)
            X[col] = X[col].fillna(X[col].median())

        return X

    # --------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if self.median_imputer:
            cols = [c for c in X.columns if c in self.median_imputer.feature_names_in_]
            X[cols] = self.median_imputer.transform(X[cols])

        if self.mice_imputer:
            cols = [c for c in X.columns if c in self.mice_imputer.feature_names_in_]
            X[cols] = self.mice_imputer.transform(X[cols])

        return X


# ========================================================================
# SECTION 4 — NEURAL NETWORK MODELS
# ========================================================================

class LSTMModel(nn.Module):
    """LSTM for temporal prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.dropout(h[-1])
        out = self.fc(out)
        return self.sigmoid(out).squeeze()


class TCNBlock(nn.Module):
    """Single TCN block."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()

        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        res = x
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        if self.down:
            res = self.down(res)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """Temporal Convolutional Network."""

    def __init__(self, input_dim, channels=128, layers=4, kernel=3, dropout=0.2):
        super().__init__()

        blocks = []
        for i in range(layers):
            dil = 2 ** i
            in_ch = input_dim if i == 0 else channels
            blocks.append(TCNBlock(in_ch, channels, kernel, dil, dropout))

        self.net = nn.Sequential(*blocks)
        self.fc = nn.Linear(channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.net(x)
        out = out.mean(dim=2)
        out = self.fc(out)
        return self.sigmoid(out).squeeze()


# ========================================================================
# SECTION 5 — DATASET + TRAINING LOOP
# ========================================================================

class TemporalDataset(Dataset):
    """PyTorch dataset for sequences."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def train_neural_network(model, train_loader, val_loader,
                         epochs=50, lr=0.001, patience=10, device="cuda"):

    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    scaler = GradScaler()

    best = float("inf")
    counter = 0
    best_state = None

    for ep in range(epochs):
        model.train()
        train_loss = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            opt.zero_grad()
            with autocast():
                pred = model(Xb)
                loss = loss_fn(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        logger.info(f"Epoch {ep+1}/{epochs} — Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            best_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    return model

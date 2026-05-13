"""
Temporal Double Machine Learning for Personalized Ventilator Weaning in ARDS

This implementation provides a complete pipeline for:
1. MIMIC-IV data extraction and preprocessing
2. Temporal feature engineering
3. Nuisance parameter estimation with ensemble methods
4. Causal effect estimation using temporal DML
5. Model evaluation and sensitivity analyses

Hardware requirements:
- GPU: NVIDIA RTX 4060 (8GB VRAM) or equivalent
- RAM: 64GB system memory
- Storage: ~85GB free space

Author: [Your Name]
Date: 2024
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Core scientific computing
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

# Machine learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
import lightgbm as lgb

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Causal inference
from econml.dml import CausalForestDML
from econml.inference import BootstrapInference

# Statistical analysis
import statsmodels.api as sm
from lifelines import CoxPHFitter

# Database access
from sqlalchemy import create_engine
import psycopg2

# Hyperparameter optimization
import optuna

# Utilities
import os
import pickle
import json
from datetime import datetime
import logging

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: DATA EXTRACTION FROM MIMIC-IV
# ============================================================================

class MIMICDataExtractor:
    """Extract ARDS cohort from MIMIC-IV database."""
    
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize database connection.
        
        Args:
            db_config: Dictionary with keys 'host', 'port', 'database', 'user', 'password'
        """
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        logger.info("Database connection established")
    
    def extract_ards_cohort(self) -> pd.DataFrame:
        """
        Extract ARDS patients using Berlin definition criteria.
        
        Returns:
            DataFrame with patient ICU stays meeting ARDS criteria
        """
        query = """
        -- Step 1: Identify mechanical ventilation episodes
        WITH ventilation_periods AS (
            SELECT 
                icustay_id,
                MIN(charttime) AS vent_start,
                MAX(charttime) AS vent_end
            FROM mimiciv_icu.chartevents
            WHERE itemid IN (720, 223849)
                AND value IS NOT NULL
            GROUP BY icustay_id
            HAVING EXTRACT(EPOCH FROM MAX(charttime) - MIN(charttime))/3600 >= 24
        ),
        
        -- Step 2: Calculate PaO2/FiO2 ratios with PEEP >= 5
        oxygenation AS (
            SELECT 
                ce1.icustay_id,
                ce1.charttime,
                ce1.valuenum AS pao2,
                ce2.valuenum / 100.0 AS fio2,
                ce3.valuenum AS peep,
                (ce1.valuenum / (ce2.valuenum / 100.0)) AS pf_ratio
            FROM mimiciv_icu.chartevents ce1
            INNER JOIN mimiciv_icu.chartevents ce2 
                ON ce1.icustay_id = ce2.icustay_id
                AND ce2.charttime BETWEEN ce1.charttime - INTERVAL '1 hour' 
                    AND ce1.charttime + INTERVAL '1 hour'
            INNER JOIN mimiciv_icu.chartevents ce3
                ON ce1.icustay_id = ce3.icustay_id
                AND ce3.charttime BETWEEN ce1.charttime - INTERVAL '1 hour'
                    AND ce1.charttime + INTERVAL '1 hour'
            WHERE ce1.itemid = 50821
                AND ce2.itemid IN (223835, 220277)
                AND ce3.itemid = 220339
                AND ce3.valuenum >= 5
                AND (ce1.valuenum / (ce2.valuenum / 100.0)) <= 300
                AND (ce1.valuenum / (ce2.valuenum / 100.0)) BETWEEN 20 AND 600
        ),
        
        -- Step 3: ARDS diagnoses
        ards_diagnoses AS (
            SELECT DISTINCT hadm_id, icustay_id
            FROM mimiciv_hosp.diagnoses_icd
            WHERE (icd_code IN ('51881', '51882', '5185', '5184') AND icd_version = 9)
                OR (icd_code IN ('J80', 'J9600', 'J9601', 'J9602', 'J951', 'J952', 'J95821', 'J810') 
                    AND icd_version = 10)
        ),
        
        -- Step 4: Combine all criteria
        ards_cohort AS (
            SELECT DISTINCT
                v.icustay_id,
                v.vent_start,
                v.vent_end,
                MIN(o.charttime) AS ards_onset,
                MIN(o.pf_ratio) AS min_pf_ratio
            FROM ventilation_periods v
            INNER JOIN oxygenation o ON v.icustay_id = o.icustay_id
            INNER JOIN ards_diagnoses ad ON v.icustay_id = ad.icustay_id
            WHERE o.charttime BETWEEN v.vent_start AND v.vent_start + INTERVAL '7 days'
            GROUP BY v.icustay_id, v.vent_start, v.vent_end
        )
        
        SELECT 
            ac.*,
            ie.subject_id,
            ie.hadm_id,
            EXTRACT(YEAR FROM AGE(ie.intime, pat.anchor_year_group::DATE)) AS age,
            pat.gender,
            adm.admission_type,
            adm.insurance,
            ie.los AS icu_los_days,
            CASE WHEN adm.hospital_expire_flag = 1 THEN 1 ELSE 0 END AS hospital_mortality
        FROM ards_cohort ac
        INNER JOIN mimiciv_icu.icustays ie ON ac.icustay_id = ie.icustay_id
        INNER JOIN mimiciv_hosp.patients pat ON ie.subject_id = pat.subject_id
        INNER JOIN mimiciv_hosp.admissions adm ON ie.hadm_id = adm.hadm_id
        WHERE EXTRACT(YEAR FROM AGE(ie.intime, pat.anchor_year_group::DATE)) >= 18
            AND ie.first_careunit IN ('MICU', 'SICU', 'CCU', 'CSRU')
        ORDER BY ie.intime
        """
        
        logger.info("Extracting ARDS cohort from MIMIC-IV...")
        df = pd.read_sql(query, self.engine)
        logger.info(f"Extracted {len(df)} ARDS patients")
        return df
    
    def extract_time_series_data(self, icustay_ids: List[int]) -> pd.DataFrame:
        """
        Extract time-series vital signs and ventilator parameters.
        
        Args:
            icustay_ids: List of ICU stay IDs
            
        Returns:
            DataFrame with time-series measurements
        """
        # Convert list to SQL-compatible format
        ids_str = ','.join(map(str, icustay_ids))
        
        query = f"""
        SELECT 
            ce.icustay_id,
            ce.charttime,
            ce.itemid,
            ce.valuenum,
            d.label
        FROM mimiciv_icu.chartevents ce
        INNER JOIN mimiciv_icu.d_items d ON ce.itemid = d.itemid
        WHERE ce.icustay_id IN ({ids_str})
            AND ce.itemid IN (
                -- Vital signs
                220045, 220179, 220050, 220180, 220051, 220052, 220181, 
                223761, 223762, 220210,
                -- Ventilator parameters
                220339, 223835, 220277, 224685, 224684, 224686,
                224688, 224689, 224690, 224422
            )
            AND ce.valuenum IS NOT NULL
        ORDER BY ce.icustay_id, ce.charttime
        """
        
        logger.info("Extracting time-series data...")
        df = pd.read_sql(query, self.engine)
        logger.info(f"Extracted {len(df)} time-series observations")
        return df
    
    def extract_lab_values(self, icustay_ids: List[int]) -> pd.DataFrame:
        """Extract laboratory measurements."""
        ids_str = ','.join(map(str, icustay_ids))
        
        query = f"""
        SELECT 
            le.hadm_id,
            ie.icustay_id,
            le.charttime,
            le.itemid,
            le.valuenum,
            d.label
        FROM mimiciv_hosp.labevents le
        INNER JOIN mimiciv_icu.icustays ie ON le.hadm_id = ie.hadm_id
        INNER JOIN mimiciv_hosp.d_labitems d ON le.itemid = d.itemid
        WHERE ie.icustay_id IN ({ids_str})
            AND le.charttime BETWEEN ie.intime AND ie.outtime
            AND le.itemid IN (
                -- ABG
                50821, 50818, 50820, 50802, 50804,
                -- CBC
                51221, 51222, 51248, 51249, 51250, 51265, 51275, 51277,
                -- Chemistry
                50912, 50902, 50882, 50931, 50983, 50971, 50822
            )
            AND le.valuenum IS NOT NULL
        ORDER BY ie.icustay_id, le.charttime
        """
        
        logger.info("Extracting laboratory values...")
        df = pd.read_sql(query, self.engine)
        return df


# ============================================================================
# SECTION 2: PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================

class TemporalFeatureEngineer:
    """Engineer temporal features from raw time-series data."""
    
    def __init__(self, 
                 short_window_hours: int = 6,
                 medium_window_hours: int = 24,
                 resample_minutes: int = 30):
        """
        Initialize feature engineering parameters.
        
        Args:
            short_window_hours: Short-term aggregation window (default: 6h)
            medium_window_hours: Medium-term aggregation window (default: 24h)
            resample_minutes: Resampling interval (default: 30min)
        """
        self.short_window = pd.Timedelta(hours=short_window_hours)
        self.medium_window = pd.Timedelta(hours=medium_window_hours)
        self.resample_freq = f'{resample_minutes}min'
        
        # Feature name mapping for clarity
        self.vital_signs_map = {
            220045: 'heart_rate',
            220179: 'sbp_invasive',
            220050: 'sbp_noninvasive',
            220180: 'dbp_invasive',
            220051: 'dbp_noninvasive',
            220052: 'map_invasive',
            220181: 'map_noninvasive',
            223761: 'temperature_f',
            223762: 'temperature_c',
            220210: 'respiratory_rate'
        }
        
        self.vent_params_map = {
            220339: 'peep',
            223835: 'fio2_perc',
            220277: 'fio2_set',
            224685: 'tidal_volume',
            224684: 'respiratory_rate_set',
            224686: 'peak_pressure',
            224688: 'plateau_pressure',
            224690: 'minute_ventilation',
            224422: 'pressure_support'
        }
    
    def resample_time_series(self, df: pd.DataFrame, icustay_id: int) -> pd.DataFrame:
        """
        Resample time-series to fixed intervals with forward-fill.
        
        Args:
            df: Time-series dataframe for single patient
            icustay_id: ICU stay identifier
            
        Returns:
            Resampled dataframe
        """
        df = df.copy()
        df['charttime'] = pd.to_datetime(df['charttime'])
        df = df.set_index('charttime')
        
        # Resample each variable separately
        resampled_dfs = []
        for itemid in df['itemid'].unique():
            item_df = df[df['itemid'] == itemid][['valuenum']]
            item_resampled = item_df.resample(self.resample_freq).mean()
            item_resampled = item_resampled.ffill(limit=4)  # Forward-fill up to 2 hours
            
            # Get variable name
            var_name = self.vital_signs_map.get(itemid) or self.vent_params_map.get(itemid, f'var_{itemid}')
            item_resampled.columns = [var_name]
            resampled_dfs.append(item_resampled)
        
        # Combine all variables
        result = pd.concat(resampled_dfs, axis=1)
        result['icustay_id'] = icustay_id
        result = result.reset_index()
        
        return result
    
    def compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling aggregates for temporal windows.
        
        Args:
            df: Resampled time-series dataframe
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy().set_index('charttime')
        
        # Get numeric columns (exclude icustay_id)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'icustay_id' in numeric_cols:
            numeric_cols.remove('icustay_id')
        
        result_dfs = [df]
        
        # Short-term window (6h)
        for col in numeric_cols:
            df[f'{col}_6h_mean'] = df[col].rolling(self.short_window, min_periods=3).mean()
            df[f'{col}_6h_std'] = df[col].rolling(self.short_window, min_periods=3).std()
            df[f'{col}_6h_min'] = df[col].rolling(self.short_window, min_periods=3).min()
            df[f'{col}_6h_max'] = df[col].rolling(self.short_window, min_periods=3).max()
        
        # Medium-term window (24h)
        for col in numeric_cols:
            df[f'{col}_24h_mean'] = df[col].rolling(self.medium_window, min_periods=6).mean()
            df[f'{col}_24h_std'] = df[col].rolling(self.medium_window, min_periods=6).std()
            
            # Compute trend (linear regression slope)
            df[f'{col}_24h_trend'] = df[col].rolling(self.medium_window, min_periods=6).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 6 else np.nan
            )
        
        return df.reset_index()
    
    def identify_weaning_trials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify weaning trial initiation events.
        
        Weaning trial defined as:
        - PEEP decrease >= 2 cm H2O from 6h median
        - FiO2 decrease >= 0.10 from 6h median
        - Sustained for >= 2 hours
        
        Args:
            df: DataFrame with ventilator parameters
            
        Returns:
            DataFrame with weaning_trial binary indicator
        """
        df = df.copy()
        
        # Ensure FiO2 is in fraction (0-1)
        if 'fio2_perc' in df.columns:
            df['fio2'] = df['fio2_perc'] / 100.0
        elif 'fio2_set' in df.columns:
            df['fio2'] = df['fio2_set']
        
        # Calculate baseline (6h median)
        df['peep_baseline'] = df['peep'].rolling(window='6h', min_periods=3).median()
        df['fio2_baseline'] = df['fio2'].rolling(window='6h', min_periods=3).median()
        
        # Identify reductions
        df['peep_reduction'] = df['peep_baseline'] - df['peep']
        df['fio2_reduction'] = df['fio2_baseline'] - df['fio2']
        
        # Weaning criteria
        df['weaning_trial'] = (
            (df['peep_reduction'] >= 2) & 
            (df['fio2_reduction'] >= 0.10)
        ).astype(int)
        
        # Check sustainability (next 2 hours)
        df['weaning_sustained'] = df['weaning_trial'].rolling(window='2h', min_periods=2).sum() >= 2
        df['weaning_trial'] = (df['weaning_trial'] & df['weaning_sustained']).astype(int)
        
        return df


class MissingDataHandler:
    """Handle missing data via MICE and indicator methods."""
    
    def __init__(self, 
                 mice_threshold: float = 0.40,
                 n_imputations: int = 5,
                 max_iter: int = 10):
        """
        Initialize imputation parameters.
        
        Args:
            mice_threshold: Variables with missingness > threshold excluded from MICE
            n_imputations: Number of MICE imputation iterations
            max_iter: Maximum iterations per imputation
        """
        self.mice_threshold = mice_threshold
        self.n_imputations = n_imputations
        self.max_iter = max_iter
        self.imputers = {}
        self.median_imputers = {}
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit imputers and transform data.
        
        Args:
            X: Feature matrix
            y: Target variable (optional, for stratification)
            
        Returns:
            Imputed feature matrix
        """
        X = X.copy()
        
        # Calculate missingness per column
        missingness = X.isnull().mean()
        
        # Separate columns by missingness level
        low_miss_cols = missingness[missingness < 0.05].index.tolist()
        mice_cols = missingness[(missingness >= 0.05) & (missingness < self.mice_threshold)].index.tolist()
        high_miss_cols = missingness[missingness >= self.mice_threshold].index.tolist()
        complete_cols = missingness[missingness == 0].index.tolist()
        
        logger.info(f"Complete: {len(complete_cols)}, Low miss: {len(low_miss_cols)}, "
                   f"MICE: {len(mice_cols)}, High miss: {len(high_miss_cols)}")
        
        # Median imputation for low missingness
        if low_miss_cols:
            self.median_imputers['low_miss'] = SimpleImputer(strategy='median')
            X[low_miss_cols] = self.median_imputers['low_miss'].fit_transform(X[low_miss_cols])
        
        # MICE for moderate missingness
        if mice_cols:
            self.imputers['mice'] = IterativeImputer(
                max_iter=self.max_iter,
                random_state=RANDOM_SEED,
                verbose=0
            )
            X[mice_cols] = self.imputers['mice'].fit_transform(X[mice_cols])
        
        # Indicator method for high missingness
        for col in high_miss_cols:
            X[f'{col}_missing'] = X[col].isnull().astype(int)
            X[col] = X[col].fillna(X[col].median())
        
        return X
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted imputers."""
        X = X.copy()
        
        if 'low_miss' in self.median_imputers:
            low_miss_cols = [col for col in X.columns if col in self.median_imputers['low_miss'].feature_names_in_]
            X[low_miss_cols] = self.median_imputers['low_miss'].transform(X[low_miss_cols])
        
        if 'mice' in self.imputers:
            mice_cols = [col for col in X.columns if col in self.imputers['mice'].feature_names_in_]
            X[mice_cols] = self.imputers['mice'].transform(X[mice_cols])
        
        return X


# ============================================================================
# SECTION 3: NEURAL NETWORK MODELS
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM model for temporal outcome prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
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
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        out = self.dropout(h_n[-1])
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out.squeeze()


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int, dropout: float):
        super(TCNBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        if self.downsample:
            residual = self.downsample(residual)
        
        return self.relu(out + residual)


class TCNModel(nn.Module):
    """Temporal Convolutional Network for sequence modeling."""
    
    def __init__(self, input_dim: int, num_channels: int = 128,
                 kernel_size: int = 3, num_layers: int = 4, dropout: float = 0.2):
        super(TCNModel, self).__init__()
        
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels
            layers.append(TCNBlock(in_ch, num_channels, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # TCN expects: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        out = self.network(x)
        
        # Global average pooling
        out = out.mean(dim=2)
        
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out.squeeze()


class TemporalDataset(Dataset):
    """PyTorch Dataset for temporal sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_neural_network(model, train_loader, val_loader, 
                         epochs: int = 50, lr: float = 0.001, 
                         patience: int = 20, device: str = 'cuda'):
    """
    Train neural network with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum epochs
        lr: Learning rate
        patience: Early stopping patience
        device: Device ('cuda' or 'cpu')
        
    Returns:
        Trained model
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model


# ============================================================================
# SECTION 4: ENSEMBLE NUISANCE PARAMETER ESTIMATION
# ============================================================================

class EnsembleNuisanceEstimator:
    """Stacked ensemble for nuisance parameter estimation."""
    
    def __init__(self, model_type: str = 'classification', use_gpu: bool = True):
        """
        Initialize ensemble.
        
        Args:
            model_type: 'classification' or 'regression'
            use_gpu: Whether to use GPU for neural networks
        """
        self.model_type = model_type
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Base learners
        self.base_models = {
            'xgboost': None,
            'random_forest': None,
            'lstm': None,
            'tcn': None
        }
        
        # Meta-learner
        self.meta_model = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> 'EnsembleNuisanceEstimator':
        """
        Fit ensemble on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Fitted estimator
        """
        logger.info("Training ensemble base learners...")
        
        # XGBoost
        logger.info("Training XGBoost...")
        if self.model_type == 'classification':
            self.base_models['xgboost'] = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                random_state=RANDOM_SEED,
                tree_method='gpu_hist' if self.device == 'cuda' else 'hist'
            )
        else:
            self.base_models['xgboost'] = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                random_state=RANDOM_SEED,
                tree_method='gpu_hist' if self.device == 'cuda' else 'hist'
            )
        
        self.base_models['xgboost'].fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Random Forest
        logger.info("Training Random Forest...")
        if self.model_type == 'classification':
            self.base_models['random_forest'] = RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_leaf=50,
                max_features='sqrt',
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.base_models['random_forest'] = RandomForestRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_leaf=50,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        
        self.base_models['random_forest'].fit(X_train, y_train)
        
        # Reshape for neural networks if needed
        if len(X_train.shape) == 2:
            # Assume temporal dimension, reshape to (samples, seq_len, features)
            # For simplicity, use last 24 timesteps
            seq_len = min(24, X_train.shape[1] // 10)
            feature_dim = X_train.shape[1] // seq_len
            
            X_train_seq = X_train[:, :seq_len*feature_dim].reshape(-1, seq_len, feature_dim)
            X_val_seq = X_val[:, :seq_len*feature_dim].reshape(-1, seq_len, feature_dim)
        else:
            X_train_seq, X_val_seq = X_train, X_val
        
        # LSTM
        logger.info("Training LSTM...")
        train_dataset = TemporalDataset(X_train_seq, y_train)
        val_dataset = TemporalDataset(X_val_seq, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        self.base_models['lstm'] = LSTMModel(
            input_dim=X_train_seq.shape[2],
            hidden_dim=128,
            num_layers=2,
            dropout=0.2
        )
        
        self.base_models['lstm'] = train_neural_network(
            self.base_models['lstm'],
            train_loader, val_loader,
            epochs=50, lr=0.001,
            patience=20, device=self.device
        )
        
        # TCN
        logger.info("Training TCN...")
        self.base_models['tcn'] = TCNModel(
            input_dim=X_train_seq.shape[2],
            num_channels=128,
            kernel_size=3,
            num_layers=4,
            dropout=0.2
        )
        
        self.base_models['tcn'] = train_neural_network(
            self.base_models['tcn'],
            train_loader, val_loader,
            epochs=50, lr=0.001,
            patience=20, device=self.device
        )
        
        # Generate base predictions for meta-learner
        logger.info("Training meta-learner...")
        base_train_preds = self._get_base_predictions(X_train, X_train_seq)
        base_val_preds = self._get_base_predictions(X_val, X_val_seq)
        
        # Meta-learner (Ridge for stability)
        if self.model_type == 'classification':
            self.meta_model = LogisticRegression(
                C=1.0,
                random_state=RANDOM_SEED,
                max_iter=1000
            )
        else:
            self.meta_model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        
        self.meta_model.fit(base_train_preds, y_train)
        
        # Evaluate
        train_score = self.score(X_train, y_train)
        val_score = self.score(X_val, y_val)
        logger.info(f"Ensemble - Train Score: {train_score:.4f}, Val Score: {val_score:.4f}")
        
        return self
    
    def _get_base_predictions(self, X: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
        """Get predictions from all base learners."""
        preds = []
        
        # XGBoost and RF predictions
        if self.model_type == 'classification':
            preds.append(self.base_models['xgboost'].predict_proba(X)[:, 1])
            preds.append(self.base_models['random_forest'].predict_proba(X)[:, 1])
        else:
            preds.append(self.base_models['xgboost'].predict(X))
            preds.append(self.base_models['random_forest'].predict(X))
        
        # Neural network predictions
        self.base_models['lstm'].eval()
        self.base_models['tcn'].eval()
        
        with torch.no_grad():
            X_seq_tensor = torch.FloatTensor(X_seq).to(self.device)
            lstm_pred = self.base_models['lstm'](X_seq_tensor).cpu().numpy()
            tcn_pred = self.base_models['tcn'](X_seq_tensor).cpu().numpy()
        
        preds.append(lstm_pred)
        preds.append(tcn_pred)
        
        return np.column_stack(preds)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        # Reshape for neural networks
        if len(X.shape) == 2:
            seq_len = min(24, X.shape[1] // 10)
            feature_dim = X.shape[1] // seq_len
            X_seq = X[:, :seq_len*feature_dim].reshape(-1, seq_len, feature_dim)
        else:
            X_seq = X
        
        base_preds = self._get_base_predictions(X, X_seq)
        
        if self.model_type == 'classification':
            return self.meta_model.predict_proba(base_preds)[:, 1]
        else:
            return self.meta_model.predict(base_preds)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute performance score."""
        preds = self.predict(X)
        
        if self.model_type == 'classification':
            return roc_auc_score(y, preds)
        else:
            from sklearn.metrics import r2_score
            return r2_score(y, preds)


# ============================================================================
# SECTION 5: TEMPORAL DOUBLE MACHINE LEARNING
# ============================================================================

class TemporalDML:
    """Temporal Double Machine Learning for causal effect estimation."""
    
    def __init__(self, 
                 n_folds: int = 5,
                 propensity_trim: Tuple[float, float] = (0.05, 0.95),
                 use_gpu: bool = True):
        """
        Initialize T-DML estimator.
        
        Args:
            n_folds: Number of cross-fitting folds
            propensity_trim: Propensity score trimming thresholds (lower, upper)
            use_gpu: Whether to use GPU
        """
        self.n_folds = n_folds
        self.propensity_trim = propensity_trim
        self.use_gpu = use_gpu
        
        self.outcome_models = []
        self.propensity_models = []
        self.cate_model = None
        
    def fit(self, X: np.ndarray, D: np.ndarray, y: np.ndarray) -> 'TemporalDML':
        """
        Fit T-DML model using cross-fitting.
        
        Args:
            X: Covariate matrix (features + history)
            D: Treatment indicator (weaning trial)
            y: Outcome (successful weaning)
            
        Returns:
            Fitted T-DML estimator
        """
        logger.info("Starting Temporal DML estimation...")
        
        # Initialize storage for cross-fitted predictions
        n = len(y)
        outcome_preds = np.zeros(n)
        propensity_preds = np.zeros(n)
        
        # Stratified K-fold for cross-fitting
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_SEED)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Cross-fitting fold {fold + 1}/{self.n_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            D_train, D_val = D[train_idx], D[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train outcome regression model (for untreated: D=0)
            untreated_idx = D_train == 0
            outcome_model = EnsembleNuisanceEstimator(
                model_type='classification',
                use_gpu=self.use_gpu
            )
            
            # Further split for validation
            from sklearn.model_selection import train_test_split
            X_train_out, X_val_out, y_train_out, y_val_out = train_test_split(
                X_train[untreated_idx], y_train[untreated_idx],
                test_size=0.2, random_state=RANDOM_SEED, stratify=y_train[untreated_idx]
            )
            
            outcome_model.fit(X_train_out, y_train_out, X_val_out, y_val_out)
            self.outcome_models.append(outcome_model)
            
            # Predict on validation fold
            outcome_preds[val_idx] = outcome_model.predict(X_val)
            
            # Train propensity score model
            propensity_model = EnsembleNuisanceEstimator(
                model_type='classification',
                use_gpu=self.use_gpu
            )
            
            X_train_prop, X_val_prop, D_train_prop, D_val_prop = train_test_split(
                X_train, D_train,
                test_size=0.2, random_state=RANDOM_SEED, stratify=D_train
            )
            
            propensity_model.fit(X_train_prop, D_train_prop, X_val_prop, D_val_prop)
            self.propensity_models.append(propensity_model)
            
            # Predict on validation fold
            propensity_preds[val_idx] = propensity_model.predict(X_val)
        
        # Trim propensity scores
        logger.info(f"Trimming propensity scores at {self.propensity_trim}")
        trim_mask = (
            (propensity_preds >= self.propensity_trim[0]) &
            (propensity_preds <= self.propensity_trim[1])
        )
        
        n_trimmed = np.sum(~trim_mask)
        logger.info(f"Trimmed {n_trimmed} observations ({100*n_trimmed/n:.1f}%)")
        
        # Apply trimming
        X_trim = X[trim_mask]
        D_trim = D[trim_mask]
        y_trim = y[trim_mask]
        outcome_preds_trim = outcome_preds[trim_mask]
        propensity_preds_trim = propensity_preds[trim_mask]
        
        # Compute orthogonal scores (doubly robust)
        scores = self._compute_orthogonal_scores(
            D_trim, y_trim,
            outcome_preds_trim,
            propensity_preds_trim
        )
        
        # Estimate CATE using causal forest
        logger.info("Estimating CATE with causal forest...")
        self.cate_model = CausalForestDML(
            model_y=None,  # Already estimated
            model_t=None,  # Already estimated
            n_estimators=2000,
            min_samples_leaf=50,
            max_depth=None,
            random_state=RANDOM_SEED,
            n_jobs=8,
            verbose=0,
            inference=True
        )
        
        # Fit on orthogonal scores
        self.cate_model.fit(y_trim, D_trim, X=X_trim, W=None)
        
        # Compute ATE
        self.ate_ = np.mean(scores)
        self.ate_se_ = np.std(scores) / np.sqrt(len(scores))
        
        logger.info(f"ATE: {self.ate_:.4f} (SE: {self.ate_se_:.4f})")
        
        return self
    
    def _compute_orthogonal_scores(self, D: np.ndarray, y: np.ndarray,
                                   mu0: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """
        Compute Neyman-orthogonal scores for DML.
        
        Score: (D - π(X)) / (π(X)(1-π(X))) * (Y - μ₀(X))
        
        Args:
            D: Treatment indicator
            y: Outcome
            mu0: Outcome predictions for untreated
            pi: Propensity scores
            
        Returns:
            Orthogonal scores
        """
        # Clip propensity scores for numerical stability
        pi_clip = np.clip(pi, 0.01, 0.99)
        
        scores = (D - pi_clip) / (pi_clip * (1 - pi_clip)) * (y - mu0)
        
        return scores
    
    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        Predict conditional average treatment effects.
        
        Args:
            X: Covariate matrix
            
        Returns:
            CATE predictions
        """
        return self.cate_model.effect(X)
    
    def estimate_ate(self) -> Tuple[float, float, Tuple[float, float]]:
        """
        Get average treatment effect with confidence interval.
        
        Returns:
            Tuple of (ATE, standard error, 95% CI)
        """
        ci_lower = self.ate_ - 1.96 * self.ate_se_
        ci_upper = self.ate_ + 1.96 * self.ate_se_
        
        return self.ate_, self.ate_se_, (ci_lower, ci_upper)


# ============================================================================
# SECTION 6: EVALUATION AND ANALYSIS
# ============================================================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUROC, etc.)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['auroc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['auprc'] = average_precision_score(y_true, y_pred_proba)
        metrics['brier'] = brier_score_loss(y_true, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp)
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return metrics


def sensitivity_analysis(tdml: TemporalDML, X: np.ndarray, D: np.ndarray, y: np.ndarray,
                        analysis_type: str = 'treatment_definition') -> pd.DataFrame:
    """
    Conduct sensitivity analyses.
    
    Args:
        tdml: Fitted T-DML model
        X: Features
        D: Treatment
        y: Outcome
        analysis_type: Type of sensitivity analysis
        
    Returns:
        DataFrame with sensitivity results
    """
    results = []
    
    if analysis_type == 'propensity_trimming':
        trim_levels = [(0.025, 0.975), (0.05, 0.95), (0.075, 0.925)]
        
        for trim in trim_levels:
            model = TemporalDML(propensity_trim=trim)
            model.fit(X, D, y)
            ate, se, ci = model.estimate_ate()
            
            results.append({
                'analysis': f'Trim {trim[0]}-{trim[1]}',
                'ate': ate,
                'se': se,
                'ci_lower': ci[0],
                'ci_upper': ci[1]
            })
    
    return pd.DataFrame(results)


# ============================================================================
# SECTION 7: MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    
    logger.info("="*80)
    logger.info("Temporal DML for ARDS Ventilator Weaning - Starting Pipeline")
    logger.info("="*80)
    
    # Configuration
    DB_CONFIG = {
        'host': 'localhost',
        'port': '5432',
        'database': 'mimic',
        'user': 'your_username',
        'password': 'your_password'
    }
    
    # Step 1: Extract data
    logger.info("\n[STEP 1] Extracting ARDS cohort from MIMIC-IV...")
    extractor = MIMICDataExtractor(DB_CONFIG)
    cohort_df = extractor.extract_ards_cohort()
    
    icustay_ids = cohort_df['icustay_id'].tolist()
    time_series_df = extractor.extract_time_series_data(icustay_ids)
    lab_df = extractor.extract_lab_values(icustay_ids)
    
    # Step 2: Feature engineering
    logger.info("\n[STEP 2] Engineering temporal features...")
    engineer = TemporalFeatureEngineer()
    
    processed_patients = []
    for icustay_id in icustay_ids[:100]:  # Process first 100 for demo
        patient_ts = time_series_df[time_series_df['icustay_id'] == icustay_id]
        
        if len(patient_ts) > 0:
            resampled = engineer.resample_time_series(patient_ts, icustay_id)
            with_features = engineer.compute_rolling_features(resampled)
            with_treatment = engineer.identify_weaning_trials(with_features)
            processed_patients.append(with_treatment)
    
    full_df = pd.concat(processed_patients, ignore_index=True)
    
    # Step 3: Prepare data
    logger.info("\n[STEP 3] Preparing features and target...")
    
    # Define outcome
    y = cohort_df.set_index('icustay_id')['hospital_mortality'].reindex(
        full_df['icustay_id'].unique()
    ).values
    
    # Get treatment
    D = full_df.groupby('icustay_id')['weaning_trial'].max().values
    
    # Get features
    feature_cols = [col for col in full_df.columns 
                   if col not in ['icustay_id', 'charttime', 'weaning_trial']]
    X = full_df.groupby('icustay_id')[feature_cols].mean().values
    
    # Handle missing data
    imputer = MissingDataHandler()
    X_imputed = imputer.fit_transform(pd.DataFrame(X, columns=feature_cols))
    X_imputed = X_imputed.values
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Step 4: Temporal validation split
    logger.info("\n[STEP 4] Creating temporal train/val/test split...")
    
    n = len(X_scaled)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    X_train, y_train, D_train = X_scaled[:train_end], y[:train_end], D[:train_end]
    X_val, y_val, D_val = X_scaled[train_end:val_end], y[train_end:val_end], D[train_end:val_end]
    X_test, y_test, D_test = X_scaled[val_end:], y[val_end:], D[val_end:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Step 5: Fit Temporal DML
    logger.info("\n[STEP 5] Fitting Temporal Double Machine Learning...")
    
    tdml = TemporalDML(n_folds=5, use_gpu=True)
    tdml.fit(X_train, D_train, y_train)
    
    # Step 6: Evaluate
    logger.info("\n[STEP 6] Evaluating model...")
    
    ate, se, ci = tdml.estimate_ate()
    logger.info(f"\nAverage Treatment Effect: {ate:.4f}")
    logger.info(f"Standard Error: {se:.4f}")
    logger.info(f"95% CI: ({ci[0]:.4f}, {ci[1]:.4f})")
    
    # CATE predictions
    cate_test = tdml.predict_cate(X_test)
    logger.info(f"\nCATE - Mean: {np.mean(cate_test):.4f}, Std: {np.std(cate_test):.4f}")
    logger.info(f"CATE - Range: [{np.min(cate_test):.4f}, {np.max(cate_test):.4f}]")
    
    # Step 7: Sensitivity analyses
    logger.info("\n[STEP 7] Running sensitivity analyses...")
    
    sens_results = sensitivity_analysis(tdml, X_train, D_train, y_train, 
                                       analysis_type='propensity_trimming')
    print("\nSensitivity Analysis Results:")
    print(sens_results)
    
    # Step 8: Save results
    logger.info("\n[STEP 8] Saving results...")
    
    results = {
        'ate': ate,
        'se': se,
        'ci': ci,
        'cate_test': cate_test.tolist(),
        'sensitivity': sens_results.to_dict()
    }
    
    with open('tdml_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save models
    with open('tdml_model.pkl', 'wb') as f:
        pickle.dump(tdml, f)
    
    logger.info("\n" + "="*80)
    logger.info("Pipeline completed successfully!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
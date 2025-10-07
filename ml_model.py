import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import joblib
import pickle
import os
warnings.filterwarnings('ignore')

class DeepLearningCropPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_scalers = {
            'min': MinMaxScaler(),
            'modal': MinMaxScaler(),
            'max': MinMaxScaler()
        }
        self.feature_columns = None
        self.is_fitted = False
        self.df_processed = None
        self.price_columns = ['Min Price (Rs)', 'Modal Price (Rs)', 'Max Price (Rs)']
        self.global_medians = {}  # Store global medians for feature fallback
        
    def load_and_prepare_data(self, file_path):
        """Load and prepare the dataset for modeling"""
        print(f"Loading data from: {file_path}")
        
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not read the CSV file with any encoding")
        
        df.columns = df.columns.str.strip()
        
        date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']
        for fmt in date_formats:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format=fmt, errors='raise')
                break
            except:
                continue
        else:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        price_cols_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'min price' in col_lower:
                price_cols_map['min'] = col
            elif 'modal price' in col_lower:
                price_cols_map['modal'] = col
            elif 'max price' in col_lower:
                price_cols_map['max'] = col
        
        if 'modal' not in price_cols_map:
            print("Warning: No 'Modal Price' column found. Using average of Min and Max for Modal Price.")
            price_cols_map['modal'] = None
        
        for price_type, col in price_cols_map.items():
            if price_type == 'modal' and col is None:
                if 'min' in price_cols_map and 'max' in price_cols_map:
                    min_col = price_cols_map['min']
                    max_col = price_cols_map['max']
                    df['Modal Price (Rs)'] = (pd.to_numeric(df[min_col], errors='coerce') + pd.to_numeric(df[max_col], errors='coerce')) / 2
                    print(f"Computed Modal Price as average of {min_col} and {max_col}")
                else:
                    raise ValueError("Cannot compute Modal Price without Min and Max columns")
            else:
                df[f"{price_type.capitalize()} Price (Rs)"] = pd.to_numeric(df[col], errors='coerce')
                print(f"Using {price_type} price column: {col}")
        
        missing_cols = [col for col in self.price_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required price columns: {missing_cols}")
        
        initial_rows = len(df)
        df = df.dropna(subset=['Date'] + self.price_columns)
        print(f"Removed {initial_rows - len(df)} rows with missing data")
        print(f"Final dataset: {len(df)} rows")
        
        return df
    
    def create_advanced_features(self, df, is_training=True):
        """Create advanced features for deep learning with enhanced fallback"""
        df = df.copy()
        
        df = df.sort_values(['APMC', 'Commodity', 'Date'])
        
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
        df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
        
        df['Season'] = df['Month'].apply(lambda x: 
            1 if x in [12, 1, 2] else  # Winter
            2 if x in [3, 4, 5] else   # Spring
            3 if x in [6, 7, 8] else   # Summer
            4)                         # Monsoon
        
        for price_col in self.price_columns:
            price_type = price_col.split()[0].lower()
            for window in [3, 7, 14, 30]:
                df[f'{price_type}_price_rolling_mean_{window}d'] = df.groupby(['APMC', 'Commodity'])[price_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'{price_type}_price_rolling_std_{window}d'] = df.groupby(['APMC', 'Commodity'])[price_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
            
            for lag in [1, 3, 7, 14]:
                df[f'{price_type}_price_lag_{lag}d'] = df.groupby(['APMC', 'Commodity'])[price_col].shift(lag)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if is_training:
            self.global_medians = {col: df[col].median() for col in numeric_cols if col not in self.price_columns}
            for col in numeric_cols:
                if col not in self.price_columns:
                    df[col] = df.groupby(['APMC', 'Commodity'])[col].fillna(method='ffill').fillna(method='bfill')
                    df[col] = df[col].fillna(self.global_medians.get(col, 0))
        else:
            for col in numeric_cols:
                if col not in self.price_columns:
                    df[col] = df[col].fillna(self.global_medians.get(col, 0))
        
        return df
    
    def prepare_features_for_dl(self, df):
        """Prepare features specifically for deep learning"""
        df = self.create_advanced_features(df, is_training=True)
        
        basic_features = ['Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'WeekOfYear', 'DayOfYear', 'Season']
        cyclical_features = ['Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 
                           'DayOfWeek_sin', 'DayOfWeek_cos', 'Quarter_sin', 'Quarter_cos']
        
        rolling_features = [col for col in df.columns if 'rolling' in col]
        lag_features = [col for col in df.columns if 'lag' in col]
        
        numeric_features = basic_features + cyclical_features + rolling_features + lag_features
        categorical_features = ['APMC', 'Commodity']
        all_features = categorical_features + numeric_features
        
        X = df[all_features].copy()
        y = df[self.price_columns].copy()
        
        for col in categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        self.feature_columns = X.columns.tolist()
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        
        return X, y, df
    
    def build_neural_network(self, input_dim, categorical_dims):
        """Build an advanced neural network for multi-output price prediction"""
        main_input = layers.Input(shape=(input_dim,), name='main_input')
        
        embeddings = []
        categorical_inputs = []
        
        for i, (col, dim) in enumerate(categorical_dims.items()):
            cat_input = layers.Input(shape=(1,), name=f'{col}_input')
            embedding_dim = min(50, (dim + 1) // 2)
            embedding = layers.Embedding(dim, embedding_dim, name=f'{col}_embedding')(cat_input)
            embedding = layers.Flatten()(embedding)
            embeddings.append(embedding)
            categorical_inputs.append(cat_input)
        
        if embeddings:
            combined = layers.Concatenate()(embeddings + [main_input])
        else:
            combined = main_input
        
        x = layers.Dense(512, activation='relu', name='dense1')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(256, activation='relu', name='dense2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation='relu', name='dense3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Dense(64, activation='relu', name='dense4')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Dense(32, activation='relu', name='dense5')(x)
        
        outputs = []
        for price_type in ['min', 'modal', 'max']:
            output = layers.Dense(1, activation='linear', name=f'{price_type}_output')(x)
            outputs.append(output)
        
        model = keras.Model(inputs=categorical_inputs + [main_input], outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={'min_output': 'huber', 'modal_output': 'huber', 'max_output': 'huber'},
            metrics={'min_output': ['mae'], 'modal_output': ['mae'], 'max_output': ['mae']}
        )
        
        return model
    
    def train(self, data_path, epochs=100):
        """Train the neural network model for multiple price predictions"""
        print("="*60)
        print("DEEP LEARNING CROP PRICE PREDICTION MODEL")
        print("="*60)
        
        df = self.load_and_prepare_data(data_path)
        
        print(f"\nData Summary:")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Unique APMCs: {df['APMC'].nunique()}")
        print(f"Unique Commodities: {df['Commodity'].nunique()}")
        for price_col in self.price_columns:
            print(f"{price_col} range: Rs.{df[price_col].min():.2f} to Rs.{df[price_col].max():.2f}")
            print(f"{price_col} average: Rs.{df[price_col].mean():.2f}")
        
        print(f"\nPreparing advanced features for deep learning...")
        X, y, df_processed = self.prepare_features_for_dl(df)
        
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.fillna(0)
        
        print(f"Total features created: {len(X.columns)}")
        print(f"Categorical features: {len(self.categorical_features)}")
        print(f"Numeric features: {len(self.numeric_features)}")
        
        print(f"\nTraining set: {len(X)} samples (full dataset)")
        
        numeric_cols = [col for col in X.columns if col not in self.categorical_features]
        X_scaled = X.copy()
        
        if numeric_cols:
            X_scaled[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        
        y_scaled = {}
        for i, price_col in enumerate(self.price_columns):
            price_type = price_col.split()[0].lower()
            y_scaled[price_type] = self.target_scalers[price_type].fit_transform(
                y[price_col].values.reshape(-1, 1)
            ).flatten()
        
        categorical_dims = {}
        for col in self.categorical_features:
            categorical_dims[col] = X[col].max() + 1
        
        print(f"\nBuilding neural network architecture...")
        input_dim = len(numeric_cols)
        self.model = self.build_neural_network(input_dim, categorical_dims)
        
        print(self.model.summary())
        
        if self.categorical_features:
            train_data = []
            for col in self.categorical_features:
                train_data.append(X_scaled[col].values)
            train_data.append(X_scaled[numeric_cols].values)
        else:
            train_data = X_scaled.values
        
        train_targets = [y_scaled['min'], y_scaled['modal'], y_scaled['max']]
        
        early_stopping = callbacks.EarlyStopping(
            monitor='loss', patience=15, restore_best_weights=True, verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1
        )
        
        print(f"\nTraining neural network for {epochs} epochs...")
        history = self.model.fit(
            train_data, train_targets,
            epochs=epochs,
            batch_size=64,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        y_pred_scaled = self.model.predict(train_data)
        y_pred = {}
        for i, price_type in enumerate(['min', 'modal', 'max']):
            y_pred[price_type] = self.target_scalers[price_type].inverse_transform(
                y_pred_scaled[i].reshape(-1, 1)
            ).flatten()
        
        metrics = {'train_metrics': {}}
        for price_col in self.price_columns:
            price_type = price_col.split()[0].lower()
            train_mae = mean_absolute_error(y[price_col], y_pred[price_type])
            train_rmse = np.sqrt(mean_squared_error(y[price_col], y_pred[price_type]))
            train_r2 = r2_score(y[price_col], y_pred[price_type])
            metrics['train_metrics'][price_type] = {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2}
        
        print("\n" + "="*50)
        print("NEURAL NETWORK MODEL PERFORMANCE")
        print("="*50)
        for price_type, price_col in zip(['min', 'modal', 'max'], self.price_columns):
            print(f"\n{price_col} Training Set:")
            print(f"  MAE: Rs.{metrics['train_metrics'][price_type]['mae']:.2f}")
            print(f"  RMSE: Rs.{metrics['train_metrics'][price_type]['rmse']:.2f}")
            print(f"  R² Score: {metrics['train_metrics'][price_type]['r2']:.4f}")
        
        self.is_fitted = True
        self.df_processed = df_processed
        
        self.plot_training_history(history)
        
        return {
            'train_metrics': metrics['train_metrics'],
            'predictions': y_pred,
            'actual': y
        }
    
    def plot_training_history(self, history):
        """Plot training history for neural network"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for price_type in ['min', 'modal', 'max']:
            ax1.plot(history.history[f'{price_type}_output_loss'], label=f'{price_type.capitalize()} Training Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        for price_type in ['min', 'modal', 'max']:
            ax2.plot(history.history[f'{price_type}_output_mae'], label=f'{price_type.capitalize()} Training MAE')
        ax2.set_title('Model Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, apmc, commodity, date):
        """Predict min, modal, and max prices using the neural network"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Call train() or load_model() first.")
        
        if isinstance(date, str):
            try:
                date = pd.to_datetime(date)
            except ValueError as e:
                raise ValueError(f"Invalid date format. Please use YYYY-MM-DD. Details: {str(e)}")
        
        apmc_normalized = apmc.upper()
        commodity_normalized = self._normalize_commodity(commodity)
        
        print(f"Normalized inputs: APMC='{apmc_normalized}', Commodity='{commodity_normalized}'")
        
        input_df = pd.DataFrame({
            'APMC': [apmc_normalized],
            'Commodity': [commodity_normalized],
            'Date': [date],
            'Min Price (Rs)': [0],
            'Modal Price (Rs)': [0],
            'Max Price (Rs)': [0]
        })
        
        input_df = self.create_advanced_features(input_df, is_training=False)
        
        # Check historical data with fallbacks
        hist_data = self.df_processed[
            (self.df_processed['APMC'] == apmc_normalized) & 
            (self.df_processed['Commodity'] == commodity_normalized) &
            (self.df_processed['Date'] < date)
        ].sort_values('Date')
        
        if len(hist_data) == 0:
            print(f"Warning: No historical data for APMC='{apmc_normalized}', Commodity='{commodity_normalized}'. Falling back to Commodity-level data.")
            hist_data = self.df_processed[
                (self.df_processed['Commodity'] == commodity_normalized) &
                (self.df_processed['Date'] < date)
            ].sort_values('Date')
        
        if len(hist_data) == 0:
            print(f"Warning: No historical data for Commodity='{commodity_normalized}'. Falling back to global data.")
            hist_data = self.df_processed[self.df_processed['Date'] < date].sort_values('Date')
        
        if len(hist_data) > 0:
            for price_col in self.price_columns:
                price_type = price_col.split()[0].lower()
                recent_prices = hist_data[price_col].values
                for window in [3, 7, 14, 30]:
                    if len(recent_prices) >= window:
                        input_df[f'{price_type}_price_rolling_mean_{window}d'] = recent_prices[-window:].mean()
                        input_df[f'{price_type}_price_rolling_std_{window}d'] = recent_prices[-window:].std() if len(recent_prices[-window:]) > 1 else 0
                    else:
                        input_df[f'{price_type}_price_rolling_mean_{window}d'] = self.global_medians.get(f'{price_type}_price_rolling_mean_{window}d', 0)
                        input_df[f'{price_type}_price_rolling_std_{window}d'] = self.global_medians.get(f'{price_type}_price_rolling_std_{window}d', 0)
                
                for lag in [1, 3, 7, 14]:
                    if len(recent_prices) >= lag:
                        input_df[f'{price_type}_price_lag_{lag}d'] = recent_prices[-lag]
                    else:
                        input_df[f'{price_type}_price_lag_{lag}d'] = self.global_medians.get(f'{price_type}_price_lag_{lag}d', 0)
        else:
            print("Warning: No historical data available. Using global medians for all price-based features.")
            for price_col in self.price_columns:
                price_type = price_col.split()[0].lower()
                for col in input_df.columns:
                    if price_type in col and ('rolling' in col or 'lag' in col):
                        input_df[col] = self.global_medians.get(col, 0)
        
        X_new = input_df[self.feature_columns].copy()
        
        for col, le in self.label_encoders.items():
            if col in X_new.columns:
                try:
                    original_value = X_new[col].iloc[0]
                    encoded_value = le.transform(X_new[col].astype(str))[0]
                    X_new[col] = encoded_value
                    print(f"  Encoded {col}: '{original_value}' -> {encoded_value}")
                except ValueError as e:
                    print(f"  WARNING: {col} value '{X_new[col].iloc[0]}' not found in training data. Using default encoding (0).")
                    print(f"  Available {col} values: {list(le.classes_)[:5]}... (total: {len(le.classes_)})")
                    X_new[col] = 0
        
        # Debug: Print date-related feature values
        print("\nFeature values for prediction:")
        date_features = ['Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'WeekOfYear', 'DayOfYear', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']
        for feature in date_features:
            if feature in X_new.columns:
                print(f"  {feature}: {X_new[feature].iloc[0]}")
        
        # Ensure no NaN values
        X_new.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in X_new.columns:
            X_new[col] = X_new[col].fillna(self.global_medians.get(col, 0))
        
        numeric_cols = [col for col in X_new.columns if col not in self.categorical_features]
        X_new_scaled = X_new.copy()
        if numeric_cols:
            X_new_scaled[numeric_cols] = self.scaler.transform(X_new[numeric_cols])
        
        if self.categorical_features:
            pred_data = []
            for col in self.categorical_features:
                pred_data.append(X_new_scaled[col].values)
            pred_data.append(X_new_scaled[numeric_cols].values)
        else:
            pred_data = X_new_scaled.values
        
        predictions_scaled = self.model.predict(pred_data, verbose=0)
        predictions = {}
        for i, price_type in enumerate(['min', 'modal', 'max']):
            predictions[price_type] = self.target_scalers[price_type].inverse_transform(
                predictions_scaled[i].reshape(-1, 1)
            ).flatten()[0]
        
        # Enforce price constraints
        min_pred = predictions['min']
        modal_pred = predictions['modal']
        max_pred = predictions['max']
        if min_pred > modal_pred:
            min_pred, modal_pred = modal_pred, min_pred
        if modal_pred > max_pred:
            modal_pred, max_pred = max_pred, modal_pred
        predictions['min'] = max(min_pred, 0)
        predictions['modal'] = max(modal_pred, 0)
        predictions['max'] = max(max_pred, 0)
        
        return predictions
    
    def _normalize_commodity(self, commodity):
        """Normalize commodity name to match training data format"""
        if not hasattr(self, 'label_encoders') or 'Commodity' not in self.label_encoders:
            return commodity
            
        available_commodities = list(self.label_encoders['Commodity'].classes_)
        
        for available in available_commodities:
            if commodity.lower() == available.lower():
                return available
        
        commodity_lower = commodity.lower()
        
        if 'chilli' in commodity_lower or 'chili' in commodity_lower:
            if '5' in commodity:
                for available in available_commodities:
                    if 'chilli-5' in available.lower():
                        return available
            elif 'teja' in commodity_lower:
                for available in available_commodities:
                    if 'chilli-teja' in available.lower():
                        return available
            elif 'badigi' in commodity_lower or 'badiga' in commodity_lower:
                for available in available_commodities:
                    if 'badigi' in available.lower() or 'badiga' in available.lower():
                        return available
            for available in available_commodities:
                if 'chilli' in available.lower():
                    return available
        
        first_letter = commodity_lower[0] if commodity else 'a'
        for available in available_commodities:
            if available.lower().startswith(first_letter):
                return available
        
        return available_commodities[0] if available_commodities else commodity
    
    def save_model(self, filename='deep_learning_crop_model'):
        """Save the trained neural network and components"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Use the script directory for saving models
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, filename)
        
        self.model.save(f"{model_path}.h5")
        
        model_components = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'target_scalers': self.target_scalers,
            'feature_columns': self.feature_columns,
            'categorical_features': self.categorical_features,
            'numeric_features': self.numeric_features,
            'global_medians': self.global_medians,
            'price_columns': self.price_columns,
            'df_processed': self.df_processed
        }
        joblib.dump(model_components, f"{model_path}_components.pkl")
        print(f"Neural network model saved as: {model_path}.h5 and {model_path}_components.pkl")

    def load_model(self, filename='deep_learning_crop_model'):
        """Load the neural network and components from saved files"""
        # Use the script directory for loading models
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models')
        model_path = os.path.join(models_dir, filename)
        model_file = f"{model_path}.h5"
        components_file = f"{model_path}_components.pkl"
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not os.path.exists(components_file):
            raise FileNotFoundError(f"Components file not found: {components_file}")
        
        try:
            # Load the neural network model
            self.model = keras.models.load_model(model_file)
            print(f"Neural network model loaded from: {model_file}")
            
            # Load the components
            components = joblib.load(components_file)
            self.label_encoders = components.get('label_encoders', {})
            self.scaler = components.get('scaler', StandardScaler())
            self.target_scalers = components.get('target_scalers', {
                'min': MinMaxScaler(),
                'modal': MinMaxScaler(),
                'max': MinMaxScaler()
            })
            self.feature_columns = components.get('feature_columns', None)
            self.categorical_features = components.get('categorical_features', [])
            self.numeric_features = components.get('numeric_features', [])
            self.global_medians = components.get('global_medians', {})
            self.price_columns = components.get('price_columns', ['Min Price (Rs)', 'Modal Price (Rs)', 'Max Price (Rs)'])
            self.df_processed = components.get('df_processed', None)
            
            # If df_processed is None (from older saved models), try to load the processed data
            if self.df_processed is None:
                print("Warning: df_processed not found in saved model. Attempting to load processed data...")
                try:
                    # Try to load the processed_data.pkl file
                    processed_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data.pkl")
                    if os.path.exists(processed_data_path):
                        print(f"Loading processed data from: {processed_data_path}")
                        with open(processed_data_path, 'rb') as f:
                            self.df_processed = pickle.load(f)
                        print(f"Successfully loaded processed data with shape: {self.df_processed.shape}")
                    else:
                        print(f"Warning: Processed data file not found at {processed_data_path}")
                except Exception as e:
                    print(f"Warning: Could not load processed data: {e}")
            
            self.is_fitted = True
            print(f"Model components loaded from: {components_file}")
        except Exception as e:
            raise RuntimeError(f"Error loading model or components: {str(e)}")

def run_deep_learning_automation():
    """Run the complete deep learning automation"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for the CSV file in the script directory first
    possible_files = [
        os.path.join(script_dir, "enam price data.csv"),
        "enam price data.csv"
    ]
    
    csv_file = None
    for filename in possible_files:
        if os.path.exists(filename):
            csv_file = filename
            print(f"Found CSV file: {csv_file}")
            break
    
    if csv_file is None:
        # Try looking in current directory for any CSV files
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            csv_file = csv_files[0]
            print(f"Using CSV file from current directory: {csv_file}")
        else:
            print("No CSV file found!")
            print(f"Looked in:")
            for path in possible_files:
                print(f"  - {path}")
            print(f"  - Current directory: {os.getcwd()}")
            return
    
    predictor = DeepLearningCropPredictor()
    
    # Use script directory for model storage
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'best_deep_learning_model')
    model_exists = os.path.exists(f"{model_path}.h5") and os.path.exists(f"{model_path}_components.pkl")
    
    processed_data_path = os.path.join(script_dir, 'processed_data.pkl')
    
    if model_exists:
        print("Found existing model. Loading model...")
        try:
            predictor.load_model('best_deep_learning_model')
            # Load processed data if available
            if os.path.exists(processed_data_path):
                predictor.df_processed = pd.read_pickle(processed_data_path)
                print("Loaded processed data from processed_data.pkl")
            else:
                print("No processed data found. Loading and processing data...")
                df = predictor.load_and_prepare_data(csv_file)
                _, _, predictor.df_processed = predictor.prepare_features_for_dl(df)
                predictor.df_processed.to_pickle(processed_data_path)
                print("Saved processed data to processed_data.pkl")
        except Exception as e:
            print(f"Error loading model: {str(e)}. Training new model...")
            model_exists = False
    
    if not model_exists:
        print("Training Neural Network Model...")
        nn_results = predictor.train(csv_file, epochs=100)
        print("\nNeural network model trained! Saving model...")
        predictor.save_model('best_deep_learning_model')
        # Save processed data
        predictor.df_processed.to_pickle(processed_data_path)
        print("Saved processed data to processed_data.pkl")
    
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    apmcs = predictor.df_processed['APMC'].unique()[:2]
    commodities = predictor.df_processed['Commodity'].unique()[:2]
    
    for apmc in apmcs:
        for commodity in commodities:
            try:
                future_date = datetime.now() + timedelta(days=30)
                predictions = predictor.predict(apmc, commodity, future_date)
                print(f"\n{apmc} - {commodity} (30 days ahead):")
                print(f"  Min Price: Rs.{predictions['min']:,.2f}")
                print(f"  Modal Price: Rs.{predictions['modal']:,.2f}")
                print(f"  Max Price: Rs.{predictions['max']:,.2f}")
            except Exception as e:
                print(f"{apmc} - {commodity}: Error - {str(e)}")
    
    print("\n" + "="*50)
    print("INTERACTIVE PREDICTION MODE")
    print("="*50)
    print("Enter APMC, Commodity, and Date (YYYY-MM-DD) to get price predictions.")
    print("Type 'quit' for APMC to exit.")
    print()
    
    while True:
        apmc = input("Enter APMC: ").strip()
        if apmc.lower() == 'quit':
            break
        
        commodity = input("Enter Commodity: ").strip()
        if not commodity:
            print("Commodity cannot be empty. Please try again.")
            continue
        
        date_str = input("Enter Date (YYYY-MM-DD): ").strip()
        if not date_str:
            print("Date cannot be empty. Please try again.")
            continue
        
        try:
            date = pd.to_datetime(date_str)
            predictions = predictor.predict(apmc, commodity, date)
            print(f"\nPredictions for {apmc} - {commodity} on {date.strftime('%Y-%m-%d')}:")
            print(f"  Min Price: Rs.{predictions['min']:,.2f}")
            print(f"  Modal Price: Rs.{predictions['modal']:,.2f}")
            print(f"  Max Price: Rs.{predictions['max']:,.2f}")
            print()
        except ValueError as e:
            print(f"Error: Invalid date format. Please use YYYY-MM-DD. Details: {str(e)}")
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            print("Possible reasons: APMC or Commodity not in training data, or insufficient historical data.")
        print()
    
    return predictor

if __name__ == "__main__":
    print("Starting Deep Learning Crop Price Prediction...")
    run_deep_learning_automation()
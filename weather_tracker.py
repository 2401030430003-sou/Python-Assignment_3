import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle
from datetime import datetime, timedelta

class WeatherPredictionSystem:
    def __init__(self, data_folder="weather_data"):
        """Initialize the weather prediction system"""
        self.data = None
        self.models = {}
        self.features = []
        self.target_columns = []
        self.data_folder = data_folder
        
        # Create necessary folders
        for folder in [self.data_folder, f"{self.data_folder}/models", f"{self.data_folder}/reports"]:
            os.makedirs(folder, exist_ok=True)
    
    def load_data(self, file_path):
        """Load weather data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def clean_data(self):
        """Clean data by handling missing values and add date features"""
        if self.data is None:
            print("No data loaded.")
            return False
        
        # Handle missing values - numeric columns with median, categorical with mode
        for col in self.data.columns:
            if self.data[col].isna().sum() > 0:
                if self.data[col].dtype in [np.float64, np.int64]:
                    self.data[col] = self.data[col].fillna(self.data[col].median())
                elif col != 'date':
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        
        # Add time-based features
        if 'date' in self.data.columns:
            self.data['day_of_year'] = self.data['date'].dt.dayofyear
            self.data['month'] = self.data['date'].dt.month
            self.data['day'] = self.data['date'].dt.day
            self.data['day_of_week'] = self.data['date'].dt.dayofweek
            
            # Add season (1=Spring, 2=Summer, 3=Fall, 4=Winter)
            self.data['season'] = pd.cut(
                self.data['month'],
                bins=[0, 3, 6, 9, 12],
                labels=[4, 1, 2, 3],
                include_lowest=True
            )
        
        print("Data cleaning completed")
        return True
    
    def set_features_and_targets(self, features, targets):
        """Set feature columns and target columns for prediction"""
        # Validate columns exist in dataframe
        all_cols = features + targets
        missing_cols = [col for col in all_cols if col not in self.data.columns]
        if missing_cols:
            print(f"Error: Columns not found: {missing_cols}")
            return False
        
        self.features = features
        self.target_columns = targets
        print(f"Features: {features}")
        print(f"Targets: {targets}")
        return True
    
    def train_models(self, test_size=0.2, random_state=42, n_estimators=100):
        """Train a separate model for each target variable"""
        if not self.features or not self.target_columns:
            print("Features and targets not set.")
            return False
        
        X = self.data[self.features]
        model_performance = {}
        
        for target in self.target_columns:
            print(f"Training model for target: {target}")
            y = self.data[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train model
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Store model and metrics
            self.models[target] = {
                'model': model,
                'y_test': y_test,
                'y_pred': y_pred,
                'metrics': {'rmse': rmse, 'r2': r2}
            }
            
            model_performance[target] = {'rmse': float(rmse), 'r2': float(r2)}
            print(f"Model for {target} - RMSE: {rmse:.2f}, R²: {r2:.2f}")
            
            # Save model
            model_path = f"{self.data_folder}/models/model_{target}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        return model_performance
    
    def cross_validate_models(self, cv=5):
        """Perform cross-validation to better evaluate model performance"""
        if not self.features or not self.target_columns:
            print("Features and targets not set.")
            return False
        
        X = self.data[self.features]
        cv_results = {}
        
        for target in self.target_columns:
            y = self.data[target]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Perform cross-validation
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
            
            cv_results[target] = {
                'mean_rmse': float(rmse_scores.mean()),
                'std_rmse': float(rmse_scores.std())
            }
            
            print(f"CV for {target} - Mean RMSE: {rmse_scores.mean():.2f}")
        
        return cv_results
    
    def load_model(self, model_path, target):
        """Load a previously saved model"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.models[target] = {'model': model, 'metrics': {}}
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, input_data):
        """Make predictions using trained models"""
        if not self.models:
            print("No trained models available.")
            return None
        
        results = {}
        for target, model_info in self.models.items():
            predictions = model_info['model'].predict(input_data[self.features])
            results[target] = predictions
        
        return pd.DataFrame(results)
    
    def generate_future_dates(self, start_date, num_days):
        """Generate dataframe with future dates for prediction"""
        start_date = pd.to_datetime(start_date)
        future_dates = [start_date + timedelta(days=i) for i in range(num_days)]
        
        # Create DataFrame with date features
        future_df = pd.DataFrame({
            'date': future_dates,
            'day_of_year': [d.dayofyear for d in future_dates],
            'month': [d.month for d in future_dates],
            'day': [d.day for d in future_dates],
            'day_of_week': [d.dayofweek for d in future_dates]
        })
        
        # Add season
        future_df['season'] = pd.cut(
            future_df['month'],
            bins=[0, 3, 6, 9, 12],
            labels=[4, 1, 2, 3],
            include_lowest=True
        )
        
        # Add missing features with median values from training data
        for feature in self.features:
            if feature not in future_df.columns:
                future_df[feature] = self.data[feature].median()
        
        return future_df
    
    def forecast_weather(self, start_date, num_days):
        """Generate weather forecast for future dates"""
        if not self.models:
            print("No trained models available.")
            return None
        
        # Generate future date features
        future_data = self.generate_future_dates(start_date, num_days)
        
        # Make predictions
        predictions = self.predict(future_data)
        if predictions is None:
            return None
        
        # Add date column
        predictions['date'] = future_data['date']
        
        # Save forecast
        forecast_path = f"{self.data_folder}/reports/forecast.csv"
        predictions.to_csv(forecast_path, index=False)
        print(f"Forecast saved to {forecast_path}")
        
        # Visualize forecast
        self.plot_forecast(predictions)
        
        return predictions
    
    def plot_forecast(self, forecast_data):
        """Plot forecast for each target variable"""
        for target in self.target_columns:
            plt.figure(figsize=(10, 5))
            plt.plot(forecast_data['date'], forecast_data[target], marker='o', linestyle='-')
            plt.title(f'Forecast: {target}')
            plt.xlabel('Date')
            plt.ylabel(target)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{self.data_folder}/reports/forecast_{target}.png")
            plt.close()
    
    def create_sample_weather_data(self, num_days=365):
        """Create sample weather data for demonstration"""
        np.random.seed(42)
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)
        dates = [start_date + timedelta(days=i) for i in range(num_days)]
        
        # Generate seasonal temperature pattern
        day_of_year = [d.timetuple().tm_yday for d in dates]
        base_temp = 20 + 15 * np.sin(np.array(day_of_year) * 2 * np.pi / 365)
        
        # Add random variations
        temperature = base_temp + np.random.normal(0, 3, num_days)
        humidity = 65 + 15 * np.cos(np.array(day_of_year) * 2 * np.pi / 365) + np.random.normal(0, 8, num_days)
        wind_speed = 10 + np.random.normal(0, 5, num_days)
        precipitation = np.clip(np.random.exponential(1, num_days) * np.sin(np.array(day_of_year) * 2 * np.pi / 365) ** 2, 0, 50)
        
        # Create DataFrame
        weather_data = pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'precipitation': precipitation
        })
        
        return weather_data

def run_weather_prediction():
    # Create system and sample data
    wps = WeatherPredictionSystem()
    sample_data = wps.create_sample_weather_data()
    sample_data.to_csv("weather_data/sample_weather.csv", index=False)
    
    # Process data
    wps.load_data("weather_data/sample_weather.csv")
    wps.clean_data()
    
    # Define features and targets
    features = ['day_of_year', 'month', 'day', 'day_of_week', 'season']
    targets = ['temperature', 'humidity', 'wind_speed', 'precipitation']
    wps.set_features_and_targets(features, targets)
    
    # Train and validate models
    wps.train_models()
    wps.cross_validate_models()
    
    # Generate forecast
    today = datetime.now().strftime('%Y-%m-%d')
    forecast = wps.forecast_weather(today, 14)
    
    print(f"Weather prediction completed! Data and results in: {wps.data_folder}")
    return wps, forecast

if __name__ == "__main__":
    run_weather_prediction()
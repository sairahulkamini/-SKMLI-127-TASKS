import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder 
data = pd.read_csv('ride_requests.csv')
data['hour'] = pd.to_datetime(data['datetime']).dt.hour 
data['day_of_week'] = pd.to_datetime(data['datetime']).dt.dayofweek
data = data.drop(columns=['datetime'])
encoder = OneHotEncoder(handle_unknown='ignore') 
weather_encoded = encoder.fit_transform(data[['weather']]).toarray()
weather_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['weather']))
data = pd.concat([data, weather_df], axis=1).drop('weather', axis=1) 
X = data.drop(columns=['ride_requests'])
y = data['ride_requests']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R-squared: {r2}')
sample = pd.DataFrame({
    'hour': [14],
    'day_of_week': [3],
    'weather': ['Sunny']  
})
sample_encoded = encoder.transform(sample[['weather']]).toarray()
sample_weather_df = pd.DataFrame(sample_encoded, columns=encoder.get_feature_names_out(['weather']))
sample = pd.concat([sample.drop('weather', axis=1), sample_weather_df], axis=1)
prediction = model.predict(sample)
print(f'Predicted ride requests: {prediction[0]}')
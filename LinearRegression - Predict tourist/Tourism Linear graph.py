import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('tourist_attractions.csv')

# Select a location for forecasting
location = 'Thailand'

# Create a new dataframe for the selected location
df = data.loc[data['location'] == location]

# Define the predictor variable (X) and the target variable (y)
X = df[['visitors']].values
y = df[['date', 'visitors']].set_index('date').diff().fillna(0).values

# Train the model
model = LinearRegression()
model.fit(X[:-1], y[1:])

# Predict the number of visitors for the next day
predicted_visitors = model.predict(X[-1:])

# Get the location with the highest number of visitors
location_with_highest_visitors = data.groupby("location")["visitors"].mean().idxmax()

# Print the predicted number of visitors and the location with the highest number of visitors
print("Predicted visitors for {}: {}".format(location, int(predicted_visitors)))
print("Location with the highest number of visitors: {}".format(location_with_highest_visitors))

# Plot the data
plt.plot(df['date'], df['visitors'])
plt.xlabel('Date')
plt.ylabel('Visitors')
plt.title(location)
plt.show()
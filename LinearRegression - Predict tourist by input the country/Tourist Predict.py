import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from CSV file
data = pd.read_csv('tourist_attractions.csv')

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Extract the month from the 'date' column and create a new column called 'month'
data['month'] = data['date'].dt.month

# Group the data by location and month, and calculate the mean visitors
grouped_data = data.groupby(['location', 'month'])['visitors'].mean().reset_index()

# Create a list of unique locations
locations = grouped_data['location'].unique()

# Create a dictionary to store the predicted number of visitors for each location
predicted_visitors = {}


# Iterate over each location
def IterPrintAll():
    for location in locations:
        # Filter the data for the current location
        loc_data = grouped_data[grouped_data['location'] == location]

        # Split the data into training and testing sets
        X_train = loc_data['month'].values.reshape(-1, 1)
        y_train = loc_data['visitors'].values.reshape(-1, 1)
        X_test = [[13]]  # Predict for next month

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the number of visitors for the next month
        predicted_visitors[location] = model.predict(X_test)[0][0]

        # Plot the actual and predicted visitors for the current location
        plt.plot(loc_data['month'], loc_data['visitors'], label='Actual')
        plt.plot([13], predicted_visitors[location], 'o', label='Predicted')
        plt.title(location)
        plt.xlabel('Month')
        plt.ylabel('Visitors')
        plt.legend()
        plt.show()


# print each location
def print_each():
    location = input("Location (Country): ")
    # Filter the data for the current location
    loc_data = grouped_data[grouped_data['location'] == location]

    # Split the data into training and testing sets
    X_train = loc_data['month'].values.reshape(-1, 1)
    y_train = loc_data['visitors'].values.reshape(-1, 1)
    X_test = [[13]]  # Predict for next month

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the number of visitors for the next month
    predicted_visitors[location] = model.predict(X_test)[0][0]

    # Plot the actual and predicted visitors for the current location
    plt.plot(loc_data['month'], loc_data['visitors'], label='Actual')
    plt.plot([13], predicted_visitors[location], 'o', label='Predicted')
    plt.title(location)
    plt.xlabel('Month')
    plt.ylabel('Visitors')
    plt.legend()
    plt.show()

# Print the predicted number of visitors for each location
for location in predicted_visitors:
    print(f"Predicted visitors for next month at {location}: {predicted_visitors[location]}")

# IterPrintAll()
print_each()

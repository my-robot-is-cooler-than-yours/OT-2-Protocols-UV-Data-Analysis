import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from prophet import Prophet

# Load data
df = pd.read_csv(r"C:\Users\Lachlan Alexander\Downloads\ANZ.csv",
                 delimiter=",",
                 parse_dates=["Date"],
                 dayfirst=True)

df_2 = pd.read_csv(r"C:\Users\Lachlan Alexander\Downloads\ANZ (1).csv",
                   delimiter=",",
                   parse_dates=["Date"],
                   dayfirst=True)

total = pd.concat([df, df_2], ignore_index=True, axis=0)

# Aggregate daily spending (sum of transactions by day)
daily_spending = total.groupby("Date")["Amount"].sum().reset_index()
daily_spending.sort_values("Date", inplace=True)

daily_spending = daily_spending[daily_spending["Amount"] < 0]

daily_spending["Amount"] = abs(daily_spending["Amount"])

# Reset index for convenience (optional)
daily_spending.reset_index(drop=True, inplace=True)

# Prepare data for Prophet
daily_spending.rename(columns={"Date": "ds", "Amount": "y"}, inplace=True)

# Initialize and fit the Prophet model
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.fit(daily_spending)

# Make predictions (e.g., for the next 30 days)
future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)

# Plot the forecast
model.plot(forecast)

total = pd.concat([df, df_2], ignore_index=True, axis=0)

# Define keyword-based rules for known categories
keyword_map = {
    "Groceries": ["coles", "woolworths", "aldi", "toscanos", "victoria gardens"],
    "Shopping": ["bunnings", "kmart", "amazon", "book"],
    "Entertainment": ["netflix", "spotify", "cinema", "hoyts", "apple", "binge", "disney"],
    "Food & Drink": ["mcdonalds", "bagels", "takeaway", "uber eats", "dumpling", "grafalis", "cinque lire", "pies", "monash", "wholefoods"],
    "Transport": ["myki", "uber", "taxi", "bus", "train"],
    "Transfers": ["tfer", "transfer"],
    "Wages": ["campaignagent", "people", "salary"]
}

# Function to assign category based on keywords
def categorize_by_keyword(description):
    for category, keywords in keyword_map.items():
        if any(keyword in description for keyword in keywords):
            return category
    return "Uncategorized"  # Default category if no keyword matches

# Apply keyword-based categorization
total["Description"] = total["Description"].str.lower().str.replace('[^\w\s]', '')
total["Category"] = total["Description"].apply(categorize_by_keyword)

# Separate data into known categories and remaining (uncategorized) descriptions
known_categories = total[total["Category"] != "Uncategorized"]
uncategorized = total[total["Category"] == "Uncategorized"]

# Vectorize descriptions of uncategorized transactions and apply KMeans clustering
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(uncategorized["Description"])

# Apply KMeans clustering
num_clusters = 3  # Adjust this based on remaining variety in uncategorized data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
uncategorized.loc[:, "Category"] = kmeans.fit_predict(X)

# Map cluster labels to meaningful category names (adjust after examining clusters)
cluster_names = {0: "Miscellaneous"}  # Extend if needed
uncategorized.loc[:, "Category"] = uncategorized["Category"].map(cluster_names)

# Combine categorized data
total = pd.concat([known_categories, uncategorized], ignore_index=True)

# Group by date and category, then calculate daily expenditure by category
categorized_spending = total.groupby(["Date", "Category"])["Amount"].sum().unstack().fillna(0)

# Add a 'Total' column to show daily total expenditure across all categories
categorized_spending["Total"] = categorized_spending.sum(axis=1)

# Plot the daily total expenditure over time
plt.figure(figsize=(12, 6))
plt.plot(categorized_spending.index, categorized_spending["Total"], label="Daily Total Expenditure", color="black")
plt.title("Daily Total Expenditure Over Time")
plt.xlabel("Date")
plt.ylabel("Amount ($)")
plt.legend()
plt.show()

# Plot trends by category over time
categorized_spending.drop(columns="Total").plot(kind="line", figsize=(12, 6))
plt.title("Spending Trends by Category")
plt.ylabel("Amount ($)")
plt.xlabel("Date")
plt.show()

# Aggregate total spending by category for bar chart
category_spending = total.groupby("Category")["Amount"].sum().abs()

# Plot as a bar chart
plt.figure(figsize=(10, 6))
category_spending.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Total Spending by Category")
plt.xlabel("Category")
plt.ylabel("Total Spending ($)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()  # Adjust layout for better fit
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Define keyword-based rules for known categories
keyword_map = {
    "Groceries": ["coles", "woolworths", "aldi", "toscanos", "victoria gardens"],
    "Shopping": ["bunnings", "kmart", "amazon", "book"],
    "Entertainment": ["netflix", "spotify", "cinema", "hoyts", "apple", "binge", "disney"],
    "Food & Drink": ["mcdonalds", "bagels", "takeaway", "uber eats", "dumpling", "grafalis", "cinque lire", "pies", "monash", "wholefoods"],
    "Transport": ["myki", "uber", "taxi", "bus", "train"],
    "Transfers": ["tfer", "transfer"],
    "Wages": ["campaignagent", "people2.0", "salary", "pay"]
}

# Function to assign category based on keywords
def categorize_by_keyword(description):
    for category, keywords in keyword_map.items():
        if any(keyword in description for keyword in keywords):
            return category
    return "Uncategorized"

# Apply keyword-based categorization
total["Description"] = total["Description"].str.lower().str.replace('[^\w\s]', '')
total["Category"] = total["Description"].apply(categorize_by_keyword)

# Separate data into known categories and remaining (uncategorized) descriptions
known_categories = total[total["Category"] != "Uncategorized"]
uncategorized = total[total["Category"] == "Uncategorized"]

# Vectorize descriptions of uncategorized transactions and apply KMeans clustering
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(uncategorized["Description"])

# Apply KMeans clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
uncategorized.loc[:, "Category"] = kmeans.fit_predict(X)

# Map cluster labels to meaningful category names
cluster_names = {0: "Miscellaneous"}
uncategorized.loc[:, "Category"] = uncategorized["Category"].map(cluster_names)

# Combine categorized data
total = pd.concat([known_categories, uncategorized], ignore_index=True)

# Convert Date to datetime format for further processing
total['Date'] = pd.to_datetime(total['Date'])

# Feature Engineering: Create additional features (e.g., day of the week, month)
total['Day'] = total['Date'].dt.day
total['Month'] = total['Date'].dt.month
total['Year'] = total['Date'].dt.year
total['Weekday'] = total['Date'].dt.weekday

# One-hot encoding for the Category and other categorical features
encoder = OneHotEncoder()
category_encoded = encoder.fit_transform(total[['Category']])

# Combine features into a single DataFrame
features = pd.DataFrame(category_encoded.toarray(), columns=encoder.get_feature_names_out())
features['Amount'] = total['Amount']
features['Day'] = total['Day']
features['Month'] = total['Month']
features['Year'] = total['Year']
features['Weekday'] = total['Weekday']

# Define target variable (Amount) and feature set
X = features.drop('Amount', axis=1)
y = features['Amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate predictions (you can use metrics such as R^2, MAE, etc.)
print(f"Predictions: {y_pred}")

# Future Predictions Example
# Prepare a sample input for future predictions (adjust this as needed)
future_dates = pd.date_range(start='2024-11-01', periods=30)  # Next 30 days
future_categories = ['Groceries', 'Entertainment', 'Food & Drink']

future_data = []
for date in future_dates:
    for category in future_categories:
        future_data.append({
            'Category': category,
            'Day': date.day,
            'Month': date.month,
            'Year': date.year,
            'Weekday': date.weekday()
        })

# Create a DataFrame for future input
future_df = pd.DataFrame(future_data)

# One-hot encode future categories
future_encoded = encoder.transform(future_df[['Category']])
future_features = pd.DataFrame(future_encoded.toarray(), columns=encoder.get_feature_names_out())
future_features['Day'] = future_df['Day']
future_features['Month'] = future_df['Month']
future_features['Year'] = future_df['Year']
future_features['Weekday'] = future_df['Weekday']

# Predict future spending
future_predictions = model.predict(future_features)

# Output future predictions
predictions_df = pd.DataFrame({
    'Date': future_dates.repeat(len(future_categories)),
    'Category': future_categories * len(future_dates),
    'Predicted Spending': future_predictions
})

print(predictions_df)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
# Use a smaller sample if the dataset is huge to speed up this demo training
# creditcard.csv is typically ~280k rows. 
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

# Preprocessing
# Scale Time and Amount, as V1-V28 are already PCA transformed
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

X = df.drop('Class', axis=1)
y = df['Class']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
# Using Random Forest with class_weight='balanced' to handle class imbalance
# Limiting depth and estimators for faster execution in this demo
print("Training model...")
model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
print(f"Model Accuracy: {model.score(X_test, y_test)}")

# Save Model and Scaler
print("Saving model and scaler...")
joblib.dump(model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl') # Note: This scaler is fit on Amount and Time only if we did it column-wise, 
# but here we replaced columns. To properly scale new inputs, we need to know which columns to scale.
# Actually, the standard scaler fits on the columns provided. 
# Let's refit a scaler on the whole X structure or handle it carefully in the app.
# Better approach for app consistency:
# Create a pipeline or just save the scaler that was fit on the specific columns?
# The scaler above was fit on single columns separately. 
# Let's re-do scaling to be cleaner for the app.

# Re-loading to be clean
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# We need to scale Time and Amount. 
# In the app, we will receive all features. V1-V28 are generally fine as is.
# Let's creating a ColumnTransformer or just manually scale in the app using saved scalers.
# For simplicity, let's train a scaler on Time and Amount specifically.

scaler_amount = StandardScaler()
scaler_time = StandardScaler()

X['Amount'] = scaler_amount.fit_transform(X[['Amount']])
X['Time'] = scaler_time.fit_transform(X[['Time']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print(f"Model Accuracy: {model.score(X_test, y_test)}")

joblib.dump(model, 'fraud_detection_model.pkl')
joblib.dump(scaler_amount, 'scaler_amount.pkl')
joblib.dump(scaler_time, 'scaler_time.pkl')

print("Done.")

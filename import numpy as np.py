print("KATHIRAVAN-24BAD057")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
bottle_df = pd.read_csv(r"C:\Users\kathiravan\OneDrive\Desktop\ML\archive (4)\bottle.csv")
cast_df = pd.read_csv(r"C:\Users\kathiravan\OneDrive\Desktop\ML\archive (4)\cast.csv")
merged_df = pd.merge(bottle_df, cast_df, on="Cst_Cnt", how="inner")
lat = [c for c in merged_df.columns if 'lat' in c.lower()]
lon = [c for c in merged_df.columns if 'lon' in c.lower()]
features = ['Depthm', 'Salnty', 'O2ml_L']
if lat:
    features.append(lat[0])
if lon:
    features.append(lon[0])
target = 'T_degC'
data = merged_df[features + [target]]
data = data.fillna(data.mean(numeric_only=True))
X = StandardScaler().fit_transform(data[features])
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print("\nLinear Regression Performance")
print("------------------------------")
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.savefig(r"C:\Users\kathiravan\OneDrive\Desktop\ML\actual_vs_predicted.png",
            dpi=300, bbox_inches='tight')
plt.show()
plt.figure(figsize=(6,4))
plt.scatter(y_pred, y_test - y_pred)
plt.axhline(0)
plt.xlabel("Predicted Temperature")
plt.ylabel("Residual Error")
plt.title("Residual Error Plot")
plt.savefig(r"C:\Users\kathiravan\OneDrive\Desktop\ML\residual_plot.png",
            dpi=300, bbox_inches='tight')
plt.show()
ridge_model = Ridge()
ridge_pred = ridge_model.fit(X_train, y_train).predict(X_test)
print("Ridge R2 Score:", r2_score(y_test, ridge_pred))
lasso_model = Lasso(alpha=0.01)
lasso_pred = lasso_model.fit(X_train, y_train).predict(X_test)
print("Lasso R2 Score:", r2_score(y_test, lasso_pred))
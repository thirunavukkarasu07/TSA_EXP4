# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 28.10.2025



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:

```
# --- Import Required Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- Load the dataset ---
data = pd.read_csv('housing_price_dataset.csv')

# --- Prepare yearly average price data ---
# Group by YearBuilt (acts like a time index)
data_yearly = data.groupby('YearBuilt')['Price'].mean().sort_index()
X = data_yearly.values  # time series values

# --- Basic visualization ---
plt.rcParams['figure.figsize'] = [12, 6]
plt.plot(data_yearly.index, X, color='blue')
plt.title('Yearly Average Housing Prices')
plt.xlabel('Year Built')
plt.ylabel('Average Price')
plt.grid(True)
plt.show()

# --- Plot ACF and PACF of Original Data ---
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(X, lags=int(len(X)/2), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=int(len(X)/2), ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

# ==========================
# --- Fit ARMA(1,1) Model ---
# ==========================
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

print("\n--- ARMA(1,1) Model Parameters ---")
print(arma11_model.params)

# Safely extract parameters by index (version-safe)
phi1_arma11 = arma11_model.params[1] if len(arma11_model.params) > 1 else 0
theta1_arma11 = arma11_model.params[2] if len(arma11_model.params) > 2 else 0

print(f"\nExtracted Parameters:\n  φ₁ = {phi1_arma11:.4f}, θ₁ = {theta1_arma11:.4f}")

# --- Simulate ARMA(1,1) Process ---
N = 1000
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.figure(figsize=(12, 5))
plt.plot(ARMA_1, color='purple')
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.grid(True)
plt.show()

# --- Plot ACF and PACF for ARMA(1,1) ---
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(ARMA_1, ax=plt.gca())
plt.title('ARMA(1,1) Simulated Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(ARMA_1, ax=plt.gca())
plt.title('ARMA(1,1) Simulated Data PACF')
plt.tight_layout()
plt.show()

# ==========================
# --- Fit ARMA(2,2) Model ---
# ==========================
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

print("\n--- ARMA(2,2) Model Parameters ---")
print(arma22_model.params)

# Safely extract parameters
phi1_arma22 = arma22_model.params[1] if len(arma22_model.params) > 1 else 0
phi2_arma22 = arma22_model.params[2] if len(arma22_model.params) > 2 else 0
theta1_arma22 = arma22_model.params[3] if len(arma22_model.params) > 3 else 0
theta2_arma22 = arma22_model.params[4] if len(arma22_model.params) > 4 else 0

print(f"\nExtracted Parameters:\n  φ₁ = {phi1_arma22:.4f}, φ₂ = {phi2_arma22:.4f}, "
      f"θ₁ = {theta1_arma22:.4f}, θ₂ = {theta2_arma22:.4f}")

# --- Simulate ARMA(2,2) Process ---
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.figure(figsize=(12, 5))
plt.plot(ARMA_2, color='orange')
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.grid(True)
plt.show()

# --- Plot ACF and PACF for ARMA(2,2) ---
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(ARMA_2, ax=plt.gca())
plt.title('ARMA(2,2) Simulated Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(ARMA_2, ax=plt.gca())
plt.title('ARMA(2,2) Simulated Data PACF')
plt.tight_layout()
plt.show()
```

OUTPUT:

<img width="1056" height="552" alt="image" src="https://github.com/user-attachments/assets/c0814c93-b117-4289-a9c8-59d2a921b23f" />

<img width="1213" height="297" alt="image" src="https://github.com/user-attachments/assets/8b63a551-0516-4a7c-9892-3514b6dc6ed5" />

<img width="1226" height="317" alt="image" src="https://github.com/user-attachments/assets/420aa449-c48c-4aea-906e-d2b3875535d9" />


SIMULATED ARMA(1,1) PROCESS:

<img width="1011" height="540" alt="image" src="https://github.com/user-attachments/assets/0afadcb3-a712-4f9e-addd-a492fa8e8e83" />



<img width="1213" height="601" alt="image" src="https://github.com/user-attachments/assets/8c4ad4cd-70e2-4db7-982a-32229e52ef82" />




SIMULATED ARMA(2,2) PROCESS:

<img width="1001" height="563" alt="image" src="https://github.com/user-attachments/assets/466b907b-71e6-4363-86e2-7c67d48bf0c7" />

<img width="1209" height="616" alt="image" src="https://github.com/user-attachments/assets/54a60713-97f7-47e3-8c60-6e52ff805b94" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.

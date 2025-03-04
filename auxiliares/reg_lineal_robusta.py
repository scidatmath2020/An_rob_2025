# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 19:25:18 2025

@author: Usuario
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
from plotnine import *
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt

import os
#%%
# Cargar datos
cisne = pd.read_csv("cisne.csv")


#%%

# Variables
X = sm.add_constant(cisne["log.Te"])
y = cisne["log.light"]

# OLS (Regresión Lineal Ordinaria)
ols_model = sm.OLS(y, X).fit()
ols_coef = ols_model.params

# LAD 
lad_model = QuantReg(y, X).fit(q=0.5)
lad_coef = lad_model.params


#%%

# Variables
X = cisne["log.Te"].values.reshape(-1, 1)  # Asegurar que sea 2D
y = cisne["log.light"].values

# Agregar constante
X_const = sm.add_constant(X)

# -------- OLS --------
ols_model = sm.OLS(y, X_const).fit()
ols_coef = ols_model.params

# -------- LMS (Least Median of Squares) --------
n = len(y)
num_subsets = 500  # Número de combinaciones aleatorias de muestras
best_median_residual = np.inf
best_model = None

for _ in range(num_subsets):
    subset = np.random.choice(n, size=int(n * 0.5), replace=False)
    X_subset = X_const[subset]
    y_subset = y[subset]
    
    model = sm.OLS(y_subset, X_subset).fit()
    residuals = np.abs(y - model.predict(X_const))
    median_residual = np.median(residuals)
    
    if median_residual < best_median_residual:
        best_median_residual = median_residual
        best_model = model

lms_coef = best_model.params if best_model else np.array([np.nan, np.nan])


#%%

# -------- Crear DataFrame con coeficientes --------
df_coef = pd.DataFrame({
    "Model": ["OLS", "LMS", "LAD"],
    "Intercept": [ols_coef[0], lms_coef[0], lad_coef[0]],
    "Slope": [ols_coef[1], lms_coef[1], lad_coef[1]]
})


(
    ggplot(cisne, aes(x="log.Te", y="log.light")) +
    geom_point() +
    geom_abline(df_coef, aes(slope="Slope", intercept="Intercept", color="Model"))
)

#%%

# Gráfica con Matplotlib
plt.figure(figsize=(8, 6))
plt.scatter(cisne["log.Te"], cisne["log.light"], label="Data", alpha=0.7)

colors = {"OLS": "blue", "LMS": "red", "LAD": "green"}
for _, row in df_coef.iterrows():
    intercept, slope = row["Intercept"], row["Slope"]
    x_vals = np.linspace(cisne["log.Te"].min(), cisne["log.Te"].max(), 100)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color=colors[row["Model"]], label=row["Model"])

plt.xlabel("log.Te")
plt.ylabel("log.light")
plt.legend()
plt.grid()
plt.show()

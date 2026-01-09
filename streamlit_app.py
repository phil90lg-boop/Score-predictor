import streamlit as st
import numpy as np
from scipy.stats import poisson

st.title("⚽ Score Predictor IA")

# Cases pour entrer les forces des équipes
h_xg = st.number_input("Force Domicile (xG moyen)", value=1.5)
a_xg = st.number_input("Force Extérieur (xG moyen)", value=1.2)

# Calcul de la probabilité
h_probs = [poisson.pmf(i, h_xg) for i in range(6)]
a_probs = [poisson.pmf(i, a_xg) for i in range(6)]
matrix = np.outer(h_probs, a_probs)

# Résultat
res_h, res_a = np.unravel_index(matrix.argmax(), matrix.shape)

st.divider()
st.header(f"Score Prédit : {res_h} - {res_a}")
st.write(f"Confiance : {np.max(matrix)*100:.1f}%")

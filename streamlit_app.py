import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Score Predictor Pro", layout="centered")
st.title("⚽ Score Predictor IA Pro")

# Entrées statistiques
col1, col2 = st.columns(2)
with col1:
    h_xg = st.number_input("xG Domicile", value=1.5, step=0.1)
with col2:
    a_xg = st.number_input("xG Extérieur", value=1.2, step=0.1)

# Calculs mathématiques
h_probs = [poisson.pmf(i, h_xg) for i in range(10)]
a_probs = [poisson.pmf(i, a_xg) for i in range(10)]
matrix = np.outer(h_probs, a_probs)

# 1. Score Exact
res_h, res_a = np.unravel_index(matrix.argmax(), matrix.shape)

# 2. Plus de 2.5 Buts
over_25 = 1 - (matrix[0,0] + matrix[0,1] + matrix[0,2] + matrix[1,0] + matrix[1,1] + matrix[2,0])

# 3. Les deux équipes marquent (BTTS)
btts = (1 - matrix[0,:].sum()) * (1 - matrix[:,0].sum())

# Affichage des résultats
st.divider()
st.subheader(f"🎯 Score prédit : {res_h} - {res_a}")
st.write(f"Confiance sur le score : {np.max(matrix)*100:.1f}%")

st.divider()
c1, c2 = st.columns(2)
c1.metric("Plus de 2.5 buts", f"{over_25*100:.1f}%")
c2.metric("Les deux équipes marquent", f"{btts*100:.1f}%")

st.info("Conseil : Si le % est > 55%, le pari est considéré comme statistiquement intéressant.")

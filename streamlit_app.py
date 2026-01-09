import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd

st.set_page_config(page_title="IA Verdict", layout="centered")
st.title("🏆 Verdict de l'IA")

# Paramètres simplifiés
st.sidebar.header("Réglages")
home_adv = st.sidebar.slider("Avantage Domicile", 1.0, 1.5, 1.10)

h_xg = st.number_input("Force Domicile (xG)", value=1.5, step=0.1)
a_xg = st.number_input("Force Extérieur (xG)", value=1.2, step=0.1)

# Calculs
l_home = h_xg * home_adv
l_away = a_xg
h_probs = [poisson.pmf(i, l_home) for i in range(7)]
a_probs = [poisson.pmf(i, l_away) for i in range(7)]
matrix = np.outer(h_probs, a_probs)

res_h, res_a = np.unravel_index(matrix.argmax(), matrix.shape)
over_25 = (1 - (matrix[0,0] + matrix[0,1] + matrix[0,2] + matrix[1,0] + matrix[1,1] + matrix[2,0])) * 100
btts = ((1 - h_probs[0]) * (1 - a_probs[0])) * 100

# Fonction pour afficher uniquement l'indice visuel
def show_verdict(label, value, safe, strong):
    if value >= strong:
        st.success(f"✅ **{label}** : CONFIANCE FORTE")
    elif value >= safe:
        st.warning(f"⚠️ **{label}** : CONFIANCE MODÉRÉE")
    else:
        st.error(f"❌ **{label}** : TROP RISQUÉ")

st.divider()
st.header(f"Pronostic : {res_h} - {res_a}")
st.divider()

# Affichage des indices de confiance uniquement
st.subheader("Indices de confiance pour ce match :")
show_verdict("Plus de 2.5 buts", over_25, 55, 65)
show_verdict("Les deux équipes marquent", btts, 58, 68)
show_verdict("Score Exact", (np.max(matrix)*100), 12, 15)

st.caption("L'IA conseille de ne jouer que les lignes avec une coche verte (✅).")

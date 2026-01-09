import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="IA Predictor Pro", layout="centered")
st.title("🏆 Aide à la Décision IA")

# --- OPTIONS DÉPLACÉES SUR L'ÉCRAN PRINCIPAL ---
st.subheader("Configuration du Match")
terrain_neutre = st.checkbox("🏟️ Ce match se joue sur terrain neutre (CAN, CDM, etc.)", value=False)

if not terrain_neutre:
    home_adv = st.slider("Avantage Domicile", 1.0, 1.5, 1.10)
else:
    home_adv = 1.0
    st.info("Mode terrain neutre activé : les deux équipes sont à égalité de lieu.")

st.divider()

# Entrées des données
col1, col2 = st.columns(2)
with col1:
    h_xg = st.number_input("Force Équipe A (xG)", value=1.5, step=0.1)
with col2:
    a_xg = st.number_input("Force Équipe B (xG)", value=1.2, step=0.1)

# Calculs
l_home = h_xg * home_adv
l_away = a_xg
h_probs = [poisson.pmf(i, l_home) for i in range(7)]
a_probs = [poisson.pmf(i, l_away) for i in range(7)]
matrix = np.outer(h_probs, a_probs)

res_h, res_a = np.unravel_index(matrix.argmax(), matrix.shape)
over_25 = (1 - (matrix[0,0] + matrix[0,1] + matrix[0,2] + matrix[1,0] + matrix[1,1] + matrix[2,0])) * 100
btts = ((1 - h_probs[0]) * (1 - a_probs[0])) * 100
prob_nul = np.trace(matrix) * 100

# Affichage des verdicts
st.divider()
st.header(f"Tendance : {res_h} - {res_a}")

def show_verdict(label, value, mini, safe, strong):
    if value >= strong: st.success(f"✅ **{label}** ({value:.1f}%) : TRÈS SOLIDE")
    elif value >= safe: st.info(f"🔵 **{label}** ({value:.1f}%) : À TENTER")
    elif value >= mini: st.warning(f"⚠️ **{label}** ({value:.1f}%) : RISQUÉ")
    else: st.error(f"❌ **{label}** ({value:.1f}%) : ÉVITER")

st.subheader("Analyses :")
show_verdict("Plus de 2.5 buts", over_25, 45, 52, 62)
show_verdict("Les deux équipes marquent", btts, 48, 55, 65)

if prob_nul > 25:
    st.info(f"⚖️ **Option Match Nul** : {prob_nul:.1f}% (Tendance forte)")

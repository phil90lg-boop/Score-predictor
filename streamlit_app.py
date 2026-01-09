import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="IA Predictor Pro", layout="centered")
st.title("🏆 Aide à la Décision IA")

# Configuration du Match
st.subheader("Configuration")
terrain_neutre = st.checkbox("🏟️ Terrain Neutre (CAN, CDM, etc.)", value=False)
home_adv = 1.0 if terrain_neutre else st.slider("Avantage Domicile", 1.0, 1.5, 1.10)

st.divider()

# Entrées des données
col1, col2 = st.columns(2)
with col1:
    h_xg = st.number_input("Force Équipe A (xG)", value=1.5, step=0.1)
with col2:
    a_xg = st.number_input("Force Équipe B (xG)", value=1.2, step=0.1)

# Calculs mathématiques
l_home = h_xg * home_adv
l_away = a_xg
h_probs = [poisson.pmf(i, l_home) for i in range(7)]
a_probs = [poisson.pmf(i, l_away) for i in range(7)]
matrix = np.outer(h_probs, a_probs)

# --- CALCULS DES PROBABILITÉS ---
prob_home = np.sum(np.tril(matrix, -1)) * 100
prob_draw = np.trace(matrix) * 100 # C'est cette ligne qui manquait !
prob_away = np.sum(np.triu(matrix, 1)) * 100

dc_1n = prob_home + prob_draw
dc_n2 = prob_away + prob_draw
over_25 = (1 - (matrix[0,0] + matrix[0,1] + matrix[0,2] + matrix[1,0] + matrix[1,1] + matrix[2,0])) * 100
btts = ((1 - h_probs[0]) * (1 - a_probs[0])) * 100

# Fonction d'affichage
def show_verdict(label, value, mini, safe, strong):
    if value >= strong: st.success(f"✅ **{label}** ({value:.1f}%) : TRÈS SOLIDE")
    elif value >= safe: st.info(f"🔵 **{label}** ({value:.1f}%) : À TENTER")
    elif value >= mini: st.warning(f"⚠️ **{label}** ({value:.1f}%) : RISQUÉ")
    else: st.error(f"❌ **{label}** ({value:.1f}%) : ÉVITER")

# --- AFFICHAGE ---
st.divider()
res_h, res_a = np.unravel_index(matrix.argmax(), matrix.shape)
st.header(f"Tendance : {res_h} - {res_a}")

st.subheader("Analyse Double Chance")
show_verdict("1N (Équipe A ou Nul)", dc_1n, 65, 75, 85)
show_verdict("N2 (Équipe B ou Nul)", dc_n2, 65, 75, 85)

st.divider()
st.subheader("Analyse des Buts")
show_verdict("Plus de 2.5 buts", over_25, 45, 52, 62)
show_verdict("Les deux marquent", btts, 48, 55, 65)

# Message pour le nul
if prob_draw > 25:
    st.info(f"⚖️ Tendance forte au Match Nul ({prob_draw:.1f}%)")

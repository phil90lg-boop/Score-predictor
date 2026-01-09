import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="IA Predictor Pro", layout="centered")
st.title("🏆 Aide à la Décision IA")

# Configuration du Match
st.subheader("Configuration du Match")
terrain_neutre = st.checkbox("🏟️ Ce match se joue sur terrain neutre (CAN, CDM, etc.)", value=False)
home_adv = 1.0 if terrain_neutre else st.slider("Avantage Domicile", 1.0, 1.5, 1.10)

st.divider()

# Entrées des données
col1, col2 = st.columns(2)
with col1:
    h_xg = st.number_input("Force Équipe A (xG)", value=1.5, step=0.1)
with col2:
    a_xg = st.number_input("Force Équipe B (xG)", value=1.2, step=0.1)

# Calculs de base
l_home = h_xg * home_adv
l_away = a_xg
h_probs = [poisson.pmf(i, l_home) for i in range(7)]
a_probs = [poisson.pmf(i, l_away) for i in range(7)]
matrix = np.outer(h_probs, a_probs)

# Calcul des probabilités de résultat (1, N, 2)
prob_home = np.sum(np.tril(matrix, -1)) * 100
prob_draw = np.trace(matrix) * 100
prob_away = np.sum(np.triu(matrix, 1)) * 100

# Calcul Double Chance
dc_1n = prob_home + prob_draw
dc_n2 = prob_away + prob_draw

# Affichage
st.divider()
res_h, res_a = np.unravel_index(matrix.argmax(), matrix.shape)
st.header(f"Tendance : {res_h} - {res_a}")

def show_verdict(label, value, mini, safe, strong):
    if value >= strong: st.success(f"✅ **{label}** ({value:.1f}%) : TRÈS SOLIDE")
    elif value >= safe: st.info(f"🔵 **{label}** ({value:.1f}%) : À TENTER")
    elif value >= mini: st.warning(f"⚠️ **{label}** ({value:.1f}%) : RISQUÉ")
    else: st.error(f"❌ **{label}** ({value:.1f}%) : ÉVITER")

st.subheader("Analyses Spéciales :")
# Seuils pour la Double Chance (généralement plus élevés car plus "facile" à valider)
show_verdict("Double Chance Équipe A (1N)", dc_1n, 65, 75, 85)
show_verdict("Double Chance Équipe B (N2)", dc_n2, 65, 75, 85)

st.divider()
st.subheader("Buts :")
over_25 = (1 - (matrix[0,0] + matrix[0,1] + matrix[0,2] + matrix[1,0] + matrix[1,1] + matrix[2,0])) * 100
show_verdict("Plus de 2.5 buts", over_25, 45, 52, 62)

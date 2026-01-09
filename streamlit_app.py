import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="IA Predictor Pro", layout="centered")
st.title("🏆 Aide au Choix IA")

# Configuration
terrain_neutre = st.checkbox("🏟️ Terrain Neutre (CAN, CDM, etc.)", value=True)
home_adv = 1.0 if terrain_neutre else 1.10

# Entrées
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

p_1 = np.sum(np.tril(matrix, -1)) * 100
p_n = np.trace(matrix) * 100
p_2 = np.sum(np.triu(matrix, 1)) * 100

# Résultats
st.divider()
res_h, res_a = np.unravel_index(matrix.argmax(), matrix.shape)
st.header(f"Score le plus probable : {res_h} - {res_a}")

# FONCTION DE VERDICT PLUS SOUPLE
def show_choice(label, value, mini, safe):
    if value >= safe:
        st.success(f"✅ **CONSEILLÉ** : {label} ({value:.1f}%)")
    elif value >= mini:
        st.info(f"🔵 **À TENTER** : {label} ({value:.1f}%)")
    else:
        st.warning(f"⚠️ **RISQUÉ** : {label} ({value:.1f}%)")

st.subheader("Verdict de l'IA pour ton choix :")

# On affiche les 3 meilleures probabilités pour t'aider à choisir
options = {
    "Double Chance 1N": p_1 + p_n,
    "Double Chance N2": p_2 + p_n,
    "Les deux marquent": ((1 - h_probs[0]) * (1 - a_probs[0])) * 100,
    "Plus de 1.5 buts": (1 - (matrix[0,0] + matrix[0,1] + matrix[1,0])) * 100
}

# Tri pour afficher le meilleur choix en premier
for label, val in sorted(options.items(), key=lambda x: x[1], reverse=True):
    show_choice(label, val, 45, 60)

st.divider()
st.subheader("Répartition des probabilités :")
st.write(f"Victoire A : **{p_1:.1f}%** | Nul : **{p_n:.1f}%** | Victoire B : **{p_2:.1f}%**")

import streamlit as st
import numpy as np
from scipy.stats import poisson
import plotly.express as px

st.set_page_config(page_title="IA Predictor Pro", layout="centered")
st.title("🏆 Aide au Choix IA")

# --- ENTRÉES ---
terrain_neutre = st.checkbox("🏟️ Terrain Neutre (CAN, CDM, etc.)", value=True)
home_adv = 1.0 if terrain_neutre else 1.10

col1, col2 = st.columns(2)
with col1:
    h_xg = st.number_input("Force Équipe A (xG)", value=1.5, step=0.1)
with col2:
    a_xg = st.number_input("Force Équipe B (xG)", value=1.2, step=0.1)

# --- CALCULS ---
l_home = h_xg * home_adv
l_away = a_xg

h_probs = [poisson.pmf(i, l_home) for i in range(11)]
a_probs = [poisson.pmf(i, l_away) for i in range(11)]
matrix = np.outer(h_probs, a_probs)

p_1 = np.sum(np.tril(matrix, -1)) * 100
p_n = np.trace(matrix) * 100
p_2 = np.sum(np.triu(matrix, 1)) * 100

def get_over_prob(matrix, threshold):
    prob_under = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (i + j) <= threshold:
                prob_under += matrix[i, j]
    return (1 - prob_under) * 100

# --- AFFICHAGE DU SCORE ---
st.divider()
res_h, res_a = np.unravel_index(matrix.argmax(), matrix.shape)
st.header(f"Score le plus probable : {res_h} - {res_a}")

# --- MATRICE GRAPHIQUE ---
st.subheader("📊 Matrice de probabilités des scores")
limit = 6 
display_matrix = matrix[:limit, :limit]
fig = px.imshow(
    display_matrix,
    labels=dict(x="Buts Équipe B", y="Buts Équipe A", color="Probabilité"),
    x=[str(i) for i in range(limit)],
    y=[str(i) for i in range(limit)],
    text_auto=".2%",
    color_continuous_scale='GnBu'
)
st.plotly_chart(fig, use_container_width=True)

# --- ANALYSE DES PARIS ---
st.subheader("Verdict de l'IA :")

# Correction : L'accolade est bien fermée ici avant de continuer
options = {
    "Plus de 1.5 buts": get_over_prob(matrix, 1),
    "Plus de 2.5 buts": get_over_prob(matrix, 2),
    "Plus de 3.5 buts": get_over_prob(matrix, 3),
    "Les deux marquent": ((1 - h_probs[0]) * (1 - a_probs[0])) * 100,
    "Double Chance 1N": p_1 + p_n,
    "Double Chance N2": p_2 + p_n,
}

for label, val in sorted(options.items(), key=lambda x: x[1], reverse=True):
    if val >= 75:
        st.success(f"✅ **CONSEILLÉ** : {label} ({val:.1f}%)")
    elif val >= 55:
        st.info(f"🔥 **À TENTER** : {label} ({val:.1f}%)")
    elif val >= 35:
        st.warning(f"⚠️ **PRUDENCE** : {label} ({val:.1f}%)")
    else:
        st.error(f"❌ **À ÉVITER** : {label} ({val:.1f}%)")

st.divider()
st.subheader("Répartition des probabilités :")
st.write(f"Victoire A : **{p_1:.1f}%** | Nul : **{p_n:.1f}%** | Victoire B : **{p_2:.1f}%**")

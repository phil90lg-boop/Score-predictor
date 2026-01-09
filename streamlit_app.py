import streamlit as st
import numpy as np
from scipy.stats import poisson
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Predictor IA Simplifié", layout="wide")
st.title("⚽ Score Predictor IA")

# Initialisation de l'historique
if 'history' not in st.session_state:
    st.session_state.history = []

# Barre latérale pour les réglages fins
st.sidebar.header("Réglages")
home_adv = st.sidebar.slider("Avantage Domicile (%)", 1.0, 1.5, 1.10)
match_name = st.sidebar.text_input("Nom du Match", "Équipe A vs Équipe B")

col1, col2 = st.columns(2)
with col1:
    h_xg = st.number_input("Force de frappe Domicile (xG)", value=1.6, step=0.1)
with col2:
    a_xg = st.number_input("Force de frappe Extérieur (xG)", value=1.2, step=0.1)

# Calcul des probabilités avec avantage domicile
l_home = h_xg * home_adv
l_away = a_xg

h_probs = [poisson.pmf(i, l_home) for i in range(7)]
a_probs = [poisson.pmf(i, l_away) for i in range(7)]
matrix = np.outer(h_probs, a_probs)

# Résultats calculés
res_h, res_a = np.unravel_index(matrix.argmax(), matrix.shape)
over_25 = (1 - (matrix[0,0] + matrix[0,1] + matrix[0,2] + matrix[1,0] + matrix[1,1] + matrix[2,0])) * 100
btts = ((1 - sum(h_probs[0:1])) * (1 - sum(a_probs[0:1]))) * 100

# Affichage principal
st.divider()
st.subheader(f"🎯 Score prédit : {res_h} - {res_a}")

c1, c2, c3 = st.columns(3)
c1.metric("Confiance", f"{np.max(matrix)*100:.1f}%")
c2.metric("Plus de 2.5 buts", f"{over_25:.1f}%")
c3.metric("Les deux marquent", f"{btts:.1f}%")

# Bouton de sauvegarde
if st.button("💾 Enregistrer dans l'historique"):
    st.session_state.history.append({
        "Match": match_name,
        "Score": f"{res_h}-{res_a}",
        "Confiance": f"{np.max(matrix)*100:.1f}%"
    })
    st.success("Pronostic enregistré !")

# Heatmap visuelle
st.subheader("🔥 Probabilités de scores")
fig = px.imshow(matrix * 100, text_auto=".1f", color_continuous_scale='Viridis',
                labels=dict(x="Extérieur", y="Domicile", color="%"),
                x=[0,1,2,3,4,5,6], y=[0,1,2,3,4,5,6])
st.plotly_chart(fig, use_container_width=True)

# Tableau d'historique
if st.session_state.history:
    st.divider()
    st.table(pd.DataFrame(st.session_state.history))

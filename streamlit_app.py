import streamlit as st
import numpy as np
from scipy.stats import poisson
import plotly.express as px

st.set_page_config(page_title="IA Score Predictor Pro", layout="wide")
st.title("🎯 IA Haute Précision + Heatmap")

# Barre latérale pour les réglages
st.sidebar.header("Paramètres du Match")
avg_goals = st.sidebar.slider("Moyenne buts championnat", 1.0, 3.5, 1.35)
home_adv = st.sidebar.slider("Avantage Domicile (%)", 1.0, 1.5, 1.10)

col1, col2 = st.columns(2)
with col1:
    h_att = st.number_input("Force Attaque Domicile", value=1.6)
    h_def = st.number_input("Faiblesse Défense Domicile", value=1.3)
with col2:
    a_att = st.number_input("Force Attaque Extérieur", value=1.1)
    a_def = st.number_input("Faiblesse Défense Extérieur", value=1.2)

# Calcul des Lambdas
l_home = (h_att * h_def) / avg_goals * home_adv
l_away = (a_att * a_def) / avg_goals

# Calcul Matrice (0 à 5 buts)
h_probs = [poisson.pmf(i, l_home) for i in range(6)]
a_probs = [poisson.pmf(i, l_away) for i in range(6)]
matrix = np.outer(h_probs, a_probs)

# Affichage des stats clés
st.divider()
res_h, res_a = np.unravel_index(matrix.argmax(), matrix.shape)
c1, c2, c3 = st.columns(3)
c1.metric("Score Prédit", f"{res_h} - {res_a}")
c2.metric("Plus de 2.5 buts", f"{(1-(matrix[0,0]+matrix[0,1]+matrix[0,2]+matrix[1,0]+matrix[1,1]+matrix[2,0]))*100:.1f}%")
c3.metric("Les deux marquent", f"{((1-matrix[0,:].sum())*(1-matrix[:,0].sum()))*100:.1f}%")

# Graphique Heatmap
st.subheader("🔥 Matrice des Probabilités de Scores")
fig = px.imshow(matrix * 100, 
                labels=dict(x="Buts Extérieur", y="Buts Domicile", color="Probabilité %"),
                x=[0,1,2,3,4,5], y=[0,1,2,3,4,5],
                text_auto=".1f", aspect="auto",
                color_continuous_scale='Viridis')
st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import numpy as np
from scipy.stats import poisson
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="IA Predictor Expert", layout="wide")
st.title("⚽ Score Predictor IA - Aide au Pari")

# Initialisation de l'historique
if 'history' not in st.session_state:
    st.session_state.history = []

# Barre latérale
st.sidebar.header("Réglages")
home_adv = st.sidebar.slider("Avantage Domicile (%)", 1.0, 1.5, 1.10)
match_name = st.sidebar.text_input("Nom du Match", "Match 1")

col1, col2 = st.columns(2)
with col1:
    h_xg = st.number_input("Force Attaque Domicile (xG)", value=1.6)
with col2:
    a_xg = st.number_input("Force Attaque Extérieur (xG)", value=1.2)

# Calculs
l_home = h_xg * home_adv
l_away = a_xg
h_probs = [poisson.pmf(i, l_home) for i in range(7)]
a_probs = [poisson.pmf(i, l_away) for i in range(7)]
matrix = np.outer(h_probs, a_probs)

res_h, res_a = np.unravel_index(matrix.argmax(), matrix.shape)
conf_score = np.max(matrix) * 100
over_25 = (1 - (matrix[0,0] + matrix[0,1] + matrix[0,2] + matrix[1,0] + matrix[1,1] + matrix[2,0])) * 100
btts = ((1 - h_probs[0]) * (1 - a_probs[0])) * 100

# --- NOUVEAU : SYSTÈME D'ALERTES ---
st.divider()
st.subheader(f"🎯 Pronostic : {res_h} - {res_a}")

def get_status(value, safe, strong):
    if value >= strong: return "🔥 Confiance Forte", "green"
    elif value >= safe: return "⚠️ Confiance Modérée", "orange"
    else: return "❌ Risqué / Peu probable", "red"

# Affichage des métriques avec couleurs
c1, c2, c3 = st.columns(3)

txt_o, col_o = get_status(over_25, 55, 65)
c1.metric("Plus de 2.5 buts", f"{over_25:.1f}%")
c1.markdown(f":{col_o}[{txt_o}]")

txt_b, col_b = get_status(btts, 58, 68)
c2.metric("Les deux marquent", f"{btts:.1f}%")
c2.markdown(f":{col_b}[{txt_b}]")

txt_s, col_s = get_status(conf_score, 12, 15)
c3.metric("Confiance Score", f"{conf_score:.1f}%")
c3.markdown(f":{col_s}[{txt_s}]")

# Sauvegarde et Heatmap (reste identique)
if st.button("💾 Enregistrer le prono"):
    st.session_state.history.append({"Match": match_name, "Prono": f"{res_h}-{res_a}", "Over 2.5": f"{over_25:.1f}%"})
    st.success("Enregistré !")

fig = px.imshow(matrix * 100, text_auto=".1f", color_continuous_scale='Viridis', x=[0,1,2,3,4,5,6], y=[0,1,2,3,4,5,6])
st.plotly_chart(fig, use_container_width=True)

if st.session_state.history:
    st.table(pd.DataFrame(st.session_state.history))

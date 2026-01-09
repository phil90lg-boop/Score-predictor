import streamlit as st
import numpy as np
from scipy.stats import poisson
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="IA Score Predictor Expert", layout="wide")
st.title("🎯 IA Predictor : Suivi des Pronos")

# Initialisation de l'historique dans la session
if 'history' not in st.session_state:
    st.session_state.history = []

# Paramètres
st.sidebar.header("Réglages")
avg_goals = st.sidebar.slider("Moyenne buts championnat", 1.0, 3.5, 1.35)
home_adv = st.sidebar.slider("Avantage Domicile", 1.0, 1.5, 1.10)
match_name = st.sidebar.text_input("Nom du Match", "Équipe A vs Équipe B")

col1, col2 = st.columns(2)
with col1:
    h_att = st.number_input("Force Attaque Dom.", value=1.6)
    h_def = st.number_input("Faiblesse Défense Dom.", value=1.3)
with col2:
    a_att = st.number_input("Force Attaque Ext.", value=1.1)
    a_def = st.number_input("Faiblesse Défense Ext.", value=1.2)

l_home = (h_att * h_def) / avg_goals * home_adv
l_away = (a_att * a_def) / avg_goals

h_probs = [poisson.pmf(i, l_home) for i in range(6)]
a_probs = [poisson.pmf(i, l_away) for i in range(6)]
matrix = np.outer(h_probs, a_probs)

res_h, res_a = np.unravel_index(matrix.argmax(), matrix.shape)
over_25 = (1 - (matrix[0,0] + matrix[0,1] + matrix[0,2] + matrix[1,0] + matrix[1,1] + matrix[2,0])) * 100

# Bouton de sauvegarde
if st.button("💾 Enregistrer ce pronostic"):
    st.session_state.history.append({
        "Match": match_name,
        "Score Prédit": f"{res_h}-{res_a}",
        "Over 2.5": f"{over_25:.1f}%",
        "Confiance": f"{np.max(matrix)*100:.1f}%"
    })
    st.success("Pronostic ajouté à l'historique !")

# Affichage des résultats actuels
st.divider()
st.subheader(f"📊 {match_name} : {res_h} - {res_a}")

# Historique
if st.session_state.history:
    st.divider()
    st.subheader("📜 Historique de tes simulations")
    df = pd.DataFrame(st.session_state.history)
    st.table(df)

# Graphique Heatmap
fig = px.imshow(matrix * 100, text_auto=".1f", color_continuous_scale='Viridis',
                labels=dict(x="Extérieur", y="Domicile", color="%"))
st.plotly_chart(fig, use_container_width=True)

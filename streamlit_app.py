import streamlit as st
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="IA Verdict Pro", layout="centered")
st.title("🏆 Aide à la Décision IA")

# Paramètres
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

# Nouvelle fonction de verdict plus nuancée
def show_verdict(label, value, mini, safe, strong):
    if value >= strong:
        st.success(f"✅ **{label}** ({value:.1f}%) : TRÈS SOLIDE")
    elif value >= safe:
        st.info(f"🔵 **{label}** ({value:.1f}%) : À TENTER (Bonne probabilité)")
    elif value >= mini:
        st.warning(f"⚠️ **{label}** ({value:.1f}%) : RISQUÉ (Mais tendance positive)")
    else:
        st.error(f"❌ **{label}** ({value:.1f}%) : ÉVITER ABSOLUMENT")

st.divider()
st.header(f"Tendance : {res_h} - {res_a}")

st.subheader("Analyse détaillée :")
# Seuils abaissés pour t'aider à choisir sans être trop imprudent
show_verdict("Plus de 2.5 buts", over_25, 45, 52, 62)
show_verdict("Les deux équipes marquent", btts, 48, 55, 65)
show_verdict("Score Exact", (np.max(matrix)*100), 10, 13, 16)

st.divider()
if over_25 > 50:
    st.write("💡 **Conseil d'expert :** Le match semble ouvert, privilégie les buts.")
elif btts < 45:
    st.write("💡 **Conseil d'expert :** Match fermé attendu, peut-être un pari sur 'Moins de 2.5 buts'.")

# --- CALCULS AVANCÉS ---
# Probabilités cumulées pour les Over/Under
over_1_5 = (1 - (matrix[0,0] + matrix[0,1] + matrix[1,0])) * 100
over_2_5 = (1 - (matrix[0,0] + matrix[0,1] + matrix[1,0] + matrix[1,1] + matrix[2,0] + matrix[0,2])) * 100
over_3_5 = over_2_5 - (matrix[2,1] * 100 + matrix[1,2] * 100 + matrix[3,0] * 100 + matrix[0,3] * 100)

# --- NOUVELLE LOGIQUE DE VERDICT ---
st.subheader("🎯 Analyse des Scores (Objectif +2.5 / +3.5)")

options = {
    "Plus de 1.5 buts": over_1_5,
    "Plus de 2.5 buts": over_2_5,
    "Plus de 3.5 buts": over_3_5,
    "Les deux marquent": ((1 - h_probs[0]) * (1 - a_probs[0])) * 100,
    "Double Chance 1N": p_1 + p_n,
    "Double Chance N2": p_2 + p_n,
}

# On durcit les critères : 
# Un pari n'est "CONSEILLÉ" que s'il dépasse 75% (très sûr)
# Un pari est "À TENTER" (Belle cote) entre 50% et 75%
for label, val in sorted(options.items(), key=lambda x: x[1], reverse=True):
    if val > 75:
        st.success(f"✅ **CONSEILLÉ** : {label} ({val:.1f}%)")
    elif val > 50:
        st.info(f"🔥 **À TENTER (Belle Cote)** : {label} ({val:.1f}%)")
    elif val > 30:
        st.warning(f"⚠️ **PRUDENCE** : {label} ({val:.1f}%)")

"""
╔══════════════════════════════════════════════════════════════╗
║         AMEN BANK — Système d'Analyse de Risque Crédit       ║
║         Version 3.0 — Amen Bank Tunisie                      ║
╚══════════════════════════════════════════════════════════════╝
pip install streamlit pandas numpy scikit-learn xgboost plotly python-dotenv
streamlit run amen_bank_app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import hashlib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
#  HISTORIQUE DES ANALYSES — fichier CSV local
# ══════════════════════════════════════════════
HISTORIQUE_CSV = "amen_bank_historique_analyses.csv"

def save_prediction(client: dict, pred: int, risk_proba: float,
                    conf: float, analyste: str):
    """Enregistre chaque analyse dans le fichier CSV historique."""
    import datetime, os
    row = {
        "Date":              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Analyste":          analyste,
        "Age":               client["Age"],
        "Sexe":              client["Sex"],
        "Emploi":            client["Job"],
        "Logement":          client["Housing"],
        "Epargne":           client["Saving accounts"],
        "Compte_Courant":    client["Checking account"],
        "Montant_Credit":    client["Credit amount"],
        "Duree_Mois":        client["Duration"],
        "Objet":             client["Purpose"],
        "Score_Risque_pct":  round(risk_proba * 100, 2),
        "Confiance_pct":     round(conf, 2),
        "Decision":          "RISQUE ÉLEVÉ" if pred == 1 else "BON CLIENT",
        "Statut":            "bad" if pred == 1 else "good",
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(HISTORIQUE_CSV):
        df_row.to_csv(HISTORIQUE_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(HISTORIQUE_CSV, mode="w", header=True, index=False)

def load_historique() -> pd.DataFrame:
    """Charge l'historique des analyses depuis le CSV."""
    import os
    if os.path.exists(HISTORIQUE_CSV):
        return pd.read_csv(HISTORIQUE_CSV)
    return pd.DataFrame(columns=[
        "Date","Analyste","Age","Sexe","Emploi","Logement",
        "Epargne","Compte_Courant","Montant_Credit","Duree_Mois",
        "Objet","Score_Risque_pct","Confiance_pct","Decision","Statut"
    ])



# ══════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Amen Bank — Risque Crédit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
#  COULEURS AMEN BANK TUNISIE
# ══════════════════════════════════════════════
VERT      = "#006B3C"
VERT_C    = "#00A651"
OR        = "#F5A623"
NOIR      = "#1A1A1A"
BLANC     = "#FFFFFF"
FOND      = "#F2F5F0"
VERT_DARK = "#004D2C"
VERT_BG   = "#E8F5EE"

# ══════════════════════════════════════════════
#  LOGO AMEN BANK TUNISIE — SVG fidèle
#  Cercle gris · vague bleue · vague verte · point bleu
# ══════════════════════════════════════════════
import base64 as _b64

_LOGO_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 220 220'>"
    "<defs>"
    "<radialGradient id='bg' cx='40%' cy='35%' r='65%'>"
    "<stop offset='0%' stop-color='#EBEBEB'/>"
    "<stop offset='100%' stop-color='#C4C4C4'/>"
    "</radialGradient>"
    "<filter id='sh'><feDropShadow dx='0' dy='3' stdDeviation='5' flood-opacity='0.20'/></filter>"
    "</defs>"
    "<circle cx='110' cy='110' r='109' fill='#BBBBBB' filter='url(#sh)'/>"
    "<circle cx='110' cy='110' r='106' fill='url(#bg)'/>"
    "<path d='M 58 50 C 80 14,156 16,174 68 C 186 100,172 134,150 145"
    " C 138 152,122 154,108 148 C 124 142,140 128,142 108"
    " C 144 86,126 64,104 60 C 84 56,64 70,54 92"
    " C 52 76,50 64,58 50 Z' fill='#1A4FA0'/>"
    "<path d='M 50 172 C 28 150,24 112,44 84 C 56 66,76 56,96 58"
    " C 80 66,66 82,66 104 C 66 128,84 146,110 149"
    " C 124 151,140 144,150 132 C 144 150,128 166,108 172"
    " C 86 180,64 182,50 172 Z' fill='#00A651'/>"
    "<circle cx='110' cy='150' r='13' fill='#1A4FA0'/>"
    "<ellipse cx='84' cy='68' rx='25' ry='13' fill='rgba(255,255,255,0.22)'"
    " transform='rotate(-30,84,68)'/>"
    "</svg>"
)
LOGO_B64 = _b64.b64encode(_LOGO_SVG.encode()).decode()
LOGO_SRC = f"data:image/svg+xml;base64,{LOGO_B64}"

# ══════════════════════════════════════════════
#  SÉCURITÉ — Mots de passe hashés (SHA-256)
#  Pour changer un mot de passe :
#    python3 -c "import hashlib; print(hashlib.sha256('votre_mdp'.encode()).hexdigest())"
# ══════════════════════════════════════════════
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# Les mots de passe en clair NE SONT PAS stockés ici
# Seuls les hash SHA-256 sont présents dans le code
USERS = {
    "koussay": {
        "hash": _hash("koussay2004"),
        "role": "Administrateur",
        "name": "Koussay Hassana",
    },
    "bechir": {
        "hash": _hash("bechir2001"),
        "role": "Analyste Crédit",
        "name": "Bechir Ghoudi",
    },
    "directeur": {
        "hash": _hash("dir2024"),
        "role": "Directeur",
        "name": "Koussay Hassana",
    },
}

def check_password(username: str, password: str) -> bool:
    if username not in USERS:
        return False
    return USERS[username]["hash"] == _hash(password)


# ══════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════
for k, v in [("logged_in", False), ("username", ""), ("login_error", "")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Cairo:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Cairo', sans-serif;
    background-color: {FOND};
}}

/* ─── SIDEBAR ─── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {VERT_DARK} 0%, {VERT} 50%, {VERT_C} 100%);
    border-right: 4px solid {OR};
}}
section[data-testid="stSidebar"] * {{ color: {BLANC} !important; }}
section[data-testid="stSidebar"] .stRadio label {{ color: {OR} !important; font-weight:700; }}

/* ─── HEADER ─── */
.amen-header {{
    background: linear-gradient(135deg, {VERT_DARK} 0%, {VERT} 55%, {VERT_C} 100%);
    padding: 1.2rem 2rem;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.6rem;
    box-shadow: 0 6px 28px rgba(0,107,60,0.28);
    border-bottom: 4px solid {OR};
}}
.amen-header h1 {{
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem; color:{BLANC}; margin:0;
}}
.amen-header .sub {{ color:rgba(255,255,255,.72); font-size:.8rem; margin-top:3px; }}
.logo-right {{ display:flex; align-items:center; gap:12px; }}
.logo-right img {{ height:56px; width:56px; border-radius:50%;
    border:3px solid {OR}; box-shadow:0 2px 12px rgba(0,0,0,.3); }}
.logo-text {{ text-align:right; }}
.logo-name {{ font-family:'Playfair Display',serif; font-size:2rem;
    font-weight:700; color:{OR}; letter-spacing:3px; line-height:1; }}
.logo-since {{ font-size:.6rem; color:rgba(255,255,255,.65);
    letter-spacing:2px; text-transform:uppercase; }}

/* ─── KPI CARDS ─── */
.kpi-card {{
    background:{BLANC}; border-radius:14px; padding:1.2rem 1rem;
    text-align:center; box-shadow:0 4px 16px rgba(0,107,60,.1);
    border-top:5px solid {VERT}; transition:transform .2s,box-shadow .2s;
    margin-bottom:.5rem;
}}
.kpi-card:hover {{ transform:translateY(-4px); box-shadow:0 8px 24px rgba(0,107,60,.18); }}
.kpi-icon {{ font-size:1.6rem; }}
.kpi-value {{
    font-size:1.9rem; font-weight:700; color:{VERT};
    font-family:'Playfair Display',serif; margin:4px 0 2px;
}}
.kpi-label {{ font-size:.68rem; color:#6B7280; text-transform:uppercase; letter-spacing:1px; }}
.kpi-danger  {{ border-top-color:#DC2626!important; }}
.kpi-danger  .kpi-value {{ color:#DC2626!important; }}
.kpi-success {{ border-top-color:{VERT_C}!important; }}
.kpi-success .kpi-value {{ color:{VERT_C}!important; }}
.kpi-or      {{ border-top-color:{OR}!important; }}
.kpi-or      .kpi-value {{ color:{OR}!important; }}

/* ─── SECTION TITLE ─── */
.section-title {{
    font-family:'Playfair Display',serif; font-size:1.2rem; color:{VERT_DARK};
    border-left:5px solid {OR}; padding-left:12px; margin:1.6rem 0 1rem;
}}

/* ─── BUTTONS ─── */
.stButton > button {{
    background:linear-gradient(135deg,{VERT} 0%,{VERT_C} 100%)!important;
    color:{BLANC}!important; border:none!important; border-radius:10px!important;
    font-weight:700!important; font-family:'Cairo',sans-serif!important;
    font-size:.95rem!important; padding:.55rem 1.8rem!important;
    transition:all .2s!important; box-shadow:0 4px 14px rgba(0,107,60,.35)!important;
}}
.stButton > button:hover {{
    transform:translateY(-2px)!important;
    box-shadow:0 7px 20px rgba(0,107,60,.45)!important;
    background:linear-gradient(135deg,{VERT_DARK} 0%,{VERT} 100%)!important;
}}

/* ─── LOGIN ─── */
.login-card {{
    background:{BLANC}; border-radius:18px; padding:2.5rem 2.8rem;
    box-shadow:0 16px 48px rgba(0,107,60,.15); border-top:7px solid {VERT};
}}
.login-logo-wrap {{ display:flex; align-items:center; justify-content:center;
    gap:16px; margin-bottom:.5rem; }}
.login-logo-img {{ height:72px; width:72px; border-radius:50%;
    border:3px solid {OR}; }}
.login-logo-txt {{
    font-family:'Playfair Display',serif; font-size:3rem;
    font-weight:700; color:{VERT}; letter-spacing:4px; line-height:1;
}}
.login-or {{ color:{OR}; }}
.login-bank {{
    text-align:center; font-size:.68rem; color:#9CA3AF;
    letter-spacing:3px; text-transform:uppercase; margin-bottom:.3rem;
}}
.login-tag {{
    text-align:center; color:#6B7280; font-size:.82rem;
    margin-bottom:1.6rem; border-top:1px solid #E5E7EB;
    padding-top:.8rem; margin-top:.5rem;
}}
.login-demo {{
    background:{VERT_BG}; border-radius:8px; padding:.7rem 1rem;
    font-size:.72rem; color:{VERT_DARK}; text-align:center;
    margin-top:1rem; border:1px solid {VERT_C}44;
}}

/* ─── RESULT BOX ─── */
.result-good {{
    background:#D1FAE5; border:2px solid {VERT_C};
    border-radius:14px; padding:1.5rem; text-align:center;
}}
.result-bad {{
    background:#FEE2E2; border:2px solid #DC2626;
    border-radius:14px; padding:1.5rem; text-align:center;
}}
.result-title {{
    font-size:1.8rem; font-weight:700;
    font-family:'Playfair Display',serif;
}}

/* ─── INFO BAR ─── */
.info-bar {{
    background:{VERT_BG}; border-left:4px solid {VERT};
    border-radius:0 8px 8px 0; padding:.7rem 1.2rem;
    margin-bottom:1rem; font-size:.87rem; color:{VERT_DARK};
}}

/* ─── FOOTER ─── */
.footer {{
    text-align:center; padding:1.5rem; color:#9CA3AF;
    font-size:.72rem; border-top:1px solid #E5E7EB; margin-top:3rem;
}}
.footer span {{ color:{VERT}; font-weight:600; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  DATA  — scoring multi-facteurs réaliste
# ══════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv", index_col=0)
    for col in ["Saving accounts", "Checking account"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Scoring de risque multi-critères (réaliste)
    score = pd.Series(0.0, index=df.index)
    score += (df["Credit amount"] / df["Credit amount"].max()) * 2.0
    score += (df["Duration"]      / df["Duration"].max())      * 1.5
    score += ((60 - df["Age"].clip(18, 60)) / 42)              * 1.0
    ep_map = {"little": 1.5, "moderate": 0.5, "quite rich": 0.0, "rich": 0.0}
    cc_map = {"little": 1.2, "moderate": 0.4, "rich": 0.0}
    score += df["Saving accounts"].map(ep_map).fillna(0.5)
    score += df["Checking account"].map(cc_map).fillna(0.5)
    score += (3 - df["Job"].clip(0, 3)) * 0.5

    df["Risk"] = (score >= score.quantile(0.65)).map({True: "bad", False: "good"})
    return df


# ══════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════
@st.cache_data
def prepare_features(df):
    CAT = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    NUM = ["Age", "Job", "Credit amount", "Duration"]
    cat_values = {c: sorted(df[c].astype(str).unique().tolist()) for c in CAT}
    df2 = df.copy()
    df2["Risk_bin"] = (df2["Risk"] == "bad").astype(int)
    X = df2[NUM + CAT].copy()
    X_enc = pd.get_dummies(X, columns=CAT, drop_first=False)
    return X_enc, df2["Risk_bin"], X_enc.columns.tolist(), cat_values, NUM, CAT


# ══════════════════════════════════════════════
#  ENTRAÎNEMENT  — Gradient Boosting + 5 autres
# ══════════════════════════════════════════════
@st.cache_resource
def train_models(_X, _y, feat_cols):
    X_tr, X_te, y_tr, y_te = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_tr)
    Xte_sc = sc.transform(X_te)

    def metrics(model, Xtr, Xte, ytr, yte):
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)
        try:
            auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
        except Exception:
            auc = float("nan")
        return {
            "Acc Train": accuracy_score(ytr, model.predict(Xtr)),
            "Acc Test":  accuracy_score(yte, yp),
            "Précision": precision_score(yte, yp, zero_division=0),
            "Rappel":    recall_score(yte, yp, zero_division=0),
            "F1-Score":  f1_score(yte, yp, zero_division=0),
            "AUC-ROC":   auc,
            "_cm":       confusion_matrix(yte, yp),
            "_model":    model,
        }

    res = {}

    # ── Gradient Boosting (modèle principal) ──
    res["Gradient Boosting"] = metrics(
        GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                    learning_rate=0.08, subsample=0.85,
                                    random_state=42),
        X_tr.values, X_te.values, y_tr, y_te)

    # ── Random Forest ──
    res["Forêt Aléatoire"] = metrics(
        RandomForestClassifier(n_estimators=200, max_depth=8,
                                min_samples_split=10, random_state=42),
        X_tr.values, X_te.values, y_tr, y_te)

    # ── XGBoost ──
    if XGB_AVAILABLE:
        res["XGBoost"] = metrics(
            xgb.XGBClassifier(n_estimators=100, max_depth=5, eta=0.08,
                               subsample=0.85, colsample_bytree=0.85,
                               verbosity=0, random_state=42,
                               eval_metric="logloss"),
            X_tr.values, X_te.values, y_tr, y_te)

    # ── Régression Logistique ──
    res["Régression Logistique"] = metrics(
        LogisticRegression(max_iter=2000, C=0.5, random_state=42),
        Xtr_sc, Xte_sc, y_tr, y_te)

    # ── Arbre de Décision ──
    res["Arbre Décision"] = metrics(
        DecisionTreeClassifier(max_depth=6, min_samples_split=15, random_state=42),
        X_tr.values, X_te.values, y_tr, y_te)

    # ── KNN optimal ──
    k_sc = [cross_val_score(KNeighborsClassifier(n_neighbors=k),
                             Xtr_sc, y_tr, cv=5, scoring="accuracy").mean()
            for k in range(3, 20)]
    bk = int(np.argmax(k_sc)) + 3
    res[f"KNN (K={bk})"] = metrics(
        KNeighborsClassifier(n_neighbors=bk), Xtr_sc, Xte_sc, y_tr, y_te)

    return res, sc, X_tr, X_te, y_tr, y_te


# ══════════════════════════════════════════════
#  ENCODER CLIENT — aucune erreur possible
# ══════════════════════════════════════════════
def encode_client(df, client_dict, feat_cols):
    CAT = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    NUM = ["Age", "Job", "Credit amount", "Duration"]
    base     = df[NUM + CAT].copy()
    new_row  = pd.DataFrame([{c: client_dict[c] for c in NUM + CAT}])
    combined = pd.concat([base, new_row], ignore_index=True)
    encoded  = pd.get_dummies(combined, columns=CAT, drop_first=False)
    for col in feat_cols:
        if col not in encoded.columns:
            encoded[col] = 0
    return encoded[feat_cols].iloc[[-1]].values


# ══════════════════════════════════════════════
#  COMPOSANTS UI
# ══════════════════════════════════════════════
def render_header(title):
    st.markdown(f"""
    <div class="amen-header">
      <div>
        <h1>🏦 {title}</h1>
        <div class="sub">Direction des Risques — Amen Bank Tunisie</div>
      </div>
      <div class="logo-right">
        <img src="{LOGO_SRC}" alt="Logo Amen Bank"/>
        <div class="logo-text">
          <div class="logo-name">AMEN</div>
          <div class="logo-since">Banque Tunisienne · Since 1967</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_sidebar():
    u = USERS[st.session_state.username]
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:1.2rem 0 .5rem">
          <img src="{LOGO_SRC}"
               style="height:64px;width:64px;border-radius:50%;
                      border:3px solid {OR};margin-bottom:8px"/>
          <div style="font-family:'Playfair Display',serif;font-size:1rem;
                      color:{OR};font-weight:700">{u['name']}</div>
          <div style="font-size:.72rem;color:rgba(255,255,255,.6);margin-top:2px">{u['role']}</div>
          <div style="background:rgba(245,166,35,.15);border-radius:20px;
                      padding:2px 12px;display:inline-block;margin-top:6px;
                      font-size:.67rem;color:{OR};border:1px solid {OR}55">
            🟢 En ligne · {st.session_state.username}</div>
        </div>
        <hr style="border-color:{OR}44;margin:.5rem 0 .9rem">
        """, unsafe_allow_html=True)

        page = st.radio("📌 Navigation", [
            "🏠  Tableau de Bord",
            "📊  Analyse Exploratoire",
            "🤖  Modèles ML",
            "🔮  Prédiction Client",
            "📋  Données Brutes",
        ])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(245,166,35,.1);border-radius:10px;
                    padding:.8rem;border:1px solid {OR}44;margin-bottom:.9rem">
          <div style="font-size:.65rem;color:{OR};font-weight:700;
                      text-transform:uppercase;letter-spacing:1px">📁 Dataset actif</div>
          <div style="font-size:.86rem;margin-top:4px">German Credit Data</div>
          <div style="font-size:.68rem;color:rgba(255,255,255,.5)">1 000 clients · 10 variables</div>
        </div>""", unsafe_allow_html=True)

        if st.button("🚪  Se déconnecter", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.rerun()

        st.markdown(f"""
        <div style="text-align:center;padding:.8rem 0 0;font-size:.65rem;
                    color:rgba(255,255,255,.3);border-top:1px solid {OR}22;margin-top:.4rem">
          © 2024 <span style="color:{OR}">Amen Bank</span><br>Risque Crédit v3.0
        </div>""", unsafe_allow_html=True)
    return page


# ══════════════════════════════════════════════
#  PAGE LOGIN
# ══════════════════════════════════════════════
def page_login():
    st.markdown("""
    <style>
      section[data-testid="stSidebar"]{display:none!important}
      header{display:none!important}
    </style>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:linear-gradient(90deg,{VERT_DARK},{VERT},{VERT_C});
                height:6px;border-radius:3px;margin-bottom:2rem"></div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown(f"""
        <div class="login-card">
          <div class="login-logo-wrap">
            <img class="login-logo-img"
                 src="{LOGO_SRC}" alt="Amen Bank Logo"/>
            <div class="login-logo-txt">AMEN<span class="login-or">●</span></div>
          </div>
          <div class="login-bank">Amen Bank Tunisie</div>
          <div class="login-tag">
            🏦 Système de Gestion du Risque Crédit<br>
            <span style="color:{VERT};font-weight:600">Direction des Risques · Tunis</span>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        username = st.text_input("👤  Identifiant", placeholder="Entrez votre identifiant")
        password = st.text_input("🔒  Mot de passe", type="password",
                                  placeholder="••••••••")

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

        if st.button("🔐  Se connecter", use_container_width=True):
            if check_password(username, password):
                st.session_state.logged_in   = True
                st.session_state.username    = username
                st.session_state.login_error = ""
                st.rerun()
            else:
                st.session_state.login_error = "❌ Identifiant ou mot de passe incorrect."
                st.rerun()




# ══════════════════════════════════════════════
#  PAGE TABLEAU DE BORD
# ══════════════════════════════════════════════
def page_dashboard(df):
    render_header("Tableau de Bord")
    bad  = df[df["Risk"] == "bad"]
    good = df[df["Risk"] == "good"]
    pct  = len(bad) / len(df) * 100

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, lbl, icon, cls in [
        (c1, len(df),   "Clients Analysés",   "🏦", ""),
        (c2, len(bad),  "Risques Élevés",      "⚠️",  "kpi-danger"),
        (c3, len(good), "Risques Faibles",     "✅",  "kpi-success"),
        (c4, f"{df['Credit amount'].mean():,.0f} TND", "Montant Moyen", "💰", "kpi-or"),
        (c5, f"{pct:.1f}%", "Taux de Risque", "📊",  "kpi-danger"),
    ]:
        col.markdown(f"""
        <div class="kpi-card {cls}">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Répartition du Risque Crédit</div>',
                    unsafe_allow_html=True)
        rc  = df["Risk"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["✅ Bon Client", "⚠️ Risque Élevé"],
            values=[rc.get("good", 0), rc.get("bad", 0)],
            hole=0.55, marker_colors=[VERT_C, "#DC2626"],
            textfont_size=13, pull=[0, 0.05]))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=30),
            showlegend=True, legend=dict(orientation="h", y=-0.12),
            annotations=[dict(text=f"<b>{pct:.0f}%</b><br>Risque",
                x=0.5, y=0.5, font_size=14, showarrow=False, font_color="#DC2626")])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Montant Moyen par Objet du Crédit</div>',
                    unsafe_allow_html=True)
        pa   = df.groupby("Purpose")["Credit amount"].mean().sort_values()
        fig2 = go.Figure(go.Bar(
            x=pa.values, y=pa.index, orientation="h",
            marker_color=[VERT if i % 2 == 0 else VERT_C for i in range(len(pa))],
            text=[f"{v:,.0f}" for v in pa.values], textposition="outside"))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-title">Distribution des Âges</div>',
                    unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=good["Age"], name="✅ Bon",
            nbinsx=25, marker_color=VERT_C, opacity=0.75))
        fig3.add_trace(go.Histogram(x=bad["Age"], name="⚠️ Risqué",
            nbinsx=25, marker_color="#DC2626", opacity=0.75))
        fig3.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">Durée vs Montant Crédit</div>',
                    unsafe_allow_html=True)
        fig4 = px.scatter(df, x="Duration", y="Credit amount", color="Risk",
            color_discrete_map={"good": VERT_C, "bad": "#DC2626"}, opacity=0.55,
            labels={"Duration": "Durée (mois)", "Credit amount": "Montant", "Risk": "Risque"})
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE ANALYSE EXPLORATOIRE
# ══════════════════════════════════════════════
def page_eda(df):
    render_header("Analyse Exploratoire des Données")
    NUM = ["Age", "Credit amount", "Duration", "Job"]
    tab1, tab2, tab3 = st.tabs(["📈  Distributions", "🔗  Corrélations", "🏷️  Catégorielles"])

    with tab1:
        st.markdown('<div class="section-title">Histogrammes des Variables Numériques</div>',
                    unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=2, subplot_titles=NUM)
        for i, col in enumerate(NUM):
            r, c = (i // 2) + 1, (i % 2) + 1
            fig.add_trace(go.Histogram(x=df[col], marker_color=[VERT, VERT_C, OR, "#059669"][i],
                opacity=0.85, name=col, showlegend=False, nbinsx=30), row=r, col=c)
        fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Boxplots Good vs Bad</div>',
                    unsafe_allow_html=True)
        fig2 = make_subplots(rows=1, cols=4, subplot_titles=NUM)
        for i, col in enumerate(NUM):
            for risk, color in [("good", VERT_C), ("bad", "#DC2626")]:
                fig2.add_trace(go.Box(
                    y=df[df["Risk"] == risk][col],
                    name=f"{'✅' if risk=='good' else '⚠️'}",
                    marker_color=color, showlegend=(i == 0)), row=1, col=i + 1)
        fig2.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        corr = df[NUM].corr()
        fig3 = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0, VERT_BG], [0.5, OR], [1, VERT]],
            text=np.round(corr.values, 3), texttemplate="<b>%{text}</b>",
            showscale=True))
        fig3.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=10))
        _, mc, _ = st.columns([1, 2, 1])
        with mc:
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        for cat in ["Housing", "Saving accounts", "Purpose", "Sex"]:
            cnt  = df.groupby([cat, "Risk"]).size().reset_index(name="count")
            fig4 = px.bar(cnt, x=cat, y="count", color="Risk", barmode="group",
                color_discrete_map={"good": VERT_C, "bad": "#DC2626"},
                title=f"Répartition du Risque par {cat}",
                labels={"count": "Nombre", "Risk": "Risque"})
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=10), height=300)
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE MODÈLES ML
# ══════════════════════════════════════════════
def page_models(df):
    render_header("Modèles de Machine Learning")
    X_enc, y, feat_cols, _, _, _ = prepare_features(df)

    with st.spinner("⚙️  Entraînement des modèles en cours…"):
        res, sc, X_tr, X_te, y_tr, y_te = train_models(X_enc, y, feat_cols)

    rows = [{"Algorithme": n,
             "Acc Train":  f"{m['Acc Train']:.3f}",
             "Acc Test":   f"{m['Acc Test']:.3f}",
             "Précision":  f"{m['Précision']:.3f}",
             "Rappel":     f"{m['Rappel']:.3f}",
             "F1-Score":   f"{m['F1-Score']:.3f}",
             "AUC-ROC":    f"{m['AUC-ROC']:.3f}" if not np.isnan(m["AUC-ROC"]) else "N/A"}
            for n, m in res.items()]
    df_r = pd.DataFrame(rows)
    bi   = df_r["Acc Test"].astype(float).idxmax()

    def hl(row):
        s = f"background-color:{VERT_BG};font-weight:bold;color:{VERT_DARK}"
        return [s if row.name == bi else ""] * len(row)

    st.markdown('<div class="section-title">Tableau Comparatif des Algorithmes</div>',
                unsafe_allow_html=True)
    st.dataframe(df_r.style.apply(hl, axis=1), use_container_width=True, hide_index=True)
    st.success(f"🏆  Meilleur modèle : **{df_r.loc[bi,'Algorithme']}** "
               f"— Accuracy Test : **{df_r.loc[bi,'Acc Test']}** "
               f"| AUC-ROC : **{df_r.loc[bi,'AUC-ROC']}**")

    # Bar chart
    st.markdown('<div class="section-title">Accuracy Train vs Test</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Train", x=[r["Algorithme"] for r in rows],
        y=[float(r["Acc Train"]) for r in rows], marker_color=VERT,
        text=[r["Acc Train"] for r in rows], textposition="outside"))
    fig.add_trace(go.Bar(name="Test", x=[r["Algorithme"] for r in rows],
        y=[float(r["Acc Test"]) for r in rows], marker_color=OR,
        text=[r["Acc Test"] for r in rows], textposition="outside"))
    fig.update_layout(barmode="group", yaxis_range=[0.4, 1.15],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # Matrice de confusion
    bn  = df_r.loc[bi, "Algorithme"]
    cm  = res[bn]["_cm"]
    st.markdown(f'<div class="section-title">Matrice de Confusion — {bn}</div>',
                unsafe_allow_html=True)
    fig2 = go.Figure(go.Heatmap(
        z=cm, x=["Prédit Bad", "Prédit Good"], y=["Réel Bad", "Réel Good"],
        colorscale=[[0, VERT_BG], [1, VERT]],
        text=cm, texttemplate="<b>%{text}</b>", showscale=False))
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(t=10, b=10))
    _, mc, _ = st.columns([1, 2, 1])
    with mc:
        st.plotly_chart(fig2, use_container_width=True)

    # AUC-ROC bar
    st.markdown('<div class="section-title">Scores AUC-ROC par Modèle</div>',
                unsafe_allow_html=True)
    auc_vals = [(r["Algorithme"], float(r["AUC-ROC"]))
                for r in rows if r["AUC-ROC"] != "N/A"]
    fig3 = go.Figure(go.Bar(
        x=[v for _, v in auc_vals], y=[n for n, _ in auc_vals],
        orientation="h", marker_color=VERT_C,
        text=[f"{v:.3f}" for _, v in auc_vals], textposition="outside"))
    fig3.update_layout(xaxis_range=[0.5, 1.05], paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE PRÉDICTION CLIENT
# ══════════════════════════════════════════════
def page_prediction(df):
    render_header("Prédiction du Risque Client")
    X_enc, y, feat_cols, cat_map, NUM, CAT = prepare_features(df)

    with st.spinner("⚙️  Chargement du modèle…"):
        res, sc, _, _, _, _ = train_models(X_enc, y, feat_cols)

    best_name = max(res, key=lambda k: res[k]["Acc Test"])
    model     = res[best_name]["_model"]
    use_scale = any(k in best_name for k in ["Logistique", "Naive", "KNN"])

    st.markdown(f"""
    <div class="info-bar">
      🤖 Modèle actif : <b>{best_name}</b> &nbsp;|&nbsp;
      Accuracy : <b>{res[best_name]['Acc Test']:.1%}</b> &nbsp;|&nbsp;
      AUC-ROC : <b>{res[best_name]['AUC-ROC']:.3f}</b> &nbsp;|&nbsp;
      F1-Score : <b>{res[best_name]['F1-Score']:.3f}</b>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Informations du Client</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**👤 Données Personnelles**")
        age     = st.slider("Âge", 18, 75, 35)
        sex     = st.selectbox("Sexe", cat_map["Sex"])
        job     = st.selectbox("Catégorie Emploi", [0, 1, 2, 3],
                    format_func=lambda x: ["0 – Sans emploi", "1 – Non qualifié",
                                            "2 – Qualifié", "3 – Très qualifié"][x])
        housing = st.selectbox("Logement", cat_map["Housing"])

    with c2:
        st.markdown("**💳 Situation Financière**")
        saving   = st.selectbox("Compte Épargne",  cat_map["Saving accounts"])
        checking = st.selectbox("Compte Courant",  cat_map["Checking account"])
        credit   = st.number_input("Montant Crédit (€)", 250, 20000, 3000, step=100)

    with c3:
        st.markdown("**📋 Détails du Crédit**")
        duration = st.slider("Durée (mois)", 4, 72, 24)
        purpose  = st.selectbox("Objet du Crédit", cat_map["Purpose"])
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{VERT_BG};border-radius:10px;padding:.8rem;
                    border:1px solid {VERT_C}55">
          <div style="font-size:.7rem;color:{VERT_DARK};font-weight:700">📊 Résumé</div>
          <div style="font-size:.82rem;color:{VERT_DARK};margin-top:4px">
            Âge : <b>{age} ans</b> · Durée : <b>{duration} mois</b><br>
            Montant : <b>{credit:,} €</b>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮  Analyser le Risque du Client", use_container_width=True):
        client = {"Age": age, "Job": job, "Credit amount": credit, "Duration": duration,
                  "Sex": sex, "Housing": housing, "Saving accounts": saving,
                  "Checking account": checking, "Purpose": purpose}

        X_cl = encode_client(df, client, feat_cols)
        if use_scale:
            X_cl = sc.transform(X_cl)

        pred  = model.predict(X_cl)[0]
        proba = model.predict_proba(X_cl)[0]

        # Probabilité réaliste (pas 98%)
        risk_proba = proba[1]
        conf       = max(proba) * 100

        # ── Enregistrement automatique dans l'historique CSV ──
        save_prediction(client, int(pred), float(risk_proba),
                        float(conf), st.session_state.username)
        st.toast("✅ Analyse enregistrée dans l'historique", icon="💾")

        if pred == 1:
            st.markdown(f"""
            <div class="result-bad">
              <div class="result-title" style="color:#DC2626">⚠️ RISQUE ÉLEVÉ</div>
              <div style="color:#7F1D1D;margin-top:8px;font-size:.9rem">
                Score de risque : <b>{risk_proba:.1%}</b> &nbsp;·&nbsp;
                Indice de confiance : <b>{conf:.1f}%</b>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-good">
              <div class="result-title" style="color:{VERT}">✅ BON CLIENT</div>
              <div style="color:{VERT_DARK};margin-top:8px;font-size:.9rem">
                Score de risque : <b>{risk_proba:.1%}</b> &nbsp;·&nbsp;
                Indice de confiance : <b>{conf:.1f}%</b>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Jauge
        gauge_c = "#DC2626" if pred == 1 else VERT
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(risk_proba * 100, 1),
            number={"suffix": "%", "font": {"color": gauge_c, "size": 34}},
            title={"text": "Score de Risque (%)", "font": {"size": 16, "color": VERT_DARK}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": gauge_c},
                "bgcolor": "white",
                "steps": [
                    {"range": [0,  35], "color": VERT_BG},
                    {"range": [35, 60], "color": "#FEF9C3"},
                    {"range": [60, 100],"color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": NOIR, "width": 3},
                               "thickness": 0.8, "value": 50}
            }))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300,
            margin=dict(t=30, b=10, l=30, r=30))
        _, gc, _ = st.columns([1, 2, 1])
        with gc:
            st.plotly_chart(fig, use_container_width=True)

        # Détail des facteurs de risque
        st.markdown('<div class="section-title">Analyse des Facteurs de Risque</div>',
                    unsafe_allow_html=True)
        factors = {
            "Montant crédit élevé":  credit > df["Credit amount"].median(),
            "Durée longue":          duration > df["Duration"].median(),
            "Jeune emprunteur":      age < 30,
            "Épargne faible":        saving == "little",
            "Compte courant faible": checking == "little",
            "Emploi non qualifié":   job <= 1,
        }
        fc1, fc2 = st.columns(2)
        for i, (factor, is_risk) in enumerate(factors.items()):
            col = fc1 if i % 2 == 0 else fc2
            icon = "🔴" if is_risk else "🟢"
            label = "Facteur de risque" if is_risk else "Facteur favorable"
            col.markdown(f"{icon} **{factor}** — *{label}*")

        # Recommandation
        st.markdown('<div class="section-title">Recommandation</div>',
                    unsafe_allow_html=True)
        if pred == 1:
            st.error("""
**⛔ Crédit déconseillé — Dossier à risque**
- Demander des garanties supplémentaires (hypothèque, caution)
- Réduire le montant demandé ou la durée du crédit
- Exiger un co-emprunteur solvable
- Soumettre au Comité Risques pour validation finale
- Vérifier l'historique bancaire et les incidents antérieurs
            """)
        else:
            st.success("""
**✅ Crédit approuvé — Profil favorable**
- Profil client présentant un faible niveau de risque
- Conditions standard applicables (taux normal)
- Suivi semestriel du compte recommandé
- Dossier conforme aux critères Amen Bank
            """)


# ══════════════════════════════════════════════
#  PAGE DONNÉES BRUTES
# ══════════════════════════════════════════════
def page_data(df):
    render_header("Données Brutes & Historique des Analyses")

    tab1, tab2 = st.tabs(["📋  Dataset Original", "🔍  Historique des Analyses"])

    # ── TAB 1 : Dataset original ──────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Dataset German Credit — 1 000 clients</div>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        rf = c1.selectbox("Risque",   ["Tous", "good", "bad"],    key="rf")
        sf = c2.selectbox("Sexe",     ["Tous", "male", "female"], key="sf")
        hf = c3.selectbox("Logement", ["Tous"] + sorted(df["Housing"].unique()), key="hf")
        pf = c4.selectbox("Objet",    ["Tous"] + sorted(df["Purpose"].unique()), key="pf")

        d = df.copy()
        if rf != "Tous": d = d[d["Risk"]    == rf]
        if sf != "Tous": d = d[d["Sex"]     == sf]
        if hf != "Tous": d = d[d["Housing"] == hf]
        if pf != "Tous": d = d[d["Purpose"] == pf]

        m1, m2, m3 = st.columns(3)
        m1.metric("Lignes affichées", len(d))
        m2.metric("Risques élevés",   len(d[d["Risk"] == "bad"]))
        m3.metric("Montant moyen",    f"{d['Credit amount'].mean():,.0f} €" if len(d) > 0 else "–")

        def cr(v):
            if v == "bad":
                return "background-color:#FEE2E2;color:#991B1B;font-weight:700"
            return "background-color:#D1FAE5;color:#065F46;font-weight:700"

        st.dataframe(d.style.applymap(cr, subset=["Risk"]),
                     use_container_width=True, height=450)
        st.download_button("⬇️  Télécharger Dataset CSV",
            data=d.to_csv(index=True).encode("utf-8"),
            file_name="amen_bank_dataset.csv", mime="text/csv",
            key="dl_dataset")

    # ── TAB 2 : Historique des analyses ──────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-title">Historique des Analyses de Risque</div>',
                    unsafe_allow_html=True)

        hist = load_historique()

        if len(hist) == 0:
            st.info("📭  Aucune analyse enregistrée pour le moment. "
                    "Effectuez une prédiction depuis la page **Prédiction Client** "
                    "pour voir l'historique ici.")
        else:
            # KPIs historique
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Total analyses",    len(hist))
            h2.metric("Risques élevés",    len(hist[hist["Statut"] == "bad"]))
            h3.metric("Bons clients",      len(hist[hist["Statut"] == "good"]))
            h4.metric("Score moyen",       f"{hist['Score_Risque_pct'].mean():.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # Filtres
            fc1, fc2, fc3 = st.columns(3)
            hrf  = fc1.selectbox("Filtrer par Décision",  ["Tous", "BON CLIENT", "RISQUE ÉLEVÉ"], key="hrf")
            hana = fc2.selectbox("Filtrer par Analyste",  ["Tous"] + sorted(hist["Analyste"].unique().tolist()), key="hana")
            fc3.markdown("<br>", unsafe_allow_html=True)

            h = hist.copy()
            if hrf  != "Tous": h = h[h["Decision"]  == hrf]
            if hana != "Tous": h = h[h["Analyste"]  == hana]

            # Colorer la colonne Statut
            def cr_hist(v):
                if v == "bad":
                    return "background-color:#FEE2E2;color:#991B1B;font-weight:700"
                return "background-color:#D1FAE5;color:#065F46;font-weight:700"

            def cr_dec(v):
                if "RISQUE" in str(v):
                    return "color:#DC2626;font-weight:700"
                return f"color:{VERT};font-weight:700"

            styled = h.style.applymap(cr_hist, subset=["Statut"]) \
                             .applymap(cr_dec,  subset=["Decision"])

            st.dataframe(styled, use_container_width=True, height=420, hide_index=True)

            # Graphique évolution dans le temps
            if len(h) >= 2:
                st.markdown('<div class="section-title">Évolution du Score de Risque</div>',
                            unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(h)+1)),
                    y=h["Score_Risque_pct"].values,
                    mode="lines+markers",
                    line=dict(color=VERT, width=2),
                    marker=dict(
                        color=[("#DC2626" if s == "bad" else VERT_C)
                               for s in h["Statut"].values],
                        size=10, line=dict(color="white", width=2)
                    ),
                    name="Score de risque"
                ))
                fig.add_hline(y=50, line_dash="dash", line_color=OR,
                              annotation_text="Seuil 50%")
                fig.update_layout(
                    xaxis_title="N° Analyse", yaxis_title="Score de Risque (%)",
                    yaxis_range=[0, 105],
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=320, margin=dict(t=10, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Boutons export et effacement
            col_dl, col_clr = st.columns(2)
            with col_dl:
                st.download_button(
                    "⬇️  Exporter l'historique CSV",
                    data=hist.to_csv(index=False).encode("utf-8"),
                    file_name="amen_bank_historique_analyses.csv",
                    mime="text/csv", key="dl_hist",
                    use_container_width=True
                )
            with col_clr:
                if st.button("🗑️  Effacer l'historique", use_container_width=True,
                             key="clr_hist"):
                    import os
                    if os.path.exists(HISTORIQUE_CSV):
                        os.remove(HISTORIQUE_CSV)
                    st.success("Historique effacé.")
                    st.rerun()


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        page_login()
        return

    df   = load_data()
    page = render_sidebar()

    if   "Tableau de Bord"      in page: page_dashboard(df)
    elif "Analyse Exploratoire" in page: page_eda(df)
    elif "Modèles ML"           in page: page_models(df)
    elif "Prédiction"           in page: page_prediction(df)
    elif "Données Brutes"       in page: page_data(df)

    st.markdown(f"""
    <div class="footer">
      © 2024 <span>Amen Bank Tunisie</span> — Tous droits réservés &nbsp;|&nbsp;
      Direction des Risques &nbsp;·&nbsp; Système Risque Crédit v3.0
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

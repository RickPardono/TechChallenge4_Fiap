# app/app.py
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# ----------------------------------------------------
# Função usada no pipeline (precisa existir para o joblib carregar)
# ----------------------------------------------------
def round_ordinal_cols(X):
    X = X.copy()
    return np.rint(X).astype(int)

# ----------------------------------------------------
# Config geral
# ----------------------------------------------------
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🩺",
    layout="wide",
)

# ----------------------------------------------------
# CSS (visual + correções)
# ----------------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background: #F3F4F6; }

      section.main > div {
        max-width: 1100px;
        padding-top: 2.2rem;
      }

      .app-title {
        font-size: 44px;
        font-weight: 900;
        line-height: 1.15;
        margin: 0 0 0.35rem 0;
      }
      .app-subtitle {
        color: rgba(0,0,0,0.60);
        margin-top: 0;
        margin-bottom: 1.2rem;
        font-size: 16px;
      }

      .card {
        background: #FFFFFF;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 22px 22px 10px 22px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.06);
      }

      .block-title {
        font-size: 28px;
        font-weight: 900;
        margin: 0.6rem 0 0.6rem 0;
      }

      /* Botão azul */
      div.stButton > button {
        background: #2563EB !important;
        color: white !important;
        border: 1px solid #2563EB !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.0rem !important;
        font-weight: 700 !important;
      }
      div.stButton > button:hover {
        background: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
      }

      /* Radio azul */
      input[type="radio"]{
        accent-color: #2563EB;
      }

      hr {
        margin: 1.1rem 0 1.1rem 0;
        border: none;
        border-top: 1px solid rgba(0,0,0,0.10);
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# Carregar modelo
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path("models/model.joblib")
    if not model_path.exists():
        st.error("Arquivo do modelo não encontrado em `models/model.joblib`.")
        st.info("Treine e salve o modelo, depois envie o `model.joblib` para a pasta `models/`.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ----------------------------------------------------
# Mapas PT -> EN (modelo)
# ----------------------------------------------------
MAP_GENDER = {"Feminino": "Female", "Masculino": "Male"}
MAP_YESNO  = {"Sim": "yes", "Não": "no"}

MAP_CAEC = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
MAP_CALC = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}

MAP_MTRANS = {
    "Automóvel": "Automobile",
    "Moto": "Motorbike",
    "Bicicleta": "Bike",
    "Transporte público": "Public_Transportation",
    "A pé": "Walking",
}

# ----------------------------------------------------
# Cabeçalho
# ----------------------------------------------------
st.markdown('<div class="app-title">🩺 Sistema Preditivo de Obesidade</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Preencha os dados do paciente para estimar a probabilidade de obesidade (classificação binária).</div>',
    unsafe_allow_html=True
)
st.info("**Observação:** Este sistema é um apoio à decisão e não substitui avaliação clínica.")

# ----------------------------------------------------
# Formulário (em um “card”)
# ----------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

with st.form("form_paciente"):

    # -------------------- Dados do paciente --------------------
    st.markdown('<div class="block-title">Dados do paciente</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        genero_pt = st.radio("Gênero", ["Feminino", "Masculino"], horizontal=True)
        idade = st.number_input("Idade (anos)", min_value=14, max_value=61, value=25, step=1)

    with c2:
        altura = st.number_input("Altura (m)", min_value=1.40, max_value=2.10, value=1.70, step=0.01)
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.5)

    with c3:
        hist_fam = st.radio("Histórico familiar de excesso de peso?", ["Sim", "Não"], horizontal=True)

    st.divider()

    # -------------------- Hábitos alimentares --------------------
    st.markdown('<div class="block-title">Hábitos alimentares</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)

    with a1:
        # FCVC (1–3) via rádio
        fcvc_pt = st.radio("Costuma comer vegetais?", ["Raramente", "Às vezes", "Sempre"], horizontal=True)

        # ✅ NCP (1–4) via number_input
        ncp = st.number_input(
            "Número de refeições diárias (1 a 4)",
            min_value=1,
            max_value=4,
            value=3,
            step=1,
            help="1 a 4 (use 4 para '4 ou mais')."
        )

    with a2:
        caec_pt = st.radio(
            "Costuma comer entre as refeições?",
            ["Não", "Às vezes", "Frequentemente", "Sempre"],
            horizontal=True
        )
        scc = st.radio("Monitora a ingestão calórica?", ["Sim", "Não"], horizontal=True)

    with a3:
        # ✅ CH2O (1–3) via number_input
        ch2o = st.number_input(
            "Consumo diário de água (litros) — escala 1 a 3",
            min_value=1,
            max_value=3,
            value=2,
            step=1,
            help="1=<1L | 2=1–2L | 3=>2L"
        )

        favc = st.radio("Costuma comer alimentos muito calóricos?", ["Sim", "Não"], horizontal=True)

    st.divider()

    # -------------------- Atividade física e rotina --------------------
    st.markdown('<div class="block-title">Atividade física e rotina</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)

    with b1:
        # ✅ FAF (0–3) via number_input
        faf = st.number_input(
            "Frequência de atividade física (dias/semana) — escala 0 a 3",
            min_value=0,
            max_value=3,
            value=1,
            step=1,
            help="0=nenhuma | 1=1–2 dias | 2=3–4 dias | 3=5+ dias"
        )

    with b2:
        # ✅ TUE (0–2) via number_input
        tue = st.number_input(
            "Tempo diário de uso de dispositivos eletrônicos (horas) — escala 0 a 2",
            min_value=0,
            max_value=2,
            value=1,
            step=1,
            help="0=0–2h | 1=3–5h | 2=5h+"
        )

    with b3:
        mtrans_pt = st.selectbox(
            "Meio de transporte habitual",
            ["Automóvel", "Moto", "Bicicleta", "Transporte público", "A pé"]
        )

    st.divider()

    # -------------------- Outros hábitos --------------------
    st.markdown('<div class="block-title">Outros hábitos</div>', unsafe_allow_html=True)
    o1, o2 = st.columns(2)

    with o1:
        calc_pt = st.radio("Consome bebida alcoólica?", ["Não", "Às vezes", "Frequentemente", "Sempre"], horizontal=True)

    with o2:
        fuma = st.radio("Fuma?", ["Sim", "Não"], horizontal=True)

    st.write("")
    enviar = st.form_submit_button("Enviar para predição")

st.markdown('</div>', unsafe_allow_html=True)  # fecha card

# ----------------------------------------------------
# Mapear inputs PT -> valores do modelo e prever
# ----------------------------------------------------
if enviar:
    FCVC_MAP = {"Raramente": 1, "Às vezes": 2, "Sempre": 3}

    row = {
        "Gender": MAP_GENDER[genero_pt],
        "Age": int(idade),
        "Height": float(altura),
        "Weight": float(peso),
        "family_history": MAP_YESNO[hist_fam],
        "FAVC": MAP_YESNO[favc],
        "FCVC": int(FCVC_MAP[fcvc_pt]),
        "NCP": int(ncp),
        "CAEC": MAP_CAEC[caec_pt],
        "SMOKE": MAP_YESNO[fuma],
        "CH2O": int(ch2o),
        "SCC": MAP_YESNO[scc],
        "FAF": int(faf),
        "TUE": int(tue),
        "CALC": MAP_CALC[calc_pt],
        "MTRANS": MAP_MTRANS[mtrans_pt],
    }

    X_input = pd.DataFrame([row])

    proba = float(model.predict_proba(X_input)[0][1])
    pred = int(model.predict(X_input)[0])

    st.markdown("### Resultado da predição")

    if pred == 1:
        st.warning(f"**Classificação:** Obeso  \n**Probabilidade estimada:** {proba:.2%}")
        st.write("**Mensagem ao profissional de saúde:** o modelo sugere maior probabilidade de obesidade. Recomenda-se avaliação clínica e acompanhamento conforme protocolo institucional.")
    else:
        st.success(f"**Classificação:** Não obeso  \n**Probabilidade estimada:** {proba:.2%}")
        st.write("**Mensagem ao profissional de saúde:** o modelo sugere menor probabilidade de obesidade. Recomenda-se manter acompanhamento e orientações preventivas conforme contexto clínico.")

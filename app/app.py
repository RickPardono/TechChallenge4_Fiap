# app/app.py
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# -----------------------------
# Função utilizada no pré-processamento do pipeline de Machine Learning
# -----------------------------
def round_ordinal_cols(X):
    X = X.copy()
    return np.rint(X).astype(int)

# -----------------------------
# Config geral
# -----------------------------
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🩺",
    layout="centered",
)

# CSS simples para melhorar estética
st.markdown(
    """
    <style>
      .main {max-width: 900px;}
      .block-container {padding-top: 1.5rem;}
      .card {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 14px;
        padding: 18px 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
      }
      .title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 0.25rem;
      }
      .subtitle {
        color: rgba(0,0,0,0.65);
        margin-top: 0;
      }
      .result-ok {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(0, 200, 80, 0.08);
      }
      .result-alert {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(255, 165, 0, 0.12);
      }
      .small-note {
        font-size: 12px;
        color: rgba(0,0,0,0.6);
      }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Carregar modelo
# -----------------------------
@st.cache_resource
def load_model():
    model_path = Path("models/model.joblib")
    if not model_path.exists():
        st.error("Arquivo do modelo não encontrado em `models/model.joblib`.")
        st.info("Treine e salve o modelo, depois envie o `model.joblib` para a pasta `models/` no GitHub.")
        st.stop()
    return joblib.load(model_path)


model = load_model()


# -----------------------------
# Mapas PT -> EN (para o modelo)
# -----------------------------
MAP_GENDER = {"Feminino": "Female", "Masculino": "Male"}

MAP_YESNO = {"Sim": "yes", "Não": "no"}

MAP_CAEC = {
    "Não": "no",
    "Às vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always",
}

MAP_CALC = {
    "Não": "no",
    "Às vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always",
}

MAP_MTRANS = {
    "Automóvel": "Automobile",
    "Moto": "Motorbike",
    "Bicicleta": "Bike",
    "Transporte público": "Public_Transportation",
    "A pé": "Walking",
}


# -----------------------------
# Cabeçalho
# -----------------------------
st.markdown('<div class="title">🩺 Sistema Preditivo de Obesidade</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Preencha os dados do paciente para estimar a probabilidade de obesidade (classificação binária).</p>',
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("**Observação:** Este sistema é um apoio à decisão e não substitui avaliação clínica.")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")


# -----------------------------
# Formulário
# -----------------------------
with st.form("form_paciente"):
    st.markdown("## Dados do paciente")

    c1, c2, c3 = st.columns(3)

    with c1:
        genero_pt = st.selectbox("Gênero", ["Feminino", "Masculino"])
        idade = st.number_input("Idade (anos)", min_value=14, max_value=61, value=25, step=1)

    with c2:
        altura = st.number_input("Altura (m)", min_value=1.40, max_value=2.10, value=1.70, step=0.01)
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.5)

    with c3:
        hist_fam = st.selectbox("Histórico familiar de excesso de peso", ["Sim", "Não"])
        favc = st.selectbox("Consumo frequente de alimentos muito calóricos (FAVC)", ["Sim", "Não"])
        fuma = st.selectbox("Fuma (SMOKE)", ["Sim", "Não"])

    st.markdown("## Hábitos e estilo de vida")
    c4, c5, c6 = st.columns(3)

    with c4:
        fcvc = st.selectbox("Consumo de vegetais (FCVC)", [1, 2, 3], help="1=raro, 2=às vezes, 3=sempre")
        ncp = st.selectbox("Nº de refeições principais por dia (NCP)", [1, 2, 3, 4], help="4 = quatro ou mais")
        caec_pt = st.selectbox("Come entre refeições (CAEC)", ["Não", "Às vezes", "Frequentemente", "Sempre"])

    with c5:
        ch2o = st.selectbox("Consumo diário de água (CH2O)", [1, 2, 3], help="1=<1L, 2=1–2L, 3=>2L")
        scc = st.selectbox("Monitora ingestão calórica (SCC)", ["Sim", "Não"])
        faf = st.selectbox("Atividade física (FAF)", [0, 1, 2, 3], help="0=nenhuma … 3=5x/sem ou mais")

    with c6:
        tue = st.selectbox("Tempo em dispositivos eletrônicos (TUE)", [0, 1, 2], help="0=0–2h … 2=>5h")
        calc_pt = st.selectbox("Consumo de álcool (CALC)", ["Não", "Às vezes", "Frequentemente", "Sempre"])
        mtrans_pt = st.selectbox("Meio de transporte (MTRANS)", ["Automóvel", "Moto", "Bicicleta", "Transporte público", "A pé"])

    enviar = st.form_submit_button("Enviar para predição")

# -----------------------------
# Predição
# -----------------------------
if enviar:
    # Mapear PT -> EN (modelo foi treinado com esses rótulos)
    row = {
        "Gender": MAP_GENDER[genero_pt],
        "Age": int(idade),
        "Height": float(altura),
        "Weight": float(peso),
        "family_history": MAP_YESNO[hist_fam],
        "FAVC": MAP_YESNO[favc],
        "FCVC": int(fcvc),
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

    # Probabilidade (classe 1 = obeso)
    proba = float(model.predict_proba(X_input)[0][1])
    pred = int(model.predict(X_input)[0])

    st.markdown("## Resultado da predição")

    if pred == 1:
        st.markdown(f'<div class="result-alert"><b>Classificação:</b> Obeso<br><b>Probabilidade estimada:</b> {proba:.2%}</div>',
                    unsafe_allow_html=True)
        st.write("")
        st.write("**Mensagem ao profissional de saúde:** o modelo sugere maior probabilidade de obesidade. Recomenda-se avaliação clínica e acompanhamento conforme protocolo institucional.")
    else:
        st.markdown(f'<div class="result-ok"><b>Classificação:</b> Não obeso<br><b>Probabilidade estimada:</b> {proba:.2%}</div>',
                    unsafe_allow_html=True)
        st.write("")
        st.write("**Mensagem ao profissional de saúde:** o modelo sugere menor probabilidade de obesidade. Recomenda-se manter acompanhamento e orientações preventivas conforme contexto clínico.")

    st.markdown('<p class="small-note">Nota: esta estimativa é probabilística e depende das informações inseridas.</p>',
                unsafe_allow_html=True)

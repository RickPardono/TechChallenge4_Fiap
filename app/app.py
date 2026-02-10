# app/app.py
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# -----------------------------
# Função utilizada no pré-processamento do pipeline de Machine Learning
# (precisa existir para o unpickle do model.joblib)
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
    layout="wide",   # ✅ wide
)

# -----------------------------
# CSS (cards + títulos + fundo)
# -----------------------------
st.markdown(
    """
    <style>
      /* Fundo externo (fora do "card") */
      .stApp { background: #F3F4F6; }

      /* Container central */
      .block-container { padding-top: 1.2rem; max-width: 1200px; }

      /* Cards */
      .card {
        background: #FFFFFF;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
      }

      /* Título topo */
      .title {
        font-size: 40px;
        font-weight: 850;
        margin: 0 0 .25rem 0;
        line-height: 1.15; /* evita “cortar” topo */
      }

      .subtitle {
        color: rgba(0,0,0,0.65);
        margin-top: 0;
        margin-bottom: 0.75rem;
      }

      /* Títulos dos blocos (alinhado à esquerda + maior) */
      .section-title {
        font-size: 28px;
        font-weight: 800;
        margin: 10px 0 6px 0;
        text-align: left;
      }

      .small-note {
        font-size: 12px;
        color: rgba(0,0,0,0.6);
      }

      /* Botão (ajuste leve: largura e borda) */
      div.stButton > button {
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-weight: 700;
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
        st.info("Envie o `model.joblib` para a pasta `models/` e tente novamente.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# -----------------------------
# Mapas PT -> EN (para o modelo)
# -----------------------------
MAP_GENDER = {"Feminino": "Female", "Masculino": "Male"}
MAP_YESNO = {"Sim": "yes", "Não": "no"}

MAP_CAEC = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
MAP_CALC = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}

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

st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("**Observação:** Este sistema é um apoio à decisão e não substitui avaliação clínica.")
st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# -----------------------------
# Formulário
# -----------------------------
with st.form("form_paciente"):
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # ====== DADOS DO PACIENTE ======
    st.markdown('<div class="section-title">Dados do paciente</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])

    with c1:
        genero_pt = st.radio("Gênero", ["Feminino", "Masculino"], horizontal=True)
        idade = st.number_input("Idade (anos)", min_value=14, max_value=61, value=25, step=1)

    with c2:
        altura = st.number_input("Altura (m)", min_value=1.40, max_value=2.10, value=1.70, step=0.01)
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.5)

    with c3:
        hist_fam = st.radio("Histórico familiar de excesso de peso?", ["Sim", "Não"], horizontal=True)

    st.divider()

    # ====== HÁBITOS ALIMENTARES ======
    st.markdown('<div class="section-title">Hábitos alimentares</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns([1.2, 2.0, 1.4])  # a2 mais largo p/ caber CAEC em 1 linha

    with a1:
        fcvc_pt = st.radio("Costuma comer vegetais?", ["Raramente", "Às vezes", "Sempre"], horizontal=True)
        ncp_pt = st.radio("Número de refeições diárias", ["1", "2", "3", "4 ou mais"], horizontal=True)

    with a2:
        caec_pt = st.radio(
            "Costuma comer entre as refeições?",
            ["Não", "Às vezes", "Frequentemente", "Sempre"],
            horizontal=True
        )
        scc_pt = st.radio("Monitora a ingestão calórica?", ["Sim", "Não"], horizontal=True)

    with a3:
        ch2o_pt = st.radio("Consumo diário de água (litros)", ["< 1 L", "1–2 L", "> 2 L"], horizontal=True)
        favc = st.radio("Costuma comer alimentos muito calóricos?", ["Sim", "Não"], horizontal=True)  # ✅ movido

    st.divider()

    # ====== ATIVIDADE FÍSICA E ROTINA ======
    st.markdown('<div class="section-title">Atividade física e rotina</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns([1.3, 1.6, 1.4])

    with r1:
        faf_pt = st.radio("Frequência de atividade física (dias/semana)", ["0", "1–2", "3–4", "5+"], horizontal=True)

    with r2:
        tue_pt = st.radio("Tempo diário de uso de dispositivos eletrônicos (horas)", ["0–2 h", "3–5 h", "> 5 h"], horizontal=True)

    with r3:
        mtrans_pt = st.selectbox("Meio de transporte habitual", ["Automóvel", "Moto", "Bicicleta", "Transporte público", "A pé"])

    st.divider()

    # ====== OUTROS HÁBITOS ======
    st.markdown('<div class="section-title">Outros hábitos</div>', unsafe_allow_html=True)
    o1, o2 = st.columns([1.6, 1.2])

    with o1:
        calc_pt = st.radio("Consome bebida alcoólica?", ["Não", "Às vezes", "Frequentemente", "Sempre"], horizontal=True)

    with o2:
        fuma = st.radio("Fuma?", ["Sim", "Não"], horizontal=True)  # ✅ movido

    st.write("")
    enviar = st.form_submit_button("Enviar para predição")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Helpers de mapeamento (PT -> valores esperados pelo modelo)
# -----------------------------
FCVC_MAP = {"Raramente": 1, "Às vezes": 2, "Sempre": 3}
NCP_MAP = {"1": 1, "2": 2, "3": 3, "4 ou mais": 4}
CH2O_MAP = {"< 1 L": 1, "1–2 L": 2, "> 2 L": 3}
FAF_MAP = {"0": 0, "1–2": 1, "3–4": 2, "5+": 3}
TUE_MAP = {"0–2 h": 0, "3–5 h": 1, "> 5 h": 2}

# -----------------------------
# Predição
# -----------------------------
if enviar:
    row = {
        "Gender": MAP_GENDER[genero_pt],
        "Age": int(idade),
        "Height": float(altura),
        "Weight": float(peso),
        "family_history": MAP_YESNO[hist_fam],
        "FAVC": MAP_YESNO[favc],
        "FCVC": int(FCVC_MAP[fcvc_pt]),
        "NCP": int(NCP_MAP[ncp_pt]),
        "CAEC": MAP_CAEC[caec_pt],
        "SMOKE": MAP_YESNO[fuma],
        "CH2O": int(CH2O_MAP[ch2o_pt]),
        "SCC": MAP_YESNO[scc_pt],
        "FAF": int(FAF_MAP[faf_pt]),
        "TUE": int(TUE_MAP[tue_pt]),
        "CALC": MAP_CALC[calc_pt],
        "MTRANS": MAP_MTRANS[mtrans_pt],
    }

    X_input = pd.DataFrame([row])

    proba = float(model.predict_proba(X_input)[0][1])  # classe 1 = obeso
    pred = int(model.predict(X_input)[0])

    st.markdown('<div class="section-title">Resultado da predição</div>', unsafe_allow_html=True)

    if pred == 1:
        st.warning(f"**Classificação:** Obeso  \n**Probabilidade estimada:** {proba:.2%}")
        st.write("**Mensagem ao profissional de saúde:** maior probabilidade de obesidade. Recomenda-se avaliação clínica e acompanhamento conforme protocolo.")
    else:
        st.success(f"**Classificação:** Não obeso  \n**Probabilidade estimada:** {proba:.2%}")
        st.write("**Mensagem ao profissional de saúde:** menor probabilidade de obesidade. Recomenda-se acompanhamento preventivo conforme contexto clínico.")

    st.markdown('<p class="small-note">Nota: esta estimativa é probabilística e depende das informações inseridas.</p>', unsafe_allow_html=True)

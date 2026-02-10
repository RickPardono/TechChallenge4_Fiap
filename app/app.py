# app/app.py
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# =========================================================
# Função auxiliar necessária para desserialização do pipeline
# (utilizada em um FunctionTransformer durante o treinamento)
# =========================================================
def round_ordinal_cols(X):
    X = X.copy()
    return np.rint(X).astype(int)


# =========================================================
# Configuração geral
# =========================================================
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🩺",
    layout="wide",   # <- você pediu wide
    initial_sidebar_state="collapsed",
)

# CSS para estética (tema claro) + títulos centralizados
st.markdown(
    """
    <style>
      /* largura e espaçamento topo (evita título cortado) */
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1200px;}

      /* títulos */
      h1, h2, h3 {letter-spacing: -0.2px;}
      .section-title{
        text-align:center;
        font-size: 26px;
        font-weight: 800;
        margin: 18px 0 12px 0;
      }

      /* cards */
      .card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
      }

      /* resultado */
      .result-ok {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.10);
        background: rgba(34, 197, 94, 0.10);  /* verde claro */
      }
      .result-alert {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.10);
        background: rgba(245, 158, 11, 0.14); /* laranja claro */
      }

      .small-note {
        font-size: 12px;
        color: rgba(0,0,0,0.60);
      }

      /* Ajuste de inputs (deixa “clean”) */
      div[data-baseweb="select"] > div {border-radius: 12px;}
      div[data-testid="stNumberInput"] input {border-radius: 12px;}

      /* Botão */
      .stButton button {
        border-radius: 12px;
        padding: 0.55rem 1rem;
        font-weight: 700;
      }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Carregar modelo (caminho robusto)
# =========================================================
@st.cache_resource
def load_model():
    # app/app.py -> raiz do projeto = pai da pasta app
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "model.joblib"

    if not model_path.exists():
        st.error("Arquivo do modelo não encontrado.")
        st.info(f"Esperado em: {model_path}")
        st.stop()

    return joblib.load(model_path)


model = load_model()


# =========================================================
# Mapas PT -> EN (modelo)
# =========================================================
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

# Mapeamento “legível” -> ordinal numérico esperado
FCVC_MAP = {"Raramente": 1, "Às vezes": 2, "Sempre": 3}
NCP_MAP = {"1": 1, "2": 2, "3": 3, "4 ou mais": 4}
CH2O_MAP = {"< 1 L": 1, "1–2 L": 2, "> 2 L": 3}
FAF_MAP = {"0": 0, "1–2": 1, "3–4": 2, "5+": 3}
TUE_MAP = {"0–2 h": 0, "3–5 h": 1, "> 5 h": 2}


# =========================================================
# Cabeçalho (usa st.title para não cortar)
# =========================================================
st.title("🩺 Sistema Preditivo de Obesidade")
st.caption("Preencha os dados do paciente para estimar a probabilidade de obesidade (classificação binária).")

st.markdown(
    """
    <div class="card">
      <b>Observação:</b> Este sistema é um apoio à decisão e não substitui avaliação clínica.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# =========================================================
# Formulário
# =========================================================
with st.form("form_paciente"):
    st.markdown('<div class="section-title">Dados do paciente</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        genero_pt = st.radio("Gênero", ["Feminino", "Masculino"], horizontal=True)
        idade = st.number_input("Idade (anos)", min_value=14, max_value=61, value=25, step=1)

    with c2:
        altura = st.number_input("Altura (m)", min_value=1.40, max_value=2.10, value=1.70, step=0.01, format="%.2f")
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.5, format="%.1f")

    with c3:
        hist_fam_pt = st.radio("Histórico familiar de excesso de peso?", ["Sim", "Não"], horizontal=True)
        favc_pt = st.radio("Costuma comer alimentos muito calóricos?", ["Sim", "Não"], horizontal=True)
        fuma_pt = st.radio("Fuma?", ["Sim", "Não"], horizontal=True)

    st.markdown('<div class="section-title">Hábitos alimentares</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)

    with a1:
        fcvc_lbl = st.radio("Costuma comer vegetais?", ["Raramente", "Às vezes", "Sempre"], horizontal=True)
        ncp_lbl = st.radio("Número de refeições diárias", ["1", "2", "3", "4 ou mais"], horizontal=True)

    with a2:
        caec_pt = st.radio("Costuma comer entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"], horizontal=True)
        scc_pt = st.radio("Monitora a ingestão calórica?", ["Sim", "Não"], horizontal=True)

    with a3:
        ch2o_lbl = st.radio("Consumo diário de água (litros)", ["< 1 L", "1–2 L", "> 2 L"], horizontal=True)

    st.markdown('<div class="section-title">Atividade física e rotina</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)

    with r1:
        faf_lbl = st.radio(
            "Frequência de atividade física (dias/semana)",
            ["0", "1–2", "3–4", "5+"],
            horizontal=True
        )

    with r2:
        tue_lbl = st.radio(
            "Tempo diário de uso de dispositivos eletrônicos (horas)",
            ["0–2 h", "3–5 h", "> 5 h"],
            horizontal=True
        )

    with r3:
        mtrans_pt = st.selectbox(
            "Meio de transporte habitual",
            ["Automóvel", "Moto", "Bicicleta", "Transporte público", "A pé"]
        )

    st.markdown('<div class="section-title">Outros hábitos</div>', unsafe_allow_html=True)
    o1, o2 = st.columns(2)

    with o1:
        calc_pt = st.radio("Consome bebida alcoólica?", ["Não", "Às vezes", "Frequentemente", "Sempre"], horizontal=True)

    with o2:
        # Já tem “Fuma?” lá em cima, mas se quiser repetir aqui, remova de cima.
        st.write("")

    enviar = st.form_submit_button("Enviar para predição")


# =========================================================
# Predição
# =========================================================
if enviar:
    row = {
        "Gender": MAP_GENDER[genero_pt],
        "Age": int(idade),
        "Height": float(altura),
        "Weight": float(peso),
        "family_history": MAP_YESNO[hist_fam_pt],
        "FAVC": MAP_YESNO[favc_pt],
        "FCVC": int(FCVC_MAP[fcvc_lbl]),
        "NCP": int(NCP_MAP[ncp_lbl]),
        "CAEC": MAP_CAEC[caec_pt],
        "SMOKE": MAP_YESNO[fuma_pt],
        "CH2O": int(CH2O_MAP[ch2o_lbl]),
        "SCC": MAP_YESNO[scc_pt],
        "FAF": int(FAF_MAP[faf_lbl]),
        "TUE": int(TUE_MAP[tue_lbl]),
        "CALC": MAP_CALC[calc_pt],
        "MTRANS": MAP_MTRANS[mtrans_pt],
    }

    X_input = pd.DataFrame([row])

    proba = float(model.predict_proba(X_input)[0][1])  # classe 1 = obeso
    pred = int(model.predict(X_input)[0])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Resultado da predição</div>', unsafe_allow_html=True)

    if pred == 1:
        st.markdown(
            f"""
            <div class="result-alert">
              <b>Classificação:</b> Obeso<br>
              <b>Probabilidade estimada:</b> {proba:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("**Mensagem ao profissional de saúde:** o modelo sugere maior probabilidade de obesidade. Recomenda-se avaliação clínica e acompanhamento conforme protocolo institucional.")
    else:
        st.markdown(
            f"""
            <div class="result-ok">
              <b>Classificação:</b> Não obeso<br>
              <b>Probabilidade estimada:</b> {proba:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("**Mensagem ao profissional de saúde:** o modelo sugere menor probabilidade de obesidade. Recomenda-se manter acompanhamento e orientações preventivas conforme contexto clínico.")

    st.markdown(
        '<p class="small-note">Nota: esta estimativa é probabilística e depende das informações inseridas.</p>',
        unsafe_allow_html=True
    )

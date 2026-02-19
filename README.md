# Tech Challenge - Sistema Preditivo de Obesidade
## 📋 Sobre o Projeto
Este projeto foi desenvolvido como parte do Tech Challenge, com o objetivo de criar um sistema de Machine Learning para auxiliar a equipe médica a prever se uma pessoa pode ter obesidade.
## 🎯 Objetivos
• Desenvolver um modelo preditivo com assertividade acima de 75%

• Criar uma aplicação Streamlit para predição em tempo real

• Construir um dashboard analítico com insights sobre obesidade

• Fornecer ferramentas para auxiliar a tomada de decisão da equipe médica
## 📊 Base de Dados
***Obesity.csv***

**Local:** data/raw

**As variáveis incluem:**

• Dados demográficos (idade, gênero)

• Histórico familiar

• Hábitos alimentares

• Atividade física

• Consumo de álcool

• Tabagismo

• Tempo de uso de dispositivos eletrônicos

• Peso e altura

**Foram criadas as variável derivadas:**

• IMC (Índice de Massa Corporal) -> Usada somente na Análise Exploratória e no Dashboard

• Variável Alvo: ObeseBinary (Obeso = 1 | Não Obeso = 0)
## 🧪 Estrutura do Repositório
<img width="756" height="540" alt="Captura de tela 2026-02-18 143342" src="https://github.com/user-attachments/assets/2a8d4f49-eeee-426d-a6ec-6c4f75ad2e1f" />

## 🔍 Análise Exploratória dos Dados (notebooks/01_eda.ipynb)
• Distribuição dos Níveis de Obesidade

• Estatísticas Descritivas

• Distribuição do IMC

• Boxplots de IMC

• Scatterplot de Relação entre Idade e IMC por Nível de Obesidade

• Distribuições das variáveis

• Probabilidades
## 🤖 Modelagem Preditiva (notebooks/02_modeling.ipynb)
### 🧱**Construção do Pipeline de Pré-processamento com ColumnTransformer para garantir:**

• Imputação de valores ausentes

• Padronização de variáveis numéricas

• Codificação de variáveis binárias

• Tratamento de variáveis ordinais numéricas

• Codificação de variáveis ordinais textuais

• One-hot encoding para variáveis nominais

Função personalizada utilizada no pipeline: a função round_original_cols foi utilizada via FunctionTransformer para corrigir ruído decimal em variáveis ordinais numéricas e ela também está presente no app.py, pois é necessária para que o joblib consiga reconstruir corretamente o pipeline no momento do deploy.

### 🔍**Modelos Testados:**

•  Regressão Logística (baseline)

•  Random Forest

•  XGBoost

## 🏆 Modelo Final Selecionado:

Optou-se pela **Regressão Logística**, pois apresentou:

• Desempenho equivalente aos modelos mais complexos

• Maior interpretabilidade

• Menor risco de overfitting

• Melhor aplicabilidade clínica

**Foi aplicado GridSearchCV para ajuste de hiperparâmetros dentro do pipeline completo.**

## 🔥 Resultados do Modelo:

• **Accuracy:** 0.9976

• **Precision:** 1.0000

• **Recall:** 0.9949

• **F1-score:** 0.9974

📌 Modelo salvo em: **models/model.joblib**

## 📊 Dashboard Analítico no Looker Studio

Dashboard desenvolvido no Looker Studio com:

• Filtros por página

• Indicadores epidemiológicos

• Fatores biológicos e comportamentais

• Hábitos alimentares

• Análise clínica do IMC

• Conclusões estratégicas

• Recomendações para a equipe médica

**🔗 Link disponível em links_entrega.txt**

**Nota: O notebook disponível em notebooks/03_dashboard_prep.ipynb tem como finalidade preparar a base de dados Obesity.csv, gerando o arquivo "obesity_dashboard.xlsx" que conecta-se ao Looker Studio e constitui a base para construção do painel interativo. O arquivo "obesity_dashboard.xlsx" encontra-se em data/processed .**

## 🌐 Aplicação Web – Streamlit
Aplicação desenvolvida com:

• Layout wide

• Tema claro

• Organização por blocos: Dados do paciente, Hábitos alimentares, Atividade física e rotina, Outros hábitos

• Inserção manual de variáveis

• Predição de risco de obesidade

• Exibição da probabilidade estimada

• Deploy realizado no Streamlit Cloud

🔗 Link: https://sistema-preditivo-obesidade-ricardo-pardono.streamlit.app

## 🚀 Como Executar o Projeto 
🔹 **1. Pré-requisitos:**

• Python 3.11

• pip atualizado

• Git instalado

• Conta no Streamlit (para deploy opcional)

🔹 **2. Clonar o repositório:**

git clone https://github.com/RickPardono/TechChallenge4_Fiap.git

cd TechChallenge4_Fiap

🔹 **3. Criar ambiente virtual**

python -m venv venv

**Ativar:**

**No Windows:**

venv\Scripts\activate

**No Mac/Linux:**

source venv/bin/activate

🔹 **4. Instalar dependências**:

pip install -r requirements.txt

🔹 **5. Gerar o modelo (caso não exista):**

notebooks/02_modeling.ipynb

**Ao final, o modelo será salvo em:**

models/model.joblib

**⚠️ Caso o arquivo já esteja presente na pasta models/, essa etapa pode ser ignorada.**

🔹 **6. Executar a aplicação Streamlit localmente:**

streamlit run app/app.py

**A aplicação abrirá em:**

http://localhost:8501

## Autor do Projeto

**Ricardo Pardono**

**Contato:** rpardono@gmail.com


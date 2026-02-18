# Tech Challenge - Sistema Preditivo de Obesidade
## 📋 Sobre o Projeto
Este projeto foi desenvolvido como parte do Tech Challenge, com o objetivo de criar um sistema de Machine Learning para auxiliar a equipe médica a prever se uma pessoa pode ter obesidade.
## 🎯 Objetivos
• Desenvolver um modelo preditivo com assertividade acima de 75%

• Criar uma aplicação Streamlit para predição em tempo real

• Construir um dashboard analítico com insights sobre obesidade

• Fornecer ferramentas para auxiliar a tomada de decisão da equipe médica
## 📊 Base de Dados
Obesity.csv

Local: data/raw

As variáveis incluem:

• Dados demográficos (idade, gênero)

• Histórico familiar

• Hábitos alimentares

• Atividade física

• Consumo de álcool

• Tabagismo

• Tempo de uso de dispositivos eletrônicos

• Peso e altura

Foram criadas as variável derivadas:

• IMC (Índice de Massa Corporal) -> Usada somente na Análise Exploratória e no Dashboard


• Variável Alvo: ObeseBinary (Obeso = 1 | Não Obeso = 0)
## 🧪 Estrutura do Repositório
TechChallenge4_Fiap/
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   └── app.py
│
├── dashboard/
│   └── README.md
│
├── data/
│   ├── raw/
│   │   └── Obesity.csv
│   │
│   └── processed/
│       └── obesity_dashboard.xlsx
│
├── models/
│   └── model.joblib
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_dashboard_prep.ipynb
│
├── .gitignore
├── links_entrega.txt
├── requirements.txt
├── runtime.txt
└── README.md


import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import binom, poisson, norm

# Layout customizado
st.set_page_config(page_title="Dashboard de Probabilidade e ROI", layout="wide")

# Título geral
st.title("📊 Dashboard: Overbooking e ROI")

# Sidebar para parâmetros
st.sidebar.header("Parâmetros Gerais")
# --- Inputs Overbooking ---
st.sidebar.subheader("Overbooking")
vendidos = st.sidebar.slider("Passagens Vendidas", 120, 200, 130)
p = st.sidebar.slider("Chance de Comparecimento (%)", 0.0, 1.0, 0.88, step=0.01)
capacidade = st.sidebar.number_input("Capacidade do Avião", min_value=1, value=120)

# --- Inputs ROI ---
st.sidebar.subheader("ROI")
receita = st.sidebar.number_input("Receita Adicional (R$)", 0, 200000, 80000)
custo = st.sidebar.number_input("Custo Operacional (R$)", 0, 50000, 10000)
inv = st.sidebar.number_input("Investimento Inicial (R$)", 0, 200000, 50000)
sims = st.sidebar.slider("Simulações Monte Carlo", 100, 5000, 1000, step=100)

# Layout de duas colunas para exibição
col1, col2 = st.columns(2)

# --- Coluna 1: Overbooking ---
with col1:
    st.header("✈️ Overbooking")
    st.markdown(f"""
**Contexto:**
Vendidas **{vendidos}** passagens para **{capacidade}** lugares, com **{(1-p)*100:.0f}%** de falta prevista.
""")

    # Distribuição Binomial
    xs = np.arange(vendidos + 1)
    pmf = binom.pmf(xs, vendidos, p)
    df_pmf = pd.DataFrame({"Comparecimentos": xs, "Probabilidade": pmf}).set_index("Comparecimentos")
    st.subheader("Distribuição de Comparecimentos")
    st.line_chart(df_pmf)

    # Probabilidade de overbooking
    prob_over = 1 - binom.cdf(capacidade, vendidos, p)
    st.metric("P(Overbooking)", f"{prob_over:.2%}")

    # Viabilidade Financeira +10 assentos
    custo_ind = st.sidebar.number_input("Custo de Indenização (R$)", 100, 5000, 500)
    preco = st.sidebar.number_input("Preço da Passagem (R$)", 100, 5000, 500)
    lucro_extra = 10 * preco
    custo_esp = prob_over * custo_ind * (vendidos - capacidade)
    st.metric("Lucro Extra (10 assentos)", f"R$ {lucro_extra:,.2f}".replace(",", "."))
    st.metric("Custo Esperado Indenizações", f"R$ {custo_esp:,.2f}".replace(",", "."))

# --- Coluna 2: ROI ---
with col2:
    st.header("💰 ROI")
    roi_esp = (receita - custo) / inv * 100
    st.metric("ROI Esperado", f"{roi_esp:.2f}%")

    # Monte Carlo
    receitas_sim = np.random.normal(receita, 0.2 * receita, sims)
    lucros_sim = receitas_sim - custo
    roi_sim = lucros_sim / inv * 100

    st.subheader("Distribuição de ROI")
    df_h = pd.DataFrame({"ROI (%)": roi_sim}).reset_index(drop=True)
    st.bar_chart(df_h.set_index("ROI (%)"))

    st.subheader("CDF de ROI")
    sorted_roi = np.sort(roi_sim)
    cdf = np.arange(1, sims+1) / sims
    df_c = pd.DataFrame({"CDF": cdf}, index=sorted_roi)
    st.line_chart(df_c)

    pct_neg = (roi_sim < 0).mean()
    st.metric("P(ROI < 0)", f"{pct_neg:.2%}")
    decisao = "Invista, ROI positivo!" if roi_sim.mean() > 0 else "Reavalie o projeto."
    st.markdown(f"**Decisão:** {decisao}")

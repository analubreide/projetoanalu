import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import binom, poisson, norm

# --- Layout de Abas ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Overbooking",
    "Poisson",
    "Normal",
    "ROI"
])

# --- ABA 1: Overbooking ---
with tab1:
    st.header("Análise de Overbooking - Aérea Confiável")
    st.markdown("""
Você vendeu 130 passagens para um avião com 120 lugares, apostando em 12% de não comparecimento.
""")

    vendidos = st.slider("Passagens Vendidas", 120, 200, 130)
    p = st.slider("Chance de Comparecimento (%)", 0.0, 1.0, 0.88, step=0.01)
    capacidade = st.number_input("Capacidade do Avião", 1, 500, 120)

    # Distribuição Binomial
    xs = np.arange(vendidos + 1)
    pmf = binom.pmf(xs, vendidos, p)
    df_pmf = pd.DataFrame({"Comparecimentos": xs, "Probabilidade": pmf}).set_index("Comparecimentos")
    st.subheader("Distribuição de Comparecimentos")
    st.bar_chart(df_pmf)

    # Probabilidade de overbooking
    prob_over = 1 - binom.cdf(capacidade, vendidos, p)
    st.metric("P(overbooking > capacidade)", f"{prob_over:.2%}")

    # Risco vs vendas
    vendas_test = np.arange(capacidade, vendidos * 2 + 1)
    riscos = 1 - binom.cdf(capacidade, vendas_test, p)
    df_risco = pd.DataFrame({
        "Passagens Vendidas": vendas_test,
        "Risco": riscos
    }).set_index("Passagens Vendidas")
    st.subheader("Risco de Overbooking (Limite 7%)")
    st.line_chart(df_risco)
    st.write("→ Se Risco ≤ 7%, vendas seguras até:",
             int(df_risco[df_risco["Risco"] <= 0.07].idxmax()))

    # Viabilidade financeira +10 assentos
    st.subheader("Venda de +10 assentos")
    custo_ind = st.number_input("Custo de Indenização (R$)", 100, 5000, 500)
    preco = st.number_input("Preço Médio da Passagem (R$)", 100, 5000, 500)
    lucro_extra = 10 * preco
    custo_esp = prob_over * custo_ind * (vendidos - capacidade)
    st.metric("Lucro Extra (10 assentos)", f"R$ {lucro_extra:,.2f}".replace(",", "."))
    st.metric("Custo Esperado Indenizações", f"R$ {custo_esp:,.2f}".replace(",", "."))

# --- ABA 2: Poisson ---
with tab2:
    st.header("Distribuição de Poisson")
    lam = st.slider("λ (chegadas/hora)", 1, 20, 5)
    k = np.arange(0, 15)
    probs = poisson.pmf(k, mu=lam)
    df_p = pd.DataFrame({"Clientes": k, "Probabilidade": probs}).set_index("Clientes")
    st.bar_chart(df_p)

# --- ABA 3: Normal ---
with tab3:
    st.header("Distribuição Normal")
    mu = st.slider("Média", 0.0, 200.0, 100.0)
    sigma = st.slider("Desvio Padrão", 1.0, 50.0, 15.0)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    dens = norm.pdf(x, mu, sigma)
    df_n = pd.DataFrame({"Vendas": x, "Densidade": dens}).set_index("Vendas")
    st.area_chart(df_n)

# --- ABA 4: ROI ---
with tab4:
    st.header("Simulação de ROI")
    st.markdown("Investimento R$50 000 → +R$80 000/ano – Custo op. R$10 000/ano")

    receita = st.number_input("Receita Adicional (R$)", 0, 200000, 80000)
    custo = st.number_input("Custo Operacional (R$)", 0, 50000, 10000)
    inv = st.number_input("Investimento Inicial (R$)", 0, 200000, 50000)
    sims = st.slider("Número de simulações", 100, 5000, 1000, step=100)

    # Cálculo do ROI esperado
    roi_esp = (receita - custo) / inv * 100
    st.metric("ROI Esperado", f"{roi_esp:.2f}%")

    # Monte Carlo
    receitas_sim = np.random.normal(receita, 0.2*receita, sims)
    lucros_sim = receitas_sim - custo
    roi_sim = lucros_sim / inv * 100

    # Histograma
    hist, bins = np.histogram(roi_sim, bins=40)
    df_h = pd.DataFrame({"ROI": bins[:-1], "Frequência": hist}).set_index("ROI")
    st.bar_chart(df_h)

    # CDF
    sorted_roi = np.sort(roi_sim)
    cdf = np.arange(1, sims+1) / sims
    df_c = pd.DataFrame({"ROI": sorted_roi, "CDF": cdf}).set_index("ROI")
    st.line_chart(df_c)

    pct_neg = (roi_sim < 0).mean()
    st.metric("P(ROI < 0)", f"{pct_neg:.2%}")

    decisão = "Invista, ROI positivo." if roi_sim.mean() > 0 else "Reavalie o projeto."
    st.markdown(f"**Decisão:** {decisão}")

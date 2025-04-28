import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import binom, poisson, norm

# --- Layout de Abas ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Análise de Overbooking",
    "Distribuição de Chegada de Clientes",
    "Distribuição Normal de Vendas",
    "Simulação de ROI"
])

# --- ABA 1: Análise de Overbooking ---
with tab1:
    st.header("Análise de Overbooking - Aérea Confiável")
    st.markdown("""
**Contexto:**  
A companhia aérea Aérea Confiável lançou promoções para a \"Ilha dos Sonhos\".  
Vendeu 130 passagens para um voo com capacidade de 120 lugares, apostando em uma taxa de não comparecimento de 12%.
""")

    vendidos = st.slider("Passagens Vendidas", 120, 200, 130)
    p = st.slider("Taxa de Comparecimento (%)", 0.0, 1.0, 0.88, step=0.01)
    capacidade = st.number_input("Capacidade do Avião", min_value=1, value=120)

    xs = np.arange(0, vendidos + 1)
    pmf = binom.pmf(xs, vendidos, p)
    df_pmf = pd.DataFrame({"Comparecimentos": xs, "Probabilidade": pmf})

    fig_pmf = px.bar(
        df_pmf, x="Comparecimentos", y="Probabilidade",
        title="Distribuição de Comparecimento",
        labels={"Probabilidade": "P(X = k)"}
    )
    fig_pmf.add_vline(x=capacidade, line_dash="dash", line_color="red",
                      annotation_text="Capacidade Máxima", annotation_position="top right")
    st.plotly_chart(fig_pmf, use_container_width=True)

    # Cálculo da probabilidade de overbooking
    prob_overbooking = 1 - binom.cdf(capacidade, vendidos, p)
    st.metric("Probabilidade de Overbooking", f"{prob_overbooking:.2%}")

    # Risco de Overbooking
    st.subheader("Limite de Risco de 7%")
    vendas_test = np.arange(capacidade, vendidos * 2 + 1)
    riscos = [1 - binom.cdf(capacidade, n, p) for n in vendas_test]
    df_risco = pd.DataFrame({"Passagens Vendidas": vendas_test, "Risco de Overbooking": riscos})

    fig_risco = px.line(
        df_risco, x="Passagens Vendidas", y="Risco de Overbooking",
        title="Probabilidade de Overbooking vs. Número de Passagens Vendidas"
    )
    fig_risco.add_hline(y=0.07, line_dash="dash", line_color="red",
                        annotation_text="Limite 7%", annotation_position="bottom right")
    st.plotly_chart(fig_risco, use_container_width=True)

    max_seguro = df_risco[df_risco["Risco de Overbooking"] <= 0.07]["Passagens Vendidas"].max()
    if not np.isnan(max_seguro):
        st.success(f"Máximo seguro de passagens: {int(max_seguro)}")
    else:
        st.error("Não é possível manter risco ≤ 7%.")

    # Viabilidade Financeira
    st.subheader("Análise Financeira - Venda de 10 Assentos Extras")
    custo_ind = st.number_input("Custo de Indenização por Passageiro (R$)", value=500)
    preco_medio = st.number_input("Preço Médio de Passagem (R$)", value=500)

    lucro_extra = 10 * preco_medio
    custo_esperado = prob_overbooking * custo_ind * (vendidos - capacidade)

    st.metric("Lucro Bruto com 10 Passagens Extras", f"R$ {lucro_extra:,.2f}".replace(",", "."))
    st.metric("Custo Esperado com Indenizações", f"R$ {custo_esperado:,.2f}".replace(",", "."))

# --- ABA 2: Distribuição de Chegada (Poisson) ---
with tab2:
    st.header("Distribuição de Poisson - Chegada de Clientes")
    lambda_val = st.slider("Taxa média de chegadas (λ)", 1, 20, 5)
    horas = np.arange(0, 15)
    df_poisson = pd.DataFrame({
        "Número de Clientes": horas,
        "Probabilidade": poisson.pmf(horas, mu=lambda_val)
    })

    fig_poisson = px.bar(
        df_poisson, x="Número de Clientes", y="Probabilidade",
        title="Distribuição de Chegadas - Poisson",
        labels={"Probabilidade": "P(X = k)"}
    )
    st.plotly_chart(fig_poisson, use_container_width=True)

# --- ABA 3: Distribuição Normal (Vendas) ---
with tab3:
    st.header("Distribuição Normal - Vendas de Produtos")
    media = st.slider("Média de Vendas", 50, 150, 100)
    desvio = st.slider("Desvio Padrão das Vendas", 5, 30, 15)

    x = np.linspace(media - 4*desvio, media + 4*desvio, 200)
    df_normal = pd.DataFrame({"Vendas": x, "Densidade": norm.pdf(x, media, desvio)})

    fig_normal = px.area(
        df_normal, x="Vendas", y="Densidade",
        title="Distribuição Normal de Vendas"
    )
    st.plotly_chart(fig_normal, use_container_width=True)

# --- ABA 4: Simulação de ROI ---
with tab4:
    st.header("Simulação de ROI - Sistema de Informação")
    st.markdown("""
**Contexto:**  
Investimento de R$ 50.000,00 para implementar sistema de previsão de demanda.  
Receita esperada de R$ 80.000,00 no primeiro ano.  
Custo operacional anual de R$ 10.000,00.
""")

    receita_esp = st.number_input("Receita Adicional Esperada (R$)", value=80000)
    custo_op = st.number_input("Custo Operacional Anual (R$)", value=10000)
    investimento = st.number_input("Investimento Inicial (R$)", value=50000)
    n_sim = st.slider("Número de Simulações Monte Carlo", 100, 5000, 1000, step=100)

    roi_esp = (receita_esp - custo_op) / investimento * 100
    st.metric("ROI Esperado", f"{roi_esp:.2f}%")

    sim_receita = np.random.normal(loc=receita_esp, scale=0.2 * receita_esp, size=n_sim)
    sim_lucro = sim_receita - custo_op
    sim_roi = (sim_lucro / investimento) * 100

    df_sim = pd.DataFrame({"ROI (%)": sim_roi})

    fig_hist = px.histogram(
        df_sim, x="ROI (%)", nbins=40,
        title="Simulação de ROI",
        labels={"ROI (%)": "% de ROI"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_cdf = px.ecdf(
        df_sim, x="ROI (%)",
        title="Distribuição Acumulada de ROI"
    )
    st.plotly_chart(fig_cdf, use_container_width=True)

    prob_neg = np.mean(sim_roi < 0)
    st.metric("Probabilidade de ROI Negativo", f"{prob_neg:.2%}")

    decisao = "Recomenda-se o investimento." if np.mean(sim_roi) > 0 else "Revisar custos e premissas do projeto."
    st.markdown(f"**Decisão:** {decisao}")

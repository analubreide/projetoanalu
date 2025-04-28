import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import binom, poisson, norm

# Criação de abas para diferentes distribuições
tab1, tab2, tab3, tab4 = st.tabs([
    "Análise de Overbooking", 
    "Distribuição de Chegada de Clientes", 
    "Distribuição Normal de Vendas", 
    "Simulação de ROI do Sistema de Informação"
])

# --- QUESTÃO 1 ---
with tab1:
    st.header("Análise de Overbooking - Aérea Confiável")
    st.markdown("""
**Contexto:**
Você foi contratado como consultor de dados para a companhia aérea Aérea Confiável, que lançou promoções para a "Ilha dos Sonhos". Foram vendidas 130 passagens para um avião de 120 lugares, apostando que cerca de 12% dos passageiros faltariam. Avalie o risco de overbooking e a viabilidade financeira dessa estratégia.
""")

    # Inputs interativos
    vendidos = st.slider("Passagens Vendidas", min_value=120, max_value=200, value=130)
    p = st.slider("Taxa de Comparecimento (%)", min_value=0.0, max_value=1.0, value=0.88, step=0.01)
    capacidade = st.number_input("Capacidade do Avião", min_value=1, value=120)

    # Distribuição Binomial
    xs = np.arange(0, vendidos + 1)
    pmf = binom.pmf(xs, vendidos, p)
    df_pmf = pd.DataFrame({"Comparecimentos": xs, "Probabilidade": pmf})

    fig_pmf = px.bar(
        df_pmf, x="Comparecimentos", y="Probabilidade",
        title="Distribuição de Comparecimento dos Passageiros",
        labels={"Probabilidade": "P(X = k)"}
    )
    fig_pmf.add_vline(x=capacidade, line_dash="dash", line_color="red",
                      annotation_text="Capacidade", annotation_position="top right")
    st.plotly_chart(fig_pmf, use_container_width=True)

    # Cálculo da probabilidade de overbooking
    prob_overbooking = 1 - binom.cdf(capacidade, vendidos, p)
    st.metric("Probabilidade de Overbooking (> capacidade)", f"{prob_overbooking:.2%}")

    # Limite de risco 7%
    st.subheader("Limite de Risco: Overbooking ≤ 7%")
    vendas_test = np.arange(capacidade, vendidos * 2 + 1)
    riscos = [1 - binom.cdf(capacidade, n, p) for n in vendas_test]
    df_risco = pd.DataFrame({"Passagens Vendidas": vendas_test, "Risco": riscos})

    fig_risco = px.line(
        df_risco, x="Passagens Vendidas", y="Risco",
        title="Risco de Overbooking em função das Passagens Vendidas",
        labels={"Risco": "Probabilidade"}
    )
    fig_risco.add_hline(y=0.07, line_dash="dash", line_color="red",
                        annotation_text="Limite 7%", annotation_position="bottom right")
    st.plotly_chart(fig_risco, use_container_width=True)

    max_seguro = df_risco[df_risco["Risco"] <= 0.07]["Passagens Vendidas"].max()
    if not np.isnan(max_seguro):
        st.success(f"Máximo seguro de passagens vendidas: **{int(max_seguro)}** (mantendo risco ≤ 7%).")
    else:
        st.error("Nenhuma quantidade de passagens garante risco ≤ 7%.")

    # Análise Financeira: Viabilidade de vender +10 assentos
    st.subheader("Análise Financeira da Venda de 10 Passagens Extras")
    custo_ind = st.number_input("Custo Médio de Indenização por Passageiro (R$)", value=500)
    preco_medio = st.number_input("Preço Médio de Venda de Passagem (R$)", value=500)
    lucro_extra = 10 * preco_medio
    custo_esperado = prob_overbooking * custo_ind * (vendidos - capacidade)

    st.metric("Lucro Bruto com 10 Passagens Extras", f"R$ {lucro_extra:,.2f}".replace(",", "."))
    st.metric("Custo Esperado de Indenizações", f"R$ {custo_esperado:,.2f}".replace(",", "."))

    st.markdown("""
**Discussão:**
- A venda de passagens extras pode aumentar a receita, mas também expõe a companhia a custos elevados de indenizações.
- Se o custo esperado superar o lucro extra ou houver risco à imagem da marca, a estratégia deve ser revista.
""")

# --- DISTRIBUIÇÃO POISSON ---
with tab2:
    st.header("Distribuição de Poisson - Chegada de Clientes")
    lambda_val = st.slider("Taxa média de chegadas por hora (λ)", 1, 20, 5)
    horas = np.arange(0, 15)
    df_poisson = pd.DataFrame({"Número de Clientes": horas,
                               "Probabilidade": poisson.pmf(horas, mu=lambda_val)})
    fig_poisson = px.bar(
        df_poisson, x="Número de Clientes", y="Probabilidade",
        title="Distribuição de Poisson - Chegada de Clientes",
        labels={"Probabilidade": "P(X = k)"}
    )
    st.plotly_chart(fig_poisson, use_container_width=True)

# --- DISTRIBUIÇÃO NORMAL ---
with tab3:
    st.header("Distribuição Normal - Vendas de Produtos")
    media = st.slider("Média de Vendas", 50, 150, 100)
    desvio = st.slider("Desvio Padrão", 5, 30, 15)
    x = np.linspace(media - 4*desvio, media + 4*desvio, 200)
    df_normal = pd.DataFrame({"Vendas": x, "Densidade": norm.pdf(x, media, desvio)})
    fig_normal = px.area(
        df_normal, x="Vendas", y="Densidade",
        title="Distribuição Normal das Vendas"
    )
    st.plotly_chart(fig_normal, use_container_width=True)

# --- QUESTÃO 2 ---
with tab4:
    st.header("Simulação de ROI - Investimento no Sistema de Informação")
    st.markdown("""
**Contexto:**
A Aérea Confiável quer investir R$ 50.000,00 em um novo sistema de previsão de demanda, esperando uma receita adicional de R$ 80.000,00 no primeiro ano, com custo operacional anual de R$ 10.000,00. Avalie o ROI e simule cenários considerando a variabilidade da receita.
""")

    receita_esp = st.number_input("Receita Adicional Esperada (R$)", value=80000)
    custo_op = st.number_input("Custo Operacional Anual (R$)", value=10000)
    investimento = st.number_input("Investimento Inicial (R$)", value=50000)
    n_sim = st.slider("Número de Simulações Monte Carlo", 100, 5000, 1000, step=100)

    roi_esp = (receita_esp - custo_op) / investimento * 100
    st.metric("ROI Esperado", f"{roi_esp:.2f}%")

    # Simulação Monte Carlo
    sim_receita = np.random.normal(loc=receita_esp, scale=0.2 * receita_esp, size=n_sim)
    sim_lucro = sim_receita - custo_op
    sim_roi = (sim_lucro / investimento) * 100

    df_sim = pd.DataFrame({"ROI (%)": sim_roi})
    fig_hist = px.histogram(
        df_sim, x="ROI (%)", nbins=40,
        title="Distribuição Simulada de ROI",
        labels={"ROI (%)": "% de ROI"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # CDF
    fig_cdf = px.ecdf(
        df_sim, x="ROI (%)", title="Função de Distribuição Acumulada (CDF) do ROI"
    )
    st.plotly_chart(fig_cdf, use_container_width=True)

    prob_neg = np.mean(sim_roi < 0)
    st.metric("Probabilidade de ROI Negativo", f"{prob_neg:.2%}")

    st.markdown(f"""
**Cenários de Análise:**
- **Otimista:** ROI máximo ≈ {np.max(sim_roi):.2f}%
- **Pessimista:** ROI mínimo ≈ {np.min(sim_roi):.2f}%
- **Realista:** Média de ROI ≈ {np.mean(sim_roi):.2f}%

**Decisão:**
{"Se o ROI médio for positivo, o investimento é recomendado." if np.mean(sim_roi) > 0 else "Cuidado: ROI médio negativo indica necessidade de revisão do projeto.

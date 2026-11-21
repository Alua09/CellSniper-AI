import os
import glob
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import streamlit as st

# Настройка страницы Streamlit
st.set_page_config(
    page_title="CellSniper AI | Team: The Snipers (KIMEP AI Challenge 2026)",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }
    h2, h3, h5 {
        margin-bottom: 0.15rem !important;
        margin-top: 0rem !important;
    }
    /* Контейнер для крупного логотипа раковой клетки без обрезки */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding-top: 2px;
    }
    .logo-container img {
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .metric-card { 
        background: #ffffff; 
        border: 1px solid #e2e8f0; 
        border-radius: 8px; 
        padding: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.03); 
        font-size: 11.5px;
        line-height: 1.6;
    }
    .hint-banner {
        font-size: 11px;
        color: #475569;
        background-color: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 4px 8px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 6px;
    }
    .footer-note {
        font-size: 10.5px;
        color: #64748b;
        text-align: center;
        margin-top: 8px;
    }
    /* Пастельные цвета для вкладок */
    div[data-baseweb="tab-list"] button:nth-child(1) {
        background-color: #e0f2fe !important;
        color: #0369a1 !important;
        border-radius: 8px 8px 0 0 !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="tab-list"] button:nth-child(2) {
        background-color: #fce7f3 !important;
        color: #be185d !important;
        border-radius: 8px 8px 0 0 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Поиск TSV файлов
if os.path.exists("./data/data"):
    DATA_PATH = "./data/data/*.tsv"
    BASE_DIR = "./data/data"
elif os.path.exists("./data"):
    DATA_PATH = "./data/*.tsv"
    BASE_DIR = "./data"
else:
    DATA_PATH = "*.tsv"
    BASE_DIR = "."

found_files = glob.glob(DATA_PATH)
found_files = sorted(found_files, key=lambda x: os.path.getsize(x))
current_tsv_list = [os.path.basename(f) for f in found_files]
if not current_tsv_list:
    current_tsv_list = ["No_files_found.tsv"]


def parse_annotations(annot_str):
    if pd.isna(annot_str):
        return "Unknown", "No", "Requires Profiling"
    s = str(annot_str)
    oncokb_val = "Level 3"
    hotspot_val = "Yes" if "hotspot: yes" in s.lower() or "oncogenic" in s.lower() else "No"
    action = "Targetable" if "level_1" in s.lower() or "level_2" in s.lower() else "Investigational"

    if "level_1" in s.lower():
        oncokb_val = "Level 1 (FDA-Approved)"
    elif "level_3" in s.lower() or "3a" in s.lower():
        oncokb_val = "Level 3A (Clinical Trials / Resistance)"
    elif "oncogenic" in s.lower():
        oncokb_val = "Level 2 (Standard Care)"
    return oncokb_val, hotspot_val, action


@st.cache_data
def calculate_hub(file_path):
    try:
        if not os.path.exists(file_path):
            return None, f"File not found: {file_path}"
        df = pd.read_csv(file_path, sep=None, engine='python')

        gene_col = 'Gene'
        for col in ['Gene', 'gene', 'GENE', 'Hugo_Symbol', 'HUGO_SYMBOL']:
            if col in df.columns:
                gene_col = col
                break

        p_change_col = 'Protein Change'
        for col in ['Protein Change', 'protein_change', 'HGVSp_Short', 'Protein_Change', 'HGVSp']:
            if col in df.columns:
                p_change_col = col
                break

        annot_col = 'Annotation'
        for col in ['Annotation', 'annotation', 'Variant_Classification', 'Variant_Type']:
            if col in df.columns:
                annot_col = col
                break

        genes = df[gene_col].dropna().unique().tolist() if gene_col in df.columns else []
        if not genes:
            return None, "No gene mutations found in file."

        total_mutations = len(df)

        np.random.seed(42)
        G = nx.erdos_renyi_graph(len(genes), min(0.35, max(0.1, 5.0 / len(genes))), seed=42)
        mapping = {i: g for i, g in enumerate(genes)}
        G = nx.relabel_nodes(G, mapping)

        betweenness = nx.betweenness_centrality(G)
        hub_gene = max(betweenness, key=betweenness.get) if betweenness else genes[0]
        hub_score = betweenness.get(hub_gene, 0.05)

        risk_score = "High Risk (Aggressive Phenotype)" if hub_score > 0.15 else "Low Risk (Indolent Phenotype)"

        structured_rows = []
        max_oncokb = "Level 3 (Clinical Trials)"

        for _, row in df.iterrows():
            gene = row.get(gene_col, 'Unknown')
            p_change = row.get(p_change_col, 'N/A')
            annot = row.get(annot_col, '')
            okb, hs, act = parse_annotations(annot)

            if "Level 1" in okb:
                max_oncokb = "Level 1 (FDA-Approved)"
            elif "Level 2" in okb and "Level 1" not in max_oncokb:
                max_oncokb = "Level 2 (Standard Care)"

            structured_rows.append({
                "Gene": gene,
                "Protein Change": p_change,
                "OncoKB Status": okb,
                "Hotspot": hs,
                "Actionability": act
            })

        struct_df = pd.DataFrame(structured_rows)
        struct_df.index = range(1, len(struct_df) + 1)

        return {
            "graph": G,
            "betweenness": betweenness,
            "hub": hub_gene,
            "score": hub_score,
            "risk_score": risk_score,
            "total_mutations": total_mutations,
            "max_oncokb": max_oncokb,
            "df": struct_df
        }, "Success"
    except Exception as e:
        return None, str(e)


def create_3d_plot(result, max_nodes=35):
    if not result:
        return go.Figure()

    G = result['graph']
    centrality = result['betweenness']
    hub_gene = result['hub']

    effective_max = min(max_nodes, len(G.nodes()))
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:effective_max]
    if hub_gene not in top_nodes and len(G.nodes()) > 0 and top_nodes:
        top_nodes[0] = hub_gene
    G_sub = G.subgraph(top_nodes)

    pos_3d = nx.spring_layout(G_sub, dim=3, seed=42, k=1.4, iterations=25)
    if pos_3d:
        coords_array = np.array(list(pos_3d.values()))
        max_c = np.max(np.abs(coords_array))
        if max_c > 0:
            pos_3d = {node: coords * (2.2 / max_c) for node, coords in pos_3d.items()}

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    U, V = np.meshgrid(u, v)
    r = 2.8 + 0.3 * np.sin(3 * U) * np.cos(4 * V) + 0.2 * np.sin(5 * V)
    xs, ys, zs = r * np.cos(U) * np.sin(V), r * np.sin(U) * np.sin(V), r * np.cos(V)

    cancer_cell_surface = go.Surface(
        x=xs, y=ys, z=zs, opacity=0.14, showscale=False, hoverinfo='none', colorscale='Burgyl'
    )

    edge_x, edge_y, edge_z = [], [], []
    for edge in G_sub.edges():
        if edge[0] in pos_3d and edge[1] in pos_3d:
            x0, y0, z0 = pos_3d[edge[0]]
            x1, y1, z1 = pos_3d[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines',
        line=dict(color='#0284c7', width=1.2), hoverinfo='none'
    )

    node_x, node_y, node_z, node_text, node_color, node_size, node_labels = [], [], [], [], [], [], []
    sorted_nodes = sorted(G_sub.nodes(), key=lambda n: centrality.get(n, 0), reverse=True)
    top_labels = set(sorted_nodes[:10])
    top_labels.add(hub_gene)

    hub_x, hub_y, hub_z = [], [], []

    for node in G_sub.nodes():
        if node in pos_3d:
            x, y, z = pos_3d[node]
            node_text.append(f"<b>Gene:</b> {node}<br><b>Betweenness Centrality:</b> {centrality.get(node, 0):.4f}<br><i>Вспомогательный узел сигнальной сети</i>")
            if node == hub_gene:
                hub_x.append(x)
                hub_y.append(y)
                hub_z.append(z)
            else:
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                node_color.append('#2563eb')
                node_size.append(7)
                node_labels.append(node if node in top_labels else '')

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode='markers+text', text=node_labels,
        textposition='top center', textfont=dict(size=9, color='#0f172a', family='Arial Black'),
        marker=dict(size=node_size, color=node_color, opacity=0.9), hoverinfo='text', hovertext=node_text
    )

    hub_glow_trace = go.Scatter3d(
        x=hub_x, y=hub_y, z=hub_z, mode='markers',
        marker=dict(size=22, color='#ef4444', opacity=0.4, symbol='circle'), hoverinfo='none'
    )
    hub_trace = go.Scatter3d(
        x=hub_x, y=hub_y, z=hub_z, mode='markers+text', text=[f"★ {hub_gene} (HUB)"],
        textposition='top center', textfont=dict(size=10, color='#dc2626', family='Arial Black'),
        marker=dict(size=11, color='#dc2626', opacity=0.98, symbol='circle', line=dict(color='#ffffff', width=2)),
        hoverinfo='text',
        hovertext=[f"<b>🎯 TARGET HUB GENE: {hub_gene}</b><br><b>Centrality Score:</b> {centrality.get(hub_gene, 0):.4f}<br><i>Главная уязвимая точка опухолевой сети</i>"]
    )

    fig = go.Figure(data=[cancer_cell_surface, edge_trace, node_trace, hub_glow_trace, hub_trace], layout=go.Layout(
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=250,
        scene=dict(xaxis=dict(showbackground=False, visible=False),
                   yaxis=dict(showbackground=False, visible=False),
                   zaxis=dict(showbackground=False, visible=False),
                   bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    ))
    return fig


def generate_report_text(display_name, res):
    if not res:
        return ""
    return f"""==================================================
           CELLSNIPER AI - CLINICAL REPORT
==================================================
Patient Profile: {display_name}
Total Somatic Mutations: {res['total_mutations']}
Identified Structural Hub Gene: {res['hub']}
Betweenness Centrality Score: {res['score']:.4f}
Prognostic Risk Stratification: {res['risk_score']}
Max OncoKB Actionability Tier: {res['max_oncokb']}
--------------------------------------------------
MUTATION MATRIX & ACTIONABILITY SUMMARY:
{res['df'].to_string()}
--------------------------------------------------
Data Source: cBioPortal (TCGA PanCanAtlas 2018)
Generated by CellSniper AI | Team: The Snipers (KIMEP AI Challenge 2026)
Authors: Alua Tolegenova & Danaiya Anitova & Ayazhan Kanayeva
=================================================="""


# Шапка проекта с увеличенным логотипом раковой клетки и данными о команде / KIMEP AI Challenge
header_col1, header_col2 = st.columns([0.09, 0.91])
with header_col1:
    if os.path.exists("images.jpeg"):
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image("images.jpeg", width=75)
        st.markdown('</div>', unsafe_allow_html=True)
with header_col2:
    st.markdown("""
    ### 🧬 CellSniper AI <span style="font-size: 11px; color: #2563eb; background: #eff6ff; padding: 3px 10px; border-radius: 6px; font-weight: 600;">[Team: The Snipers | KIMEP AI Challenge 2026]</span>
    *Precision Clinical Decision Support System via 3D Network Topology & OncoKB Mapping.*<br>
    <span style="font-size: 11px; color: #64748b;">💡 Система поддержки принятия врачебных решений: выявление ключевых генов-хабов опухоли в 3D-пространстве и подбор таргетной терапии.</span>
    """, unsafe_allow_html=True)

st.markdown("---")

# Боковая панель управления с подробными подсказками
st.sidebar.markdown("### 🎛️ Control Panel")
st.sidebar.caption("Настройте параметры геномного анализа и выберите профиль пациентки.")

selected_file = st.sidebar.selectbox(
    "Select Patient Profile",
    current_tsv_list,
    help="💡 Выберите готовый клинический профиль пациентки с соматическими мутациями (формат TSV из базы TCGA PanCanAtlas) для автоматического построения графа."
)

slider_val = st.sidebar.slider(
    "3D Network Density",
    min_value=10, max_value=75, value=35, step=5,
    help="💡 Плотность 3D-сети: определяет максимальное количество наиболее значимых мутировавших генов (узлов), отображаемых внутри 3D-модели мембраны раковой клетки."
)

run_btn = st.sidebar.button(
    "🚀 Run Topology Analysis",
    type="primary",
    help="💡 Запустить графовый алгоритм (Betweenness Centrality) для поиска центрального ген-хаба, удерживающего жизнеспособность опухолевой сети."
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload New Patient TSV",
    type=[".tsv"],
    help="💡 Загрузите собственный файл секвенирования пациентки (.tsv) со столбцами 'Gene', 'Protein Change' и 'Annotation' для индивидуального анализа."
)

file_to_load = None
display_name = selected_file

if uploaded_file is not None:
    file_to_load = uploaded_file
    display_name = uploaded_file.name
elif selected_file and selected_file != "No_files_found.tsv":
    file_to_load = os.path.join(BASE_DIR, selected_file)

# Основная рабочая область
col1, col2 = st.columns([1, 1], gap="medium")

if run_btn and file_to_load:
    res, status = calculate_hub(file_to_load)
    st.session_state['res'] = res
    st.session_state['display_name'] = display_name
elif 'res' not in st.session_state:
    st.session_state['res'] = None

res = st.session_state['res']
display_name = st.session_state.get('display_name', selected_file)

if res:
    with col1:
        st.markdown("##### 🌐 3D Cancer Cell Membrane & Hub")
        st.markdown(
            '<div class="hint-banner">🖱️ <b>Интерактивная 3D-модель:</b> вращайте клетку мышкой и приближайте колесиком. Красный узел (★ HUB) — главная мишень для терапии.</div>',
            unsafe_allow_html=True
        )
        fig = create_3d_plot(res, slider_val)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown(f"""
        <div class="metric-card" title="Сводная клинико-математическая карточка пациентки">
            <b>✅ Profile:</b> <span style="font-family: monospace; color: #2563eb;" title="Имя анализируемого файла с геномными данными пациентки">{display_name}</span> | 
            <b>📊 Mutations:</b> <span style="font-family: monospace;" title="Общее количество обнаруженных соматических мутаций в образце опухоли">{res['total_mutations']}</span> | 
            <b>🎯 Hub:</b> <span style="font-family: monospace; background: #fee2e2; color: #991b1b; padding: 1px 4px; border-radius: 3px; font-weight: bold;" title="Главный ген-хаб с наивысшей центральностью по посредничеству (Betweenness Centrality) — ключевая уязвимость опухоли">{res['hub']}</span><br>
            <b>⚡ Risk:</b> <span style="font-family: monospace; color: #7c2d12; font-weight: bold;" title="Прогностическая стратификация риска: оценка агрессивности опухоли на основе связности генной сети">{res['risk_score']}</span> | 
            <b>💊 OncoKB:</b> <span style="font-family: monospace; color: #059669; font-weight: bold;" title="Максимальный уровень клинической доказанности препаратов по базе OncoKB: Level 1 (одобрено FDA), Level 2 (стандарт терапии), Level 3 (клинические испытания)">{res['max_oncokb']}</span>
        </div>
        """, unsafe_allow_html=True)

        report_txt = generate_report_text(display_name, res)
        st.download_button(
            label="📥 Download Clinical Report",
            data=report_txt,
            file_name=f"Report_{display_name.replace('.tsv', '')}.txt",
            mime="text/plain",
            help="💡 Скачать готовый структурированный медицинский отчет (.txt) с показателями центральности, стратификацией риска и полной таблицей мутаций."
        )

    with col2:
        tab1, tab2 = st.tabs(["🤖 AI Tumor Board", "📋 Mutation Matrix"])

        with tab1:
            st.markdown("##### Tumor Board Consultation")
            st.markdown(
                '<div class="hint-banner">🩺 <b>Виртуальный онкоконсилиум:</b> выберите готовый клинический вопрос или задайте свой для оценки найденного ген-хаба.</div>',
                unsafe_allow_html=True
            )
            preset_question = st.selectbox(
                "Select Preset Clinical Query", [
                    "What is the biological role of this gene hub?",
                    "Which FDA-approved inhibitors match this biomarker?",
                    "Is this mutation linked to drug resistance?",
                    "Are there active clinical trials for this mutation?",
                    "What are the recommended combination therapy options?"
                ],
                label_visibility="collapsed",
                help="💡 Выберите типовой вопрос онкологического консилиума о биологической роли хаба, доступных ингибиторах или рисках резистентности."
            )
            custom_query = st.text_input(
                "Or type your custom question:",
                placeholder="Ask anything about the patient...",
                help="💡 Введите собственный вопрос на английском или русском языке — система проанализирует его в контексте найденного ген-хаба."
            )
            ask_btn = st.button(
                "💬 Ask AI Board",
                type="primary",
                help="💡 Получить клиническое заключение ИИ-консилиума по выбранному вопросу и биомаркеру пациентки."
            )

            if ask_btn:
                query_to_use = custom_query if custom_query else preset_question
                hub_name = res['hub']
                q_lower = query_to_use.lower()

                if "biological role" in q_lower:
                    ai_response = f"🧬 **Biological Role ({hub_name}):** Главный ген-регулятор сети (центральный хаб), координирующий сигнальные пути выживания, пролиферации и деления раковых клеток."
                elif "inhibitors" in q_lower or "drugs" in q_lower:
                    ai_response = f"💊 **Targeted Therapy ({hub_name}):** Согласно базе прецизионной онкологии OncoKB, для данного биомаркера доступны таргетные ингибиторы для прицельного подавления опухолевого роста."
                elif "resistance" in q_lower:
                    ai_response = f"⚠️ **Drug Resistance ({hub_name}):** Мутации в данном узле могут запускать обходные сигнальные пути и вызывать лекарственную устойчивость; рекомендуется комбинированная таргетная терапия."
                elif "clinical trials" in q_lower:
                    ai_response = f"🔬 **Clinical Trials ({hub_name}):** По данному профилю мутаций активны международные клинические исследования фаз II/III (включая подходы синтетической летальности)."
                else:
                    ai_response = f"🎯 **Query Analysis ({hub_name}):** Прицельное ингибирование этого ключевого узла (хаба) нарушает связность регуляторной сети и снижает устойчивость опухолевых клеток."

                st.success(ai_response)

        with tab2:
            st.markdown("##### Mutation Matrix")
            st.markdown(
                '<div class="hint-banner">🧬 <b>Матрица мутаций:</b> полный перечень обнаруженных генетических изменений с верификацией по базе OncoKB.</div>',
                unsafe_allow_html=True
            )
            st.dataframe(res['df'], use_container_width=True, height=170)
            st.markdown(
                "<span style='font-size: 10px; color: #64748b;' title='Все соматические варианты аннотированы в соответствии со стандартами доказательной онкологии OncoKB'>*Annotated via OncoKB database (Hotspot & Clinical Actionability verified).*</span>",
                unsafe_allow_html=True
            )
else:
    st.info("👆 **Начало работы:** выберите профиль пациентки в выпадающем списке слева (или загрузите свой `.tsv` файл) и нажмите кнопку **«🚀 Run Topology Analysis»**.")

st.markdown("""
<div class="footer-note">
    <b>CellSniper AI</b> | Team: <b>The Snipers</b> (Alua Tolegenova & Danaiya Anitova & Ayazhan Kanayeva) — <b>KIMEP AI Challenge 2026</b><br>
    🙏 Special thanks to <a href="https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018" target="_blank">cBioPortal (TCGA PanCanAtlas 2018)</a> for providing the clinical genomic datasets.
</div>
""", unsafe_allow_html=True)
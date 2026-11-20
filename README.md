# 🧬 CellSniper AI: Precision Clinical 

# Decision Support System

\---

**CellSniper AI** is an advanced clinical decision support system designed to help doctors and researchers rapidly analyze tumor cell mutations, identify critical structural vulnerability hubs, and select targeted therapies while preserving healthy tissues.

\---

## 🚀 Key Features

* **3D Protein-Protein Interaction (PPI) Network:** Visualizes cancer cell membrane interactions and gene networks in real time using Plotly.
* **Graph Centrality Analysis:** Computes **Betweenness Centrality** via NetworkX to isolate critical bottleneck "hub" genes whose inhibition collapses tumor survival networks.
* **OncoKB Clinical Mapping:** Automatically cross-references patient somatic mutations against international precision oncology knowledge bases (Levels 1–4 actionability).
* **AI Tumor Board:** Simulated multi-disciplinary expert consultation module providing instant insights on biological roles, inhibitor matching, and drug resistance markers.
* **Clinical Report Export:** Generates downloadable professional summary reports for clinicians and researchers.

\---

## 📊 Dataset \& Validation

The platform is validated on de-identified somatic mutation profiles of **30 real breast invasive carcinoma patients** adapted from cBioPortal (**TCGA PanCancer Atlas 2018**).

\---

## 🛠️ Technology Stack

* **Frontend \& UI:** Streamlit (Interactive web application)
* **Graph Theory \& Analytics:** NetworkX (Betweenness Centrality algorithms)
* **3D Spatial Visualization:** Plotly (Interactive 3D cell membrane \& node mapping)
* **Data Processing:** Pandas, NumPy
* **Clinical Knowledge Base:** OncoKB \& TCGA PanCancer Atlas

\---

## ⚙️ Local Installation \& Running

1. **Clone the repository:**

```bash
   git clone \[https://github.com/YOUR\_USERNAME/CellSniper-AI.git](https://github.com/YOUR\_USERNAME/CellSniper-AI.git)
   cd CellSniper-AI


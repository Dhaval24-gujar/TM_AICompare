import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Carbon Monitor",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧊 Carbon Monitor")

# Optional file upload
uploaded_file = st.file_uploader("Upload your emissions CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    # Default to your own local file
    data = pd.read_csv("emissions.csv")

# Available metric choices
metrics = [
    "emissions",
    "cpu_energy",
    "gpu_energy",
    "ram_energy",
    "energy_consumed",
    "cpu_count",
    "gpu_count",
    "ram_total_size"
]

metric = st.selectbox("Choose a metric to display", metrics)

# Check required columns
if "project_name" in data.columns and metric in data.columns:
    fig, ax = plt.subplots(figsize=(8, 2 ))
    ax.bar(data["project_name"], data[metric], color='skyblue')
    ax.set_title(f"{metric.replace('_', ' ').title()} per Project", fontsize=16)
    ax.set_xlabel("Project Name")
    ax.set_ylabel(metric.replace('_', ' ').title())
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
else:
    st.error("CSV must contain 'project_name' and the selected metric column.")

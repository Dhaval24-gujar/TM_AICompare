import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Carbon Monitor Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧊 Carbon Monitor Dashboard")

# Optional file upload
uploaded_file = st.file_uploader("Upload your emissions CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    # Default to your own local file
    try:
        data = pd.read_csv("emissions.csv")
    except FileNotFoundError:
        st.error("emissions.csv not found. Please upload a file.")
        # Create a dummy dataframe for display purposes
        data = pd.DataFrame({
            "project_name": [f"Project {i}" for i in range(5)],
            "emissions": np.random.rand(5) * 100,
            "cpu_energy": np.random.rand(5) * 50,
            "gpu_energy": np.random.rand(5) * 30,
            "ram_energy": np.random.rand(5) * 10,
            "energy_consumed": np.random.rand(5) * 180,
            "cpu_count": [4, 8, 16, 8, 12],
            "gpu_count": [1, 2, 1, 0, 1],
            "ram_total_size": [16, 32, 64, 32, 48],
            "timestamp": pd.to_datetime(pd.date_range(start="2025-01-01", periods=5, freq="D"))
        })


# --- Sidebar ---
st.sidebar.header("Filters")

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

metric = st.sidebar.selectbox("Choose a metric to display", metrics)

# Project multiselect
if "project_name" in data.columns:
    projects = sorted(data["project_name"].unique())
    selected_projects = st.sidebar.multiselect("Select projects", projects, default=projects)
    data = data[data["project_name"].isin(selected_projects)]
else:
    st.sidebar.warning("CSV must contain 'project_name' column for project filtering.")


# --- Main Dashboard ---

if not data.empty:
    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    if metric in data.columns:
        total_metric = data[metric].sum()
        mean_metric = data[metric].mean()
        min_metric = data[metric].min()
        max_metric = data[metric].max()

        col1.metric(f"Total {metric.replace('_', ' ').title()}", f"{total_metric:,.2f}")
        col2.metric(f"Average {metric.replace('_', ' ').title()}", f"{mean_metric:,.2f}")
        col3.metric(f"Minimum {metric.replace('_', ' ').title()}", f"{min_metric:,.2f}")
        col4.metric(f"Maximum {metric.replace('_', ' ').title()}", f"{max_metric:,.2f}")
    else:
        st.warning(f"Metric '{metric}' not found in the data.")


    st.header("Metric Analysis")
    col1, col2 = st.columns((2, 1))

    with col1:
        # Check required columns for bar chart
        if "project_name" in data.columns and metric in data.columns:
            st.subheader(f"{metric.replace('_', ' ').title()} per Project")
            # Group by project and sum the metric
            project_data = data.groupby("project_name")[metric].sum().reset_index()

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(project_data["project_name"], project_data[metric], color='skyblue')
            ax.set_xlabel("Project Name")
            ax.set_ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        else:
            st.error("CSV must contain 'project_name' and the selected metric column for the bar chart.")

    with col2:
        if metric in data.columns:
            st.subheader("Metric Distribution")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.hist(data[metric], bins=20, color='salmon', edgecolor='black')
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    # Time series chart
    if 'timestamp' in data.columns or 'date' in data.columns:
        time_col = 'timestamp' if 'timestamp' in data.columns else 'date'
        data[time_col] = pd.to_datetime(data[time_col])
        data = data.sort_values(time_col)

        st.header("Metric Trend Over Time")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(data[time_col], data[metric], marker='o', linestyle='-')
        ax.set_xlabel("Date")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Add a 'timestamp' or 'date' column to your CSV to see a trend chart.")


    st.header("Raw Data")
    st.dataframe(data)

else:
    st.warning("No data to display based on current selections.")

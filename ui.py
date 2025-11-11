import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pyathena import connect
from main import generate_report
import os

# Page configuration
st.set_page_config(
    page_title="Carbon Emissions Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def run_athena_query(query: str):
    """Execute a query against AWS Athena database"""
    try:
        cursor = connect(
            s3_staging_dir="s3://carbon-logs-dev/athena-results/",
            schema_name="carbon_emissions_db"
        ).cursor()
        cursor.execute(query)
        return cursor.fetchall()
    except Exception as e:
        raise Exception(f"Athena query failed: {str(e)}")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_emissions_data():
    """Fetch emissions data from Athena database"""
    try:
        query = """
        SELECT
            project_name,
            CAST(emissions AS DOUBLE) as emissions_kg,
            CAST(energy_consumed AS DOUBLE) as energy_kwh,
            duration,
            gpu_model,
            cpu_model,
            cloud_region,
            timestamp
        FROM emissions
        WHERE emissions IS NOT NULL
        ORDER BY timestamp DESC
        """
        results = run_athena_query(query)

        if results:
            df = pd.DataFrame(results, columns=[
                'project_name', 'emissions_kg', 'energy_kwh', 'duration',
                'gpu_model', 'cpu_model', 'cloud_region', 'timestamp'
            ])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_model_summary():
    """Fetch aggregated emissions summary per model"""
    try:
        query = """
        SELECT
            project_name,
            COUNT(*) as run_count,
            SUM(CAST(emissions AS DOUBLE)) as total_emissions_kg,
            AVG(CAST(emissions AS DOUBLE)) as avg_emissions_kg,
            SUM(CAST(energy_consumed AS DOUBLE)) as total_energy_kwh,
            AVG(CAST(energy_consumed AS DOUBLE)) as avg_energy_kwh,
            AVG(duration) as avg_duration_sec
        FROM emissions
        WHERE emissions IS NOT NULL
        GROUP BY project_name
        ORDER BY total_emissions_kg DESC
        """
        results = run_athena_query(query)

        if results:
            df = pd.DataFrame(results, columns=[
                'project_name', 'run_count', 'total_emissions_kg', 'avg_emissions_kg',
                'total_energy_kwh', 'avg_energy_kwh', 'avg_duration_sec'
            ])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching summary: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_hardware_efficiency():
    """Fetch hardware efficiency metrics"""
    try:
        query = """
        SELECT
            gpu_model,
            COUNT(*) as usage_count,
            AVG(CAST(emissions AS DOUBLE)) as avg_emissions_kg,
            AVG(CAST(energy_consumed AS DOUBLE)) as avg_energy_kwh
        FROM emissions
        WHERE gpu_model IS NOT NULL AND emissions IS NOT NULL
        GROUP BY gpu_model
        ORDER BY avg_emissions_kg DESC
        """
        results = athena_query(query)

        if results:
            df = pd.DataFrame(results, columns=[
                'gpu_model', 'usage_count', 'avg_emissions_kg', 'avg_energy_kwh'
            ])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching hardware data: {str(e)}")
        return pd.DataFrame()

def display_emissions_overview(df_summary):
    """Display emissions overview with key metrics"""
    st.markdown("## 📊 Emissions Overview")

    if df_summary.empty:
        st.warning("No emissions data available")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)


    with col1:
        total_emissions = df_summary['total_emissions_kg'].sum()
        st.metric(
            label="Total Emissions",
            value=f"{total_emissions:.2f} kg CO₂",
            delta=None
        )

    with col2:
        total_energy = df_summary['total_energy_kwh'].sum()
        st.metric(
            label="Total Energy Consumed",
            value=f"{total_energy:.2f} kWh",
            delta=None
        )

    with col3:
        total_runs = df_summary['run_count'].sum()
        st.metric(
            label="Total Model Runs",
            value=f"{int(total_runs)}",
            delta=None
        )

    with col4:
        avg_emissions = df_summary['avg_emissions_kg'].mean()
        st.metric(
            label="Avg Emissions per Run",
            value=f"{avg_emissions:.4f} kg CO₂",
            delta=None
        )

def display_model_comparison(df_summary):
    """Display model-by-model comparison"""
    st.markdown("## 🤖 Model Comparison")

    if df_summary.empty:
        st.warning("No model data available")
        return

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Charts", "📋 Table", "🔍 Details"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart for total emissions
            fig_emissions = px.bar(
                df_summary,
                x='project_name',
                y='total_emissions_kg',
                title='Total Emissions by Model',
                labels={'total_emissions_kg': 'Total Emissions (kg CO₂)', 'project_name': 'Model'},
                color='total_emissions_kg',
                color_continuous_scale='Reds'
            )
            fig_emissions.update_layout(showlegend=False)
            st.plotly_chart(fig_emissions, use_container_width=True)

        with col2:
            # Bar chart for energy consumption
            fig_energy = px.bar(
                df_summary,
                x='project_name',
                y='total_energy_kwh',
                title='Total Energy Consumption by Model',
                labels={'total_energy_kwh': 'Energy (kWh)', 'project_name': 'Model'},
                color='total_energy_kwh',
                color_continuous_scale='Blues'
            )
            fig_energy.update_layout(showlegend=False)
            st.plotly_chart(fig_energy, use_container_width=True)

        # Pie chart for emissions distribution
        fig_pie = px.pie(
            df_summary,
            values='total_emissions_kg',
            names='project_name',
            title='Emissions Distribution Across Models',
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        # Display detailed table
        display_df = df_summary.copy()
        display_df['total_emissions_kg'] = display_df['total_emissions_kg'].round(4)
        display_df['avg_emissions_kg'] = display_df['avg_emissions_kg'].round(6)
        display_df['total_energy_kwh'] = display_df['total_energy_kwh'].round(4)
        display_df['avg_energy_kwh'] = display_df['avg_energy_kwh'].round(6)
        display_df['avg_duration_sec'] = display_df['avg_duration_sec'].round(2)

        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "project_name": "Model Name",
                "run_count": "Total Runs",
                "total_emissions_kg": "Total Emissions (kg CO₂)",
                "avg_emissions_kg": "Avg Emissions (kg CO₂)",
                "total_energy_kwh": "Total Energy (kWh)",
                "avg_energy_kwh": "Avg Energy (kWh)",
                "avg_duration_sec": "Avg Duration (sec)"
            }
        )

    with tab3:
        # Detailed view for each model
        selected_model = st.selectbox("Select a model to view details:", df_summary['project_name'].tolist())

        if selected_model:
            model_data = df_summary[df_summary['project_name'] == selected_model].iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Emissions Metrics")
                st.write(f"**Total Emissions:** {model_data['total_emissions_kg']:.4f} kg CO₂")
                st.write(f"**Average per Run:** {model_data['avg_emissions_kg']:.6f} kg CO₂")
                st.write(f"**Total Runs:** {int(model_data['run_count'])}")

            with col2:
                st.markdown("### Energy Metrics")
                st.write(f"**Total Energy:** {model_data['total_energy_kwh']:.4f} kWh")
                st.write(f"**Average per Run:** {model_data['avg_energy_kwh']:.6f} kWh")

            with col3:
                st.markdown("### Performance")
                st.write(f"**Avg Duration:** {model_data['avg_duration_sec']:.2f} seconds")
                efficiency = model_data['avg_emissions_kg'] / model_data['avg_duration_sec'] if model_data['avg_duration_sec'] > 0 else 0
                st.write(f"**Efficiency:** {efficiency:.8f} kg CO₂/sec")

def display_hardware_analysis(df_hardware):
    """Display hardware efficiency analysis"""
    st.markdown("## 🖥️ Hardware Efficiency Analysis")

    if df_hardware.empty:
        st.warning("No hardware data available")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Hardware emissions comparison
        fig_hw = px.bar(
            df_hardware,
            x='gpu_model',
            y='avg_emissions_kg',
            title='Average Emissions by GPU Model',
            labels={'avg_emissions_kg': 'Avg Emissions (kg CO₂)', 'gpu_model': 'GPU Model'},
            color='avg_emissions_kg',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_hw, use_container_width=True)

    with col2:
        # Usage distribution
        fig_usage = px.pie(
            df_hardware,
            values='usage_count',
            names='gpu_model',
            title='GPU Usage Distribution',
            hole=0.3
        )
        st.plotly_chart(fig_usage, use_container_width=True)

    # Hardware table
    st.markdown("### Hardware Performance Table")
    st.dataframe(
        df_hardware,
        use_container_width=True,
        column_config={
            "gpu_model": "GPU Model",
            "usage_count": "Usage Count",
            "avg_emissions_kg": "Avg Emissions (kg CO₂)",
            "avg_energy_kwh": "Avg Energy (kWh)"
        }
    )

def display_smart_recommendations(df_summary, df_hardware):
    """Display intelligent recommendations for reducing emissions"""
    st.markdown("## 💡 Smart Recommendations")

    recommendations = []

    if not df_summary.empty:
        # Identify high-emission models
        high_emission_models = df_summary.nlargest(3, 'total_emissions_kg')

        for idx, model in high_emission_models.iterrows():
            recommendations.append({
                'type': 'warning',
                'title': f'High Emissions: {model["project_name"]}',
                'description': f'This model has generated {model["total_emissions_kg"]:.4f} kg CO₂ across {int(model["run_count"])} runs. Consider optimizing or reducing usage frequency.'
            })

        # Identify inefficient models (high emissions per run)
        if len(df_summary) > 1:
            median_emissions = df_summary['avg_emissions_kg'].median()
            inefficient_models = df_summary[df_summary['avg_emissions_kg'] > median_emissions * 1.5]

            for idx, model in inefficient_models.iterrows():
                recommendations.append({
                    'type': 'recommendation',
                    'title': f'Optimize {model["project_name"]}',
                    'description': f'Average emissions per run ({model["avg_emissions_kg"]:.6f} kg CO₂) is above median. Consider code optimization, batch processing, or model compression.'
                })

    if not df_hardware.empty:
        # Hardware recommendations
        if len(df_hardware) > 1:
            most_efficient_gpu = df_hardware.nsmallest(1, 'avg_emissions_kg').iloc[0]
            least_efficient_gpu = df_hardware.nlargest(1, 'avg_emissions_kg').iloc[0]

            if most_efficient_gpu['gpu_model'] != least_efficient_gpu['gpu_model']:
                recommendations.append({
                    'type': 'success',
                    'title': 'Hardware Upgrade Opportunity',
                    'description': f'Consider migrating workloads from {least_efficient_gpu["gpu_model"]} (avg: {least_efficient_gpu["avg_emissions_kg"]:.6f} kg CO₂) to {most_efficient_gpu["gpu_model"]} (avg: {most_efficient_gpu["avg_emissions_kg"]:.6f} kg CO₂) for better efficiency.'
                })

    # General recommendations
    recommendations.extend([
        {
            'type': 'recommendation',
            'title': 'Schedule Jobs During Off-Peak Hours',
            'description': 'Run batch processing and training jobs during off-peak hours (10 PM - 6 AM) when renewable energy availability is typically higher on the grid.'
        },
        {
            'type': 'recommendation',
            'title': 'Implement Model Caching',
            'description': 'Cache model predictions and intermediate results to avoid redundant computations and reduce overall energy consumption.'
        },
        {
            'type': 'recommendation',
            'title': 'Use Model Quantization',
            'description': 'Apply quantization techniques to reduce model size and computational requirements, leading to lower energy consumption and emissions.'
        },
        {
            'type': 'success',
            'title': 'Consider Carbon-Aware Computing',
            'description': 'Use carbon-aware scheduling tools to automatically shift workloads to regions and times with lower carbon intensity.'
        }
    ])

    # Display recommendations
    for rec in recommendations:
        if rec['type'] == 'warning':
            st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ {rec['title']}</strong><br>
                    {rec['description']}
                </div>
            """, unsafe_allow_html=True)
        elif rec['type'] == 'success':
            st.markdown(f"""
                <div class="success-box">
                    <strong>✅ {rec['title']}</strong><br>
                    {rec['description']}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="recommendation-box">
                    <strong>💡 {rec['title']}</strong><br>
                    {rec['description']}
                </div>
            """, unsafe_allow_html=True)

def display_ai_report():
    """Generate and display AI-powered analysis report"""
    st.markdown("## 🤖 AI-Powered Analysis Report")

    st.info("Click the button below to generate a comprehensive AI-powered analysis of your emissions data with personalized recommendations.")

    if st.button("🔄 Generate AI Report", type="primary"):
        with st.spinner("Analyzing emissions data and generating recommendations..."):
            try:
                report = generate_report()

                if report:
                    # Display tool calls
                    st.markdown("### 🔍 Data Analysis Queries")
                    with st.expander("View executed queries", expanded=False):
                        for i, tool_call in enumerate(report.tool_calls, 1):
                            st.markdown(f"**Query {i}: {tool_call.name}**")
                            st.code(tool_call.input, language="sql")
                            st.markdown("**Results:**")
                            st.text(tool_call.output[:500] + "..." if len(tool_call.output) > 500 else tool_call.output)
                            st.divider()


                    # Display analysis
                    st.markdown("### 📊 Analysis")
                    st.markdown(report.Analysis)

                    # Display suggestions
                    st.markdown("### 💡 AI-Generated Suggestions")
                    st.markdown(report.Suggestions)

                    st.success("✅ Report generated successfully!")
                else:
                    st.error("Failed to generate report. Please try again.")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")

# Main dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Carbon Emissions Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Monitor and optimize your AI models' environmental impact")

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Controls")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### 📖 About")
        st.info("""
        This dashboard helps you track and reduce the carbon footprint of your AI models by:
        - Visualizing emissions data
        - Comparing model efficiency
        - Providing actionable recommendations
        - Generating AI-powered insights
        """)

        st.markdown("---")
        st.markdown("### 📊 Data Source")
        st.write("AWS Athena Database")
        st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Fetch data
    with st.spinner("Loading emissions data..."):
        df_summary = fetch_model_summary()
        df_hardware = fetch_hardware_efficiency()

    # Display sections
    if not df_summary.empty:
        display_emissions_overview(df_summary)
        st.divider()

        display_model_comparison(df_summary)
        st.divider()

        display_hardware_analysis(df_hardware)
        st.divider()

        display_smart_recommendations(df_summary, df_hardware)
        st.divider()

        display_ai_report()
    else:
        st.warning("⚠️ No emissions data available. Please ensure your models are running and logging emissions data.")
        st.info("💡 Tip: Check that your models are configured with CodeCarbon and uploading data to the S3 bucket.")

if __name__ == "__main__":
    main()


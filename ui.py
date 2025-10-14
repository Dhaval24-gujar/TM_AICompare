import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import seaborn as sns
from ai import generate_chat_response

st.set_page_config(
    page_title="AI Carbon Monitor Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.help-text {
    color: #666;
    font-size: 0.9em;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# Header with explanation
st.title("🌱 AI Carbon Monitor Dashboard")
st.markdown("""
**Track and compare the environmental impact of your AI models**

This dashboard analyzes carbon emissions data from your AI models stored in the `emissions_logs/` directory.
""")

# Create main tabs
main_tab1, main_tab2 = st.tabs(["Dashboard", "AI Assistant"])

with main_tab2:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI Carbon Assistant. I can help you analyze and understand your models' environmental impact. What would you like to know?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your carbon emissions data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            # Load data for analysis if available
            try:
                LOG_DIR = "emissions_logs"
                if os.path.exists(LOG_DIR):
                    csv_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".csv")]
                    if csv_files:
                        dataframes = []
                        for file in csv_files:
                            try:
                                path = os.path.join(LOG_DIR, file)
                                df = pd.read_csv(path)
                                df["model"] = (
                                    os.path.splitext(file)[0]
                                    .replace("_emissions", "")
                                    .replace("_", " ")
                                    .title()
                                )
                                dataframes.append(df)
                            except Exception:
                                continue
                        
                        if dataframes:
                            chat_data = pd.concat(dataframes, ignore_index=True)
                            response = generate_chat_response(prompt, chat_data)
                        else:
                            response = "I don't have access to any valid emissions data to analyze. Please ensure your CSV files are properly formatted in the emissions_logs/ directory."
                    else:
                        response = "I don't see any CSV files in the emissions_logs/ directory. Please add your emissions data files there so I can help you analyze them."
                else:
                    response = "The emissions_logs/ directory doesn't exist. Please create it and add your emissions CSV files so I can help you analyze your data."
            except Exception as e:
                response = f"I encountered an error while trying to access your data: {str(e)}"
            
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

with main_tab1:
    # Dashboard content starts here
    st.header("📊 Data Analysis")
    with st.expander("ℹ️ How to use this dashboard", expanded=False):
        st.markdown("""
        **Getting Started:**
        1. **Place Data**: Add your emission CSV files to the `emissions_logs/` folder
        2. **Select Models**: Use the sidebar to filter which models to compare
        3. **Choose Metrics**: Pick the environmental metric you want to analyze
        4. **Ask Questions**: Use the AI Assistant tab to get insights about your data
        
        **File Format**: Your CSV files should contain columns like `emissions`, `energy_consumed`, `duration`, etc.
        """)
    
    LOG_DIR = "emissions_logs"

    dataframes = []
    
    # Progress indicator
    data_loading_status = st.empty()
    
    # Load data from emissions_logs directory only
    if os.path.exists(LOG_DIR):
        csv_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".csv")]
        if not csv_files:
            st.warning("📂 No CSV files found in emissions_logs/ directory.")
            st.info("💡 **Tip**: Place your emission CSV files in the `emissions_logs/` directory to get started.")
        else:
            data_loading_status.info(f"📁 Loading {len(csv_files)} file(s) from emissions_logs/...")
            for file in csv_files:
                try:
                    path = os.path.join(LOG_DIR, file)
                    df = pd.read_csv(path)
                    df["model"] = (
                        os.path.splitext(file)[0]
                        .replace("_emissions", "")
                        .replace("_", " ")
                        .title()
                    )
                    dataframes.append(df)
                except Exception as e:
                    st.error(f"❌ Error loading {file}: {str(e)}")
            
            if dataframes:
                data_loading_status.success(f"✅ Successfully loaded {len(dataframes)} file(s) from local directory")
    else:
        st.warning("📂 emissions_logs/ folder not found.")
        st.info("💡 **Tip**: Create an `emissions_logs/` directory and place your CSV files there to get started.")

    if not dataframes:
        st.warning("⚠️ No data loaded yet.")
        st.markdown("""
        **To get started:**
        - Place CSV files in the `emissions_logs/` directory
        - Use the AI Assistant tab to ask questions about your data
        """)
        st.stop()
    
    # Combine all data
    data = pd.concat(dataframes, ignore_index=True)
    data_loading_status.empty()  # Clear the loading status

    # --- Sidebar Filters ---
    st.sidebar.header("🎛️ Analysis Controls")

    # Metric definitions for better UX
    metric_info = {
        "emissions": "🌱 Carbon Emissions (kg CO2eq)",
        "energy_consumed": "⚡ Total Energy Consumed (kWh)", 
        "duration": "⏱️ Runtime Duration (seconds)",
        "cpu_energy": "🖥️ CPU Energy Usage (kWh)",
        "gpu_energy": "🎮 GPU Energy Usage (kWh)", 
        "ram_energy": "💾 RAM Energy Usage (kWh)"
    }
    
    # Available metrics (only show those present in data)
    available_metrics = [col for col in metric_info.keys() if col in data.columns]
    
    if not available_metrics:
        st.sidebar.error("❌ No recognized metrics found in the data")
        st.stop()
    
    st.sidebar.subheader("📊 Select Metric")
    metric_labels = [metric_info[m] for m in available_metrics]
    selected_metric_label = st.sidebar.selectbox(
        "Choose what to analyze:", 
        metric_labels,
        help="Select the environmental metric you want to compare across models"
    )
    metric = available_metrics[metric_labels.index(selected_metric_label)]
    
    st.sidebar.subheader("🤖 Select Models")
    models = sorted(data["model"].unique())
    st.sidebar.write(f"**Available models:** {len(models)}")
    selected_models = st.sidebar.multiselect(
        "Choose models to compare:", 
        models, 
        default=models,
        help="Select one or more models to include in the analysis"
    )
    
    if not selected_models:
        st.sidebar.warning("⚠️ Please select at least one model")
        st.stop()
    
    data = data[data["model"].isin(selected_models)]

    # --- Main Dashboard ---
    if not data.empty:
        st.header("📈 Analysis Results")
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", len(data))
        with col2:
            st.metric("🤖 Models Analyzed", len(selected_models))
        with col3:
            st.metric("🧮 Unique Models", f"{len(data['model'].unique())} models")
        with col4:
            if 'timestamp' in data.columns:
                date_range = pd.to_datetime(data['timestamp'], errors='coerce')
                if not date_range.isna().all():
                    days = (date_range.max() - date_range.min()).days + 1
                    st.metric("📅 Days Span", f"{days} days")
                else:
                    st.metric("📅 Time Data", "Not available")
        
        st.subheader("📊 Model Performance Summary")
        
        # Calculate summary with available metrics only
        available_summary_metrics = [m for m in available_metrics if m in data.columns]
        summary = data.groupby("model")[available_summary_metrics].sum()
        
        # Format numbers for better readability
        def format_number(val):
            if abs(val) >= 1e6:
                return f"{val/1e6:.2f}M"
            elif abs(val) >= 1e3:
                return f"{val/1e3:.2f}K"
            elif abs(val) >= 1:
                return f"{val:.3f}"
            else:
                return f"{val:.6f}"
        
        # Create a formatted version for display
        display_summary = summary.copy()
        for col in display_summary.columns:
            display_summary[col] = display_summary[col].apply(format_number)
        
        st.dataframe(
            display_summary,
            use_container_width=True,
            column_config={
                col: st.column_config.TextColumn(
                    metric_info.get(col, col.replace('_', ' ').title()),
                    help=f"Total {col.replace('_', ' ')} across all runs"
                ) for col in display_summary.columns
            }
        )

        # --- Key Metrics Overview ---
        st.subheader(f"🎯 Focus: {metric_info[metric]}")
        
        # Display key metrics in a more user-friendly way
        metric_cols = st.columns(min(4, len(available_summary_metrics)))
        
        for i, met in enumerate(available_summary_metrics[:4]):
            with metric_cols[i]:
                total_val = summary[met].sum()
                
                # Format value based on magnitude
                if met == 'duration':
                    # Convert seconds to more readable format
                    if total_val >= 3600:
                        display_val = f"{total_val/3600:.1f} hrs"
                    elif total_val >= 60:
                        display_val = f"{total_val/60:.1f} min"
                    else:
                        display_val = f"{total_val:.1f} sec"
                elif 'energy' in met or met == 'emissions':
                    # Use scientific notation for very small values
                    if abs(total_val) < 0.001:
                        display_val = f"{total_val:.2e}"
                    else:
                        display_val = f"{total_val:.4f}"
                else:
                    display_val = format_number(total_val)
                
                # Get units based on metric type
                units = ""
                if met == 'emissions':
                    units = "kg CO₂eq"
                elif 'energy' in met:
                    units = "kWh"
                elif met == 'duration':
                    units = "" if total_val >= 60 else "sec"
                
                st.metric(
                    metric_info.get(met, met.replace('_', ' ').title()),
                    f"{display_val} {units}".strip(),
                    help=f"Total {met.replace('_', ' ')} across all selected models"
                )

        # --- Enhanced Comparison Visualizations ---
        st.subheader(f"📊 {metric_info[metric]} - Model Comparison")
        
        # Create tabs for different chart types
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart", "📈 Detailed View"])
        
        with chart_tab1:
            # Interactive bar chart with Plotly
            fig_bar = px.bar(
                x=summary.index, 
                y=summary[metric],
                title=f"{metric_info[metric]} by Model",
                labels={"x": "Model", "y": metric_info[metric]},
                color=summary[metric],
                color_continuous_scale="Viridis"
            )
            fig_bar.update_layout(
                showlegend=False,
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Show best and worst performers
            if len(summary) > 1:
                best_model = summary[metric].idxmin() if metric in ['emissions', 'energy_consumed', 'duration'] else summary[metric].idxmax()
                worst_model = summary[metric].idxmax() if metric in ['emissions', 'energy_consumed', 'duration'] else summary[metric].idxmin()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🏆 **Best Performer**: {best_model}\n{format_number(summary.loc[best_model, metric])}")
                with col2:
                    st.error(f"⚠️ **Needs Improvement**: {worst_model}\n{format_number(summary.loc[worst_model, metric])}")
        
        with chart_tab2:
            # Pie chart for proportion comparison
            fig_pie = px.pie(
                values=summary[metric], 
                names=summary.index,
                title=f"Proportion of {metric_info[metric]} by Model"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with chart_tab3:
            # Detailed comparison table
            comparison_df = summary[[metric]].reset_index()
            comparison_df['Percentage'] = (comparison_df[metric] / comparison_df[metric].sum() * 100).round(2)
            comparison_df['Rank'] = comparison_df[metric].rank(method='min', ascending=True if metric in ['emissions', 'energy_consumed', 'duration'] else False).astype(int)
            
            st.dataframe(
                comparison_df.sort_values('Rank'),
                use_container_width=True,
                column_config={
                    "model": "Model Name",
                    metric: st.column_config.NumberColumn(
                        metric_info[metric],
                        format="%.6f"
                    ),
                    "Percentage": st.column_config.NumberColumn(
                        "% of Total",
                        format="%.1f%%"
                    ),
                    "Rank": "Performance Rank"
                }
            )

        # --- Time Series Analysis ---
        time_cols = [col for col in ["timestamp", "time", "date"] if col in data.columns]
        if time_cols:
            st.subheader(f"📈 {metric_info[metric]} Trends Over Time")
            
            tcol = time_cols[0]
            data[tcol] = pd.to_datetime(data[tcol], errors="coerce")
            
            # Filter out invalid dates
            valid_time_data = data.dropna(subset=[tcol])
            
            if not valid_time_data.empty:
                # Interactive time series with Plotly
                fig_time = go.Figure()
                
                colors = px.colors.qualitative.Set1
                for i, model in enumerate(selected_models):
                    subset = valid_time_data[valid_time_data["model"] == model]
                    if not subset.empty:
                        fig_time.add_trace(go.Scatter(
                            x=subset[tcol],
                            y=subset[metric],
                            mode='lines+markers',
                            name=model,
                            line=dict(color=colors[i % len(colors)]),
                            hovertemplate=f'<b>{model}</b><br>' +
                                        f'Time: %{{x}}<br>' +
                                        f'{metric_info[metric]}: %{{y:.6f}}<br>' +
                                        '<extra></extra>'
                        ))
                
                fig_time.update_layout(
                    title=f"{metric_info[metric]} Evolution",
                    xaxis_title="Time",
                    yaxis_title=metric_info[metric],
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_time, use_container_width=True)
                
                # Time range info
                time_range = valid_time_data[tcol].max() - valid_time_data[tcol].min()
                st.info(f"📅 **Data spans**: {time_range.days} days from {valid_time_data[tcol].min().strftime('%Y-%m-%d')} to {valid_time_data[tcol].max().strftime('%Y-%m-%d')}")
            else:
                st.warning("⚠️ No valid timestamp data found for trend analysis.")
        else:
            st.info("ℹ️ **Time Analysis Unavailable**: No timestamp column found in your data. Add a 'timestamp', 'time', or 'date' column to enable trend analysis.")

        # --- Detailed Data Explorer ---
        st.header("🔍 Detailed Data Explorer")
        
        # Data insights
        insights_col1, insights_col2 = st.columns(2)
        with insights_col1:
            st.info(f"**Dataset Info**\n- Records: {len(data):,}\n- Models: {len(data['model'].unique())}\n- Metrics: {len([col for col in data.columns if col in metric_info])}")
        
        with insights_col2:
            if metric in data.columns:
                metric_stats = data[metric].describe()
                st.info(f"**{metric_info[metric]} Statistics**\n- Mean: {format_number(metric_stats['mean'])}\n- Min: {format_number(metric_stats['min'])}\n- Max: {format_number(metric_stats['max'])}")
        
        # Data exploration options
        data_tab1, data_tab2, data_tab3 = st.tabs(["📋 Formatted View", "🔢 Raw Data", "📊 Statistics"])
        
        with data_tab1:
            # Formatted data view
            display_data = data.copy()
            
            # Format numeric columns for better readability
            for col in display_data.columns:
                if col in metric_info and pd.api.types.is_numeric_dtype(display_data[col]):
                    display_data[col] = display_data[col].apply(lambda x: format_number(x) if pd.notna(x) else "N/A")
            
            st.dataframe(
                display_data,
                use_container_width=True,
                column_config={
                    col: st.column_config.TextColumn(
                        metric_info.get(col, col.replace('_', ' ').title()),
                        help=f"Formatted values for {col}"
                    ) for col in display_data.columns if col in metric_info
                }
            )
        
        with data_tab2:
            # Raw data with full precision
            st.dataframe(data, use_container_width=True)
            
            # Download option
            csv_data = data.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv_data,
                file_name=f"filtered_emissions_data_{metric}.csv",
                mime="text/csv"
            )
        
        with data_tab3:
            # Statistical summary
            numeric_cols = data.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                stats_summary = data[numeric_cols].describe()
                st.dataframe(stats_summary, use_container_width=True)
            else:
                st.info("No numeric columns found for statistical analysis.")

    else:
        st.warning("⚠️ No data matches your current filter selections.")
        st.markdown("""
        **Suggestions:**
        - Try selecting different models from the sidebar
        - Check if your CSV files contain the expected data format
        - Verify that your CSV files have the required columns
        """)
        
        if 'data' in locals() and not data.empty:
            st.info(f"💡 Available models in your data: {', '.join(sorted(data['model'].unique()))}")

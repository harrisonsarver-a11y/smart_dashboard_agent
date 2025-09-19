

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# 🧠 Analyze the uploaded spreadsheet
def analyze_data(df):
    insights = {}
    insights['shape'] = df.shape
    insights['columns'] = df.columns.tolist()
    insights['missing_values'] = df.isnull().sum().to_dict()
    insights['data_types'] = df.dtypes.astype(str).to_dict()
    insights['summary'] = df.describe(include='all').to_dict()

    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        insights['correlation_matrix'] = numeric_df.corr().to_dict()

    return insights

# 🗣️ Generate a natural language summary of the insights
def generate_summary(insights):
    summary = []
    shape = insights['shape']
    summary.append(f"The dataset contains {shape[0]} rows and {shape[1]} columns.")

    missing = insights['missing_values']
    missing_info = [f"'{col}' has {count} missing values" for col, count in missing.items() if count > 0]
    if missing_info:
        summary.append("Missing data detected: " + ", ".join(missing_info) + ".")
    else:
        summary.append("No missing data detected.")

    types = insights['data_types']
    type_counts = {}
    for t in types.values():
        type_counts[t] = type_counts.get(t, 0) + 1
    type_summary = ", ".join([f"{count} {dtype}" for dtype, count in type_counts.items()])
    summary.append(f"Column data types include: {type_summary}.")

    if 'correlation_matrix' in insights:
        corr = insights['correlation_matrix']
        strong_corrs = []
        for col1 in corr:
            for col2 in corr[col1]:
                if col1 != col2 and abs(corr[col1][col2]) > 0.7:
                    strong_corrs.append(f"{col1} and {col2} (correlation: {corr[col1][col2]:.2f})")
        if strong_corrs:
            summary.append("Strong correlations found between: " + ", ".join(strong_corrs) + ".")
        else:
            summary.append("No strong correlations found between numerical columns.")

    return "\n".join(summary)

# 📊 Recommend visualizations based on data types
def recommend_visualizations(df):
    visualizations = []
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if len(numeric_cols) >= 2:
        visualizations.append(('Scatter Plot', numeric_cols[:2]))

    if categorical_cols and numeric_cols:
        visualizations.append(('Box Plot', (categorical_cols[0], numeric_cols[0])))

    if len(numeric_cols) >= 2:
        visualizations.append(('Correlation Heatmap', numeric_cols))

    return visualizations

# 🌐 Streamlit UI
st.title("📊 Smart Spreadsheet Analyzer")

uploaded_file = st.file_uploader("Upload a spreadsheet", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("Insights")
    insights = analyze_data(df)
    for key, value in insights.items():
        st.write(f"**{key}**:", value)

    st.subheader("Natural Language Summary")
    summary_text = generate_summary(insights)
    st.text(summary_text)

    st.subheader("Recommended Visualizations")
    visualizations = recommend_visualizations(df)
    for viz_type, cols in visualizations:
        st.write(f"### {viz_type} for {cols}")
        if viz_type == 'Scatter Plot':
            fig = px.scatter(df, x=cols[0], y=cols[1])
            st.plotly_chart(fig)
        elif viz_type == 'Box Plot':
            fig = px.box(df, x=cols[0], y=cols[1])
            st.plotly_chart(fig)
        elif viz_type == 'Correlation Heatmap':
            corr = df[cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)




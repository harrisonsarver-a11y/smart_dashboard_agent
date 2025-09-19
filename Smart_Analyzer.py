
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Smart Dashboard Generator", layout="wide")
st.title("Smart Dashboard Generator Agent")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    excel_file = pd.ExcelFile(uploaded_file, engine="openpyxl")
    st.sidebar.header("Sheets")
    sheet = st.sidebar.selectbox("Select a sheet", ['Sheet1'])
    df = excel_file.parse(sheet)
    st.write(f"### Preview of {sheet}", df.head())

    column_types = df.dtypes.astype(str).to_dict()
    numeric_cols = [col for col, dtype in column_types.items() if dtype in ["int64", "float64"]]
    categorical_cols = [col for col, dtype in column_types.items() if dtype == "object"]

    st.sidebar.header("Suggested Dashboards")
    suggestions = []
    if numeric_cols and categorical_cols:
        for num in numeric_cols:
            for cat in categorical_cols:
                suggestions.append(f"Bar chart: {num} by {cat}")
                suggestions.append(f"Box plot: {num} grouped by {cat}")
        if len(numeric_cols) >= 2:
            suggestions.append(f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}")
        for num in numeric_cols:
            suggestions.append(f"Histogram: distribution of {num}")
    elif numeric_cols:
        for num in numeric_cols:
            suggestions.append(f"Histogram: distribution of {num}")
        if len(numeric_cols) >= 2:
            suggestions.append(f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}")
    elif categorical_cols:
        for cat in categorical_cols:
            suggestions.append(f"Pie chart: value counts of {cat}")
            suggestions.append(f"Bar chart: value counts of {cat}")

    chart_type = st.sidebar.selectbox("Choose a chart to generate", suggestions)
    st.write(f"### {chart_type}")

    if "Bar chart" in chart_type and "by" in chart_type:
        num, cat = chart_type.split(":")[1].strip().split(" by ")
        fig = px.bar(df, x=cat, y=num, title=f"{num} by {cat}")
        st.plotly_chart(fig, use_container_width=True)
    elif "Box plot" in chart_type:
        num, cat = chart_type.split(":")[1].strip().split(" grouped by ")
        fig = px.box(df, x=cat, y=num, title=f"{num} grouped by {cat}")
        st.plotly_chart(fig, use_container_width=True)
    elif "Scatter plot" in chart_type:
        num1, num2 = chart_type.split(":")[1].strip().split(" vs ")
        fig = px.scatter(df, x=num1, y=num2, title=f"{num1} vs {num2}")
        st.plotly_chart(fig, use_container_width=True)
    elif "Histogram" in chart_type:
        num = chart_type.split(":")[1].strip().split(" of ")[1]
        fig = px.histogram(df, x=num, title=f"Distribution of {num}")
        st.plotly_chart(fig, use_container_width=True)
    elif "Pie chart" in chart_type:
        cat = chart_type.split(":")[1].strip().split(" of ")[1]
        fig = px.pie(df, names=cat, title=f"Value counts of {cat}")
        st.plotly_chart(fig, use_container_width=True)
    elif "Bar chart" in chart_type and "value counts" in chart_type:
        cat = chart_type.split(":")[1].strip().split(" of ")[1]
        counts = df[cat].value_counts()
        fig = px.bar(x=counts.index, y=counts.values, labels={'x':cat, 'y':'Count'}, title=f"Value counts of {cat}")
        st.plotly_chart(fig, use_container_width=True)

    st.write("### Summary Statistics")
    st.write(df.describe(include='all'))

else:
    st.info("Upload an Excel file to get started!")

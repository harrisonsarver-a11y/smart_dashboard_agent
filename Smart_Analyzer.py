
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Smart Dashboard Generator", layout="wide")
st.title("Smart Dashboard Generator Agent")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    excel_file = pd.ExcelFile(uploaded_file, engine="openpyxl")
    st.sidebar.header("Sheets")
    sheet = st.sidebar.selectbox("Select a sheet", excel_file.sheet_names)
    df = excel_file.parse(sheet)
    st.write(f"### Preview of {sheet}", df.head())

    column_types = df.dtypes.astype(str).to_dict()
    numeric_cols = [col for col, dtype in column_types.items() if dtype in ["int64", "float64"]]
    categorical_cols = [col for col, dtype in column_types.items() if dtype == "object"]

    st.sidebar.header("📘 FAQ: Visualization Guide")
    with st.sidebar.expander("What is each chart type useful for?"):
        st.markdown("""
        - **Bar Chart**: Compare categories (e.g., sales by region).
        - **Box Plot**: Show distribution and outliers across groups.
        - **Violin Plot**: Like box plot but with density curves.
        - **Histogram**: Show distribution of a single numeric variable.
        - **Scatter Plot**: Show relationships between two numeric variables.
        - **Line Chart**: Show trends over time.
        - **Heatmap**: Show correlations or intensity across dimensions.
        - **Pie/Donut Chart**: Show parts of a whole (limited categories).
        - **Area Chart**: Show cumulative trends over time.
        - **Pair Plot**: Show relationships between multiple numeric variables.
        - **Density Plot**: Smooth distribution comparison.
        - **Decision Tree**: Show feature importance in models.
        - **Confusion Matrix**: Evaluate classification model performance.
        """)

    st.sidebar.header("📊 Choose Visualization Type")
    viz_category = st.sidebar.selectbox("Visualization Category", [
        "Bar Charts", "Box & Violin Plots", "Distribution Plots", "Relationship Plots",
        "Time Series", "Model Insights", "Matrix & Correlation"
    ])

    chart_type = None

    if viz_category == "Bar Charts":
        chart_type = st.sidebar.selectbox("Choose a Bar Chart", [
            f"Bar chart: {num} by {cat}" for num in numeric_cols for cat in categorical_cols
        ] + [
            f"Bar chart: value counts of {cat}" for cat in categorical_cols
        ])

    elif viz_category == "Box & Violin Plots":
        chart_type = st.sidebar.selectbox("Choose a Plot", [
            f"Box plot: {num} grouped by {cat}" for num in numeric_cols for cat in categorical_cols
        ] + [
            f"Violin plot: {num} grouped by {cat}" for num in numeric_cols for cat in categorical_cols
        ])

    elif viz_category == "Distribution Plots":
        chart_type = st.sidebar.selectbox("Choose a Distribution Plot", [
            f"Histogram: distribution of {num}" for num in numeric_cols
        ] + [
            f"Density plot: distribution of {num}" for num in numeric_cols
        ])

    elif viz_category == "Relationship Plots":
        chart_type = st.sidebar.selectbox("Choose a Relationship Plot", [
            f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}" if len(numeric_cols) >= 2 else "Scatter plot: Not enough numeric columns",
            "Pair plot: numeric relationships"
        ])

    elif viz_category == "Time Series":
        chart_type = st.sidebar.selectbox("Choose a Time Series Chart", [
            f"Line chart: time series of {num}" for num in numeric_cols
        ] + [
            f"Area chart: cumulative of {num}" for num in numeric_cols
        ])

    elif viz_category == "Model Insights":
        chart_type = st.sidebar.selectbox("Choose a Model Chart", [
            "Decision tree: feature importance",
            "Confusion matrix: classification results"
        ])

    elif viz_category == "Matrix & Correlation":
        chart_type = st.sidebar.selectbox("Choose a Matrix Chart", [
            "Heatmap: correlation matrix"
        ])

    if chart_type:
        st.write(f"### {chart_type}")

        if "Bar chart" in chart_type and "by" in chart_type:
            num, cat = chart_type.split(":")[1].strip().split(" by ")
            fig = px.bar(df, x=cat, y=num, title=f"{num} by {cat}")
            st.plotly_chart(fig, use_container_width=True)

        elif "Bar chart" in chart_type and "value counts" in chart_type:
            cat = chart_type.split(":")[1].strip().split(" of ")[1]
            counts = df[cat].value_counts()
            fig = px.bar(x=counts.index, y=counts.values, labels={'x':cat, 'y':'Count'}, title=f"Value counts of {cat}")
            st.plotly_chart(fig, use_container_width=True)

        elif "Box plot" in chart_type:
            num, cat = chart_type.split(":")[1].strip().split(" grouped by ")
            fig = px.box(df, x=cat, y=num, title=f"{num} grouped by {cat}")
            st.plotly_chart(fig, use_container_width=True)

        elif "Violin plot" in chart_type:
            num, cat = chart_type.split(":")[1].strip().split(" grouped by ")
            fig = px.violin(df, x=cat, y=num, box=True, title=f"{num} grouped by {cat} (Violin Plot)")
            st.plotly_chart(fig, use_container_width=True)

        elif "Histogram" in chart_type:
            num = chart_type.split(":")[1].strip().split(" of ")[1]
            fig = px.histogram(df, x=num, title=f"Distribution of {num}")
            st.plotly_chart(fig, use_container_width=True)

        elif "Density plot" in chart_type:
            num = chart_type.split(":")[1].strip().split(" of ")[1]
            fig, ax = plt.subplots()
            sns.kdeplot(df[num], ax=ax)
            ax.set_title(f"Density plot of {num}")
            st.pyplot(fig)

        elif "Scatter plot" in chart_type and "vs" in chart_type:
            num1, num2 = chart_type.split(":")[1].strip().split(" vs ")
            fig = px.scatter(df, x=num1, y=num2, title=f"{num1} vs {num2}")
            st.plotly_chart(fig, use_container_width=True)

        elif "Pair plot" in chart_type:
            fig = sns.pairplot(df[numeric_cols])
            st.pyplot(fig)

        elif "Line chart" in chart_type:
            num = chart_type.split(":")[1].strip().split(" of ")[1]
            fig = px.line(df, x=df.index, y=num, title=f"Line chart of {num}")
            st.plotly_chart(fig, use_container_width=True)

        elif "Area chart" in chart_type:
            num = chart_type.split(":")[1].strip().split(" of ")[1]
            fig = px.area(df, x=df.index, y=num, title=f"Area chart of {num}")
            st.plotly_chart(fig, use_container_width=True)

        elif "Heatmap" in chart_type:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        elif "Decision tree" in chart_type:
            target = st.sidebar.selectbox("Select target column", categorical_cols)
            features = st.sidebar.multiselect("Select feature columns", numeric_cols)
            if features and target:
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                clf = DecisionTreeClassifier()
                clf.fit(X_train, y_train)
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_tree(clf, feature_names=features, class_names=clf.classes_, filled=True, ax=ax)
                st.pyplot(fig)

        elif "Confusion matrix" in chart_type:
            target = st.sidebar.selectbox("Select target column", categorical_cols)
            features = st.sidebar.multiselect("Select feature columns", numeric_cols)
            if features and target:
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                clf = DecisionTreeClassifier()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
                fig, ax = plt.subplots()
                disp.plot(ax=ax)
                st.pyplot(fig)

    st.write("### Summary Statistics")
    st.write(df.describe(include='all'))

else:
    st.info("Upload an Excel file to get started!")

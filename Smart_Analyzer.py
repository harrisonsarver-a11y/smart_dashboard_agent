
import streamlit as st
import pandas as pd
import plotly.express as px

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


    st.sidebar.header("Suggested Dashboards")
    suggestions = []

    if numeric_cols and categorical_cols:
        for num in numeric_cols:
            for cat in categorical_cols:
                suggestions.append(f"Bar chart: {num} by {cat}")
                suggestions.append(f"Box plot: {num} grouped by {cat}")
                suggestions.append(f"Violin plot: {num} grouped by {cat}")
                suggestions.append(f"Donut chart: value counts of {cat}")
        if len(numeric_cols) >= 2:
            suggestions.append(f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}")
            suggestions.append("Pair plot: numeric relationships")
        for num in numeric_cols:
            suggestions.append(f"Histogram: distribution of {num}")
            suggestions.append(f"Line chart: time series of {num}")
            suggestions.append(f"Area chart: cumulative of {num}")
            suggestions.append(f"Density plot: distribution of {num}")
    elif numeric_cols:
        for num in numeric_cols:
            suggestions.append(f"Histogram: distribution of {num}")
            suggestions.append(f"Line chart: time series of {num}")
            suggestions.append(f"Area chart: cumulative of {num}")
            suggestions.append(f"Density plot: distribution of {num}")
        if len(numeric_cols) >= 2:
            suggestions.append(f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}")
            suggestions.append("Pair plot: numeric relationships")
    elif categorical_cols:
        for cat in categorical_cols:
            suggestions.append(f"Pie chart: value counts of {cat}")
            suggestions.append(f"Bar chart: value counts of {cat}")
            suggestions.append(f"Donut chart: value counts of {cat}")

    # Model-based visualizations (if both types exist)
    if numeric_cols and categorical_cols:
        suggestions.append("Heatmap: correlation matrix")
        suggestions.append("Decision tree: feature importance")
        suggestions.append("Confusion matrix: classification results")

    chart_type = st.sidebar.selectbox("Choose a chart to generate", suggestions)


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
    elif "Violin plot" in chart_type:
        num, cat = chart_type.split(":")[1].strip().split(" grouped by ")
        fig = px.violin(df, x=cat, y=num, box=True, title=f"{num} grouped by {cat} (Violin Plot)")
        st.plotly_chart(fig, use_container_width=True)
    elif "Line chart" in chart_type:
        num = chart_type.split(":")[1].strip().split(" of ")[1]
        fig = px.line(df, x=df.index, y=num, title=f"Line chart of {num}")
        st.plotly_chart(fig, use_container_width=True)
    elif "Heatmap" in chart_type:
        import seaborn as sns
        import matplotlib.pyplot as plt
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    elif "Donut chart" in chart_type:
        cat = chart_type.split(":")[1].strip().split(" of ")[1]
        fig = px.pie(df, names=cat, hole=0.4, title=f"Donut chart of {cat}")
        st.plotly_chart(fig, use_container_width=True)
    elif "Area chart" in chart_type:
        num = chart_type.split(":")[1].strip().split(" of ")[1]
        fig = px.area(df, x=df.index, y=num, title=f"Area chart of {num}")
        st.plotly_chart(fig, use_container_width=True)
    elif "Pair plot" in chart_type:
        import seaborn as sns
        fig = sns.pairplot(df[numeric_cols])
        st.pyplot(fig)
    elif "Density plot" in chart_type:
        import seaborn as sns
        import matplotlib.pyplot as plt
        num = chart_type.split(":")[1].strip().split(" of ")[1]
        fig, ax = plt.subplots()
        sns.kdeplot(df[num], ax=ax)
        ax.set_title(f"Density plot of {num}")
        st.pyplot(fig)
    elif "Decision tree" in chart_type:
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt

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
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt

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

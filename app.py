import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Waste Management",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. LOAD DATASET AND TRAINED MODELS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "waste_management.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_classifier.pkl")

LOCATION_ENCODER_PATH = os.path.join(BASE_DIR, "models", "location_encoder.pkl")

DAY_ENCODER_PATH = os.path.join(BASE_DIR, "models", "day_encoder.pkl")

KMEANS_PATH = os.path.join(BASE_DIR, "models", "kmeans_model.pkl")

SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    le_location = joblib.load(LOCATION_ENCODER_PATH)
    le_day = joblib.load(DAY_ENCODER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, le_location, le_day, kmeans, scaler

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


try:
    model, le_location, le_day, kmeans, scaler = load_models()
    df = load_dataset()

except FileNotFoundError:
    st.error(
        "Required dataset or model files are missing. "
        "Please execute the Stage 1 Jupyter Notebook first."
    )

    st.stop()

# ============================================================
# 3. APPLICATION HEADER
# ============================================================

st.title("♻️ Smart Waste Management System")
st.markdown(
    """
    An intelligent waste monitoring and collection
    recommendation system using Data Mining and Machine Learning.

    **Algorithms Used:** Random Forest Classification and K-Means Clustering
    """
)
st.markdown("---")

# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("♻️ Smart Waste")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select a Section",
    [
        "Dashboard",
        "Waste Prediction",
        "Location Analysis",
        "Collection Priority",
        "Cluster Analysis"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Project Information**

    Smart Waste Management System

    Domain: Data Mining

    Classification: Random Forest

    Clustering: K-Means
    """
)

# ============================================================
# 5. DASHBOARD
# ============================================================

if page == "Dashboard":
    st.header("📊 Waste Management Dashboard")

    # Calculate statistics
    total_bins = len(df)

    high_bins = len(df[df["Waste_Category"] == "High"])

    medium_bins = len(df[df["Waste_Category"] == "Medium"])

    low_bins = len(df[df["Waste_Category"] == "Low"])

    average_waste = df["Waste_Level"].mean()

    # Display metrics

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bins", total_bins)
    col2.metric("High Priority", high_bins)
    col3.metric("Medium Priority", medium_bins)
    col4.metric("Average Waste", f"{average_waste:.2f}%")

    st.markdown("---")

    # Waste category distribution

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Waste Category Distribution")
        category_counts = df["Waste_Category"].value_counts()
        st.bar_chart(category_counts)

    with col2:
        st.subheader("Location-wise Average Waste")
        location_waste = df.groupby("Location")["Waste_Level"].mean()
        st.bar_chart(location_waste)

    st.markdown("---")

    # Recent records

    st.subheader("Garbage Bin Records")

    st.dataframe(
        df[
            [
                "Bin_ID",
                "Location",
                "Day",
                "Waste_Level",
                "Waste_Category",
                "Collection_Priority"
            ]
        ].head(15),
        use_container_width=True
    )

# ============================================================
# 6. WASTE PREDICTION
# ============================================================

elif page == "Waste Prediction":
    st.header("🔮 Waste Level Prediction")

    st.write(
        "Enter garbage bin information to predict its "
        "waste category using the trained Random Forest model."
    )

    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox("Select Location", list(le_location.classes_))
        day = st.selectbox("Select Day", list(le_day.classes_))

    with col2:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=20,
            max_value=40,
            value=30
        )

        population = st.number_input(
            "Population",
            min_value=1,
            max_value=100000,
            value=1200
        )

    st.markdown("---")

    if st.button("Predict Waste Level", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            "Location": [le_location.transform([location])[0]],
            "Day": [le_day.transform([day])[0]],
            "Temperature": [temperature],
            "Population": [population]
        })

        # Predict category
        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        confidence = max(probabilities) * 100

        # Display results

        st.subheader("Prediction Result")

        if prediction == "High":

            st.error("🚨 HIGH WASTE LEVEL")

            st.warning(
                "URGENT COLLECTION REQUIRED"
            )

        elif prediction == "Medium":

            st.warning("⚠️ MEDIUM WASTE LEVEL")

            st.info(
                "SCHEDULE GARBAGE COLLECTION"
            )

        else:

            st.success("✅ LOW WASTE LEVEL")

            st.info(
                "NO IMMEDIATE COLLECTION REQUIRED"
            )

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Category",
            prediction
        )

        col2.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.markdown("---")

        # Probability visualization

        st.subheader("Prediction Probability")

        probability_df = pd.DataFrame({

            "Category": model.classes_,

            "Probability": probabilities

        })

        st.bar_chart(
            probability_df.set_index("Category")
        )

        st.caption(
            "Confidence represents the model's predicted class probability, "
            "not guaranteed real-world accuracy."
        )


# ============================================================
# 7. LOCATION ANALYSIS
# ============================================================

elif page == "Location Analysis":

    st.header("📍 Location-wise Waste Analysis")

    st.write(
        "Analyse waste generation patterns across different locations."
    )

    # Generate location statistics

    location_summary = df.groupby("Location").agg(

        Average_Waste=("Waste_Level", "mean"),

        Maximum_Waste=("Waste_Level", "max"),

        Total_Bins=("Bin_ID", "count")

    ).reset_index()

    location_summary["Average_Waste"] = (
        location_summary["Average_Waste"].round(2)
    )

    location_summary["Maximum_Waste"] = (
        location_summary["Maximum_Waste"].round(2)
    )

    st.subheader("Location Statistics")

    st.dataframe(
        location_summary,
        use_container_width=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Average Waste Generation")

        st.bar_chart(
            location_summary.set_index("Location")[
                "Average_Waste"
            ]
        )

    with col2:

        st.subheader("Maximum Waste Level")

        st.bar_chart(
            location_summary.set_index("Location")[
                "Maximum_Waste"
            ]
        )

    st.markdown("---")

    # Highest waste generating location

    highest_location = location_summary.loc[
        location_summary["Average_Waste"].idxmax()
    ]

    st.info(
        f"Highest average waste generation is observed in "
        f"{highest_location['Location']} "
        f"({highest_location['Average_Waste']}%)."
    )


# ============================================================
# 8. COLLECTION PRIORITY
# ============================================================

elif page == "Collection Priority":

    st.header("🚨 Garbage Collection Priority")

    st.write(
        "Identify bins that require immediate garbage collection."
    )

    # Calculate priority statistics

    high_priority = df[
        df["Waste_Category"] == "High"
    ]

    medium_priority = df[
        df["Waste_Category"] == "Medium"
    ]

    low_priority = df[
        df["Waste_Category"] == "Low"
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🔴 Urgent Collection",
        len(high_priority)
    )

    col2.metric(
        "🟡 Scheduled Collection",
        len(medium_priority)
    )

    col3.metric(
        "🟢 Low Priority",
        len(low_priority)
    )

    st.markdown("---")

    # Priority filter

    priority_filter = st.selectbox(
        "Filter Collection Priority",
        [
            "All",
            "URGENT COLLECTION",
            "SCHEDULE COLLECTION",
            "NO IMMEDIATE COLLECTION"
        ]
    )

    if priority_filter == "All":

        result = df.copy()

    else:

        result = df[
            df["Collection_Priority"] == priority_filter
        ].copy()

    # Sort by waste level

    result = result.sort_values(
        "Waste_Level",
        ascending=False
    )

    st.subheader("Collection Records")

    st.dataframe(
        result[
            [
                "Bin_ID",
                "Location",
                "Day",
                "Waste_Level",
                "Waste_Category",
                "Collection_Priority"
            ]
        ],
        use_container_width=True
    )

    st.markdown("---")

    # Download collection report

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Collection Report",
        data=csv,
        file_name="waste_collection_report.csv",
        mime="text/csv"
    )


# ============================================================
# 9. CLUSTER ANALYSIS
# ============================================================

elif page == "Cluster Analysis":

    st.header("📊 Waste Pattern Clustering")

    st.write(
        "Visualize garbage bins grouped using the K-Means "
        "clustering algorithm."
    )

    # Cluster statistics

    cluster_summary = df.groupby("Cluster").agg(

        Average_Waste=("Waste_Level", "mean"),

        Average_Population=("Population", "mean"),

        Total_Bins=("Bin_ID", "count")

    ).reset_index()

    cluster_summary["Average_Waste"] = (
        cluster_summary["Average_Waste"].round(2)
    )

    cluster_summary["Average_Population"] = (
        cluster_summary["Average_Population"].round(2)
    )

    st.subheader("Cluster Statistics")

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

    st.markdown("---")

    # Scatter plot

    st.subheader("K-Means Cluster Visualization")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(

        data=df,

        x="Population",

        y="Waste_Level",

        hue="Cluster",

        palette="viridis",

        s=70,

        ax=ax

    )

    ax.set_title("Garbage Bin Clusters")

    ax.set_xlabel("Population")

    ax.set_ylabel("Waste Level (%)")

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("---")

    # Average waste per cluster

    st.subheader("Average Waste per Cluster")

    st.bar_chart(
        cluster_summary.set_index("Cluster")[
            "Average_Waste"
        ]
    )

    # Cluster interpretation

    highest_cluster = cluster_summary.loc[
        cluster_summary["Average_Waste"].idxmax()
    ]

    st.info(
        f"Cluster {highest_cluster['Cluster']} has the highest "
        f"average waste level of "
        f"{highest_cluster['Average_Waste']}%."
    )


# ============================================================
# 10. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:gray;">

    Smart Waste Management System

    <br>

    Developed using Python, Streamlit and Machine Learning

    <br>

    DMCT Assignment Project

    </div>
    """,
    unsafe_allow_html=True
)
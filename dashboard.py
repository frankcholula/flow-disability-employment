import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from collections import Counter
from typing import List
import math

# modeling specific packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from interview_wordcloud import (
    generate_wordcloud,
    工作責任感,
    工作意願,
    自我身心照顧,
    日常興趣社交,
    溝通表達,
    學習動機,
    家人支持,
    解決問題意願,
)

dataset_url = "https://raw.githubusercontent.com/frankcholula/flow-disability-employment/main/data/scores.csv"

# supress pyplot warning
st.set_option("deprecation.showPyplotGlobalUse", False)

# 內在指標 + 外在指標 + PPSS指標
inside_features = ["工作意願和動機", "學習動力", "基本溝通表達", "工作責任感", "解決問題意願"]
outside_features = ["社群和社交活動", "自我身心照顧", "家人支持程度", "私人企業工作經驗", "量化求職考量", "先天後天"]
ppss_features = [
    "PPSS積極性",
    "PPSS責任性",
    "PPSS成熟性",
    "PPSS務實性",
    "PPSS社交性",
    "PPSS合群性",
    "PPSS創意性",
    "PPSS表達性",
    "PPSS學習性",
    "PPSS細心",
    "PPSS耐心",
    "PPSS親和性",
    "PPSS領導性",
    "PPSS邏輯性",
]

meta_features = ["受訪者", "內外部", "關鍵TA"]
target = "關鍵TA"

# page setup
st.set_page_config(
    page_title="若水身障就業資料分析",
    page_icon="🚰",
    layout="wide",
)


# data preparation
@st.cache_data
def read_data(file_path) -> pd.DataFrame:
    """
    Read data from csv file and return 3 dataframes
    """
    scores_df = pd.read_csv(file_path)
    return scores_df


class Visualization:
    """_summary_"""

    def __init__(self, vis, *args):
        self.vis = vis
        dispatcher = {
            "wordcloud": self.generate_interview_wordcloud,
            "personality": self.generate_all_radar_charts,
            "distribution": self.generate_distribution,
            "median": self.generate_median_radar_chart,
            "correlation": self.generate_correlation_matrix,
            "disability": self.generate_disability_histogram,
            "education": self.generate_education_histogram,
            "job_consideration": self.generate_job_consideration_histogram,
            "job_channel": self.generate_job_channel_histogram,
            "models": self.generate_model_performance,
        }

        init_func = dispatcher.get(vis)
        if not init_func:
            raise ValueError(f"Uknown visualization type: {vis}")
        else:
            init_func(*args)

    def generate_interview_wordcloud(self, text):
        wc = generate_wordcloud(text)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        st.pyplot()

    def generate_median_radar_chart(self, df, features):
        def get_median_df(df):
            try:
                if df["關鍵TA"].unique()[0] == "T":
                    key = "關鍵TA中間值"
                else:
                    key = "非關鍵TA中間值"
                key = "外部" + key if df["內外部"].unique()[0] == "外部" else "內部" + key
                avg_inside_features = df[inside_features].median()
                avg_outside_features = df[outside_features].median()
                avg_ta = pd.Series(
                    [key]
                    + [None] * (len(meta_features) - 1)
                    + avg_inside_features.tolist()
                    + avg_outside_features.tolist()
                    + [key],
                    index=["受訪者"]
                    + meta_features[1:]
                    + inside_features
                    + outside_features
                    + [target],
                )
            except Exception as e:
                avg_ta = pd.Series([0.0] * len(df.columns), index=df.columns)
            return avg_ta.to_frame().transpose()

        median_col1, median_col2 = st.columns(2)
        with median_col1:
            inside_outside_filter = st.selectbox("選擇內外部", pd.unique(df["內外部"]))
        df = df[df["內外部"] == inside_outside_filter].reset_index(drop=True)
        if inside_outside_filter == "外部":
            with median_col2:
                ta_filter = st.selectbox("選擇關鍵TA", pd.unique(scores_df["關鍵TA"]))
            df = df[df["關鍵TA"] == ta_filter].reset_index(drop=True)
        st.markdown("### 六大指標中間值")
        metrics = get_median_df(df)
        is1, is2, is3, is4, is5, is6 = st.columns(6)
        is1.metric("工作意願和動機", int(metrics["工作意願和動機"].values[0]))
        is2.metric("學習動力", int(metrics["學習動力"].values[0]))
        is3.metric("基本溝通表達", int(metrics["基本溝通表達"].values[0]))
        is4.metric("工作責任感", int(metrics["工作責任感"].values[0]))
        is5.metric("解決問題意願", int(metrics["解決問題意願"].values[0]))
        is6.metric("自我身心照顧", int(metrics["自我身心照顧"].values[0]))
        self.generate_all_radar_charts(metrics, radar_features, 1, 0, 1)

    def generate_all_radar_charts(
        self,
        df,
        features,
        charts_per_row,
        vertical_spacing=0.07,
        horizontal_spacing=0.6,
    ):
        MAX_VALUES = {
            "工作意願和動機": 5,
            "學習動力": 3,
            "基本溝通表達": 3,
            "工作責任感": 3,
            "解決問題意願": 3,
            "社群和社交活動": 3,
            "家人支持程度": 5,
            "私人企業工作經驗": 1,
            "量化求職考量": 3,
            "先天後天": 1,
            "自我身心照顧": 6,
        }
        n_rows = math.ceil(len(df) / charts_per_row)
        features_closed = features + [features[0]]

        # Create a subplot layout
        if charts_per_row > 1:
            fig = self._generate_multiple_charts(
                df,
                features,
                features_closed,
                charts_per_row,
                n_rows,
                MAX_VALUES,
                vertical_spacing,
                horizontal_spacing,
            )
        else:
            fig = self._generate_single_chart(df, features, features_closed, MAX_VALUES)

        st.plotly_chart(fig, use_container_width=True)

    def _generate_multiple_charts(
        self,
        df,
        features,
        features_closed,
        charts_per_row,
        n_rows,
        MAX_VALUES,
        vertical_spacing,
        horizontal_spacing,
    ):
        fig = make_subplots(
            rows=n_rows,
            cols=charts_per_row,
            specs=[[{"type": "polar"}] * charts_per_row] * n_rows,
            subplot_titles=(df.受訪者),
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
        )

        layout_update = {}

        for index, row in df.iterrows():
            row_normalized = {col: row[col] / MAX_VALUES[col] for col in features}
            row_normalized_list = list(row_normalized.values()) + [
                list(row_normalized.values())[0]
            ]

            subplot_row = index // charts_per_row + 1
            subplot_col = index % charts_per_row + 1
            polar_name = f"polar{index+1}"
            layout_update[polar_name] = dict(radialaxis=dict(showticklabels=False))

            fig.add_trace(
                go.Scatterpolar(
                    name=row.受訪者,
                    r=row_normalized_list,
                    theta=features_closed,
                    fill="toself",
                    showlegend=False,
                ),
                row=subplot_row,
                col=subplot_col,
            )

        # Update layout to remove radial tick labels and adjust layout
        fig.update_layout(
            **layout_update, margin=dict(t=50, b=50, l=100, r=100), height=1000
        )
        fig.update_polars(radialaxis=dict(range=[0, 1]))

        return fig

    def _generate_single_chart(self, df, features, features_closed, MAX_VALUES):
        row_normalized = {col: df[col][0] / MAX_VALUES[col] for col in features}
        row_normalized_list = list(row_normalized.values()) + [
            list(row_normalized.values())[0]
        ]

        fig = px.line_polar(
            df,
            r=row_normalized_list,
            theta=features_closed,
        )
        fig.update_traces(fill="toself", showlegend=False)

        return fig

    def generate_distribution(self, df: pd.DataFrame, features: List[str]):
        dist_df = df.copy()
        dist_df = dist_df[dist_df["內外部"] == "外部"].reset_index(drop=True)
        dist_df["六大總分"] = dist_df[features].sum(axis=1)
        dist_col1, dist_col2 = st.columns(2)

        with dist_col1:
            feature_filter = st.selectbox(
                "選擇特質分數",
                features + ["六大總分"] + ["六大總分(核密度)"],
            )
        if feature_filter == "六大總分(核密度)":
            ta_list = dist_df[dist_df["關鍵TA"] == "T"]["六大總分"].dropna().values.tolist()
            nonta_list = (
                dist_df[dist_df["關鍵TA"] == "F"]["六大總分"].dropna().values.tolist()
            )
            with dist_col2:
                selected_group = st.selectbox("選擇族群", ("關鍵vs.非關鍵TA", "全部TA"))
            print(dist_col2)
            if selected_group == "關鍵vs.非關鍵TA":
                hist_data = [ta_list, nonta_list]
                fig = ff.create_distplot(
                    hist_data,
                    ["關鍵TA", "非關鍵TA"],
                    bin_size=2,
                    colors=["red", "blue"],
                )
            elif selected_group == "全部TA":
                hist_data = ta_list + nonta_list
                fig = ff.create_distplot(
                    [hist_data],
                    ["全部TA"],
                    bin_size=2,
                    colors=["green"],
                )
        else:
            num_bins = int(
                dist_df[feature_filter].max() - dist_df[feature_filter].min() + 1
            )
            fig = px.histogram(
                dist_df,
                x=feature_filter,
                marginal="box",
                title=f"外部關鍵TA vs 外部非關鍵的{feature_filter}常態分佈",
                nbins=num_bins,
                color="關鍵TA",
                color_discrete_sequence=["red", "blue"],
            )

            fig.update_layout(
                xaxis=dict(
                    tickmode="linear", tick0=dist_df[feature_filter].min(), dtick=1
                ),
                yaxis=dict(title="人數"),
            )

            fig.update_traces(opacity=0.75)
        st.plotly_chart(fig, use_container_width=True)

    def generate_correlation_matrix(
        self,
        df: pd.DataFrame,
        features: List[str],
        title="外部訪談者內在 vs. 外在的特質相關度 (等級相關係數)",
        xaxis="外在特質",
        yaxis="內在特質",
    ):
        corr_df = df.copy()
        corr_df = corr_df[corr_df["內外部"] == "外部"].reset_index(drop=True)
        corr_mx = corr_df[corr_features].corr(method="kendall")
        corr_mx = corr_mx[0:5][outside_features]
        fig = ff.create_annotated_heatmap(
            z=corr_mx.values,
            x=list(corr_mx.columns),
            y=list(corr_mx.index),
            annotation_text=corr_mx.round(2).values,
            colorscale="Viridis",
            showscale=True,
            hoverinfo="z",
        )
        fig.update_layout(
            title=title,
            xaxis=dict(title=xaxis, side="bottom"),
            yaxis=dict(title=yaxis),
        )
        st.plotly_chart(fig, use_container_width=True)

    def generate_disability_histogram(self, df):
        my_df = df.copy()
        key_ta_df = my_df[my_df["關鍵TA"] == "T"]
        nonkey_ta_df = my_df[my_df["關鍵TA"] == "F"]

        key_ta_experience_distribution = key_ta_df["障別"].value_counts()
        non_key_ta_experience_distribution = nonkey_ta_df["障別"].value_counts()

        combined_data = pd.concat(
            [
                key_ta_experience_distribution.rename("是"),
                non_key_ta_experience_distribution.rename("否"),
            ],
            axis=1,
        ).fillna(0)

        combined_data["Total"] = combined_data["是"] + combined_data["否"]
        combined_data["Total"] = combined_data["Total"].astype(int)
        combined_data = combined_data.reset_index().rename(
            columns={"index": "Disability Type"}
        )
        combined_data.reset_index(inplace=True)
        combined_data.rename(columns={"index": "Experience Type"}, inplace=True)
        fig = px.bar(
            combined_data,
            x="障別",
            y=["是", "否"],
            barmode="stack",
            title="障別分佈圖",
            labels={"value": "人數", "variable": "關鍵TA?"},
            color_discrete_sequence=["red", "blue"],
        )
        for i, total in enumerate(combined_data["Total"]):
            fig.add_annotation(
                x=combined_data["Experience Type"][i],
                y=total,
                text=str(total),
                showarrow=False,
                yshift=10,  # Adjust this value to position the annotation above the bar
            )
        st.plotly_chart(fig, use_container_width=True)

    def generate_education_histogram(self, df):
        my_df = df.copy()
        key_ta_df = my_df[my_df["關鍵TA"] == "T"]
        nonkey_ta_df = my_df[my_df["關鍵TA"] == "F"]

        key_ta_experience_distribution = (
            key_ta_df["問卷學歷"].str.split("、").explode().value_counts()
        )
        non_key_ta_experience_distribution = (
            nonkey_ta_df["問卷學歷"].str.split("、").explode().value_counts()
        )

        combined_data = pd.concat(
            [
                key_ta_experience_distribution.rename("是"),
                non_key_ta_experience_distribution.rename("否"),
            ],
            axis=1,
        ).fillna(0)

        combined_data["Total"] = combined_data["是"] + combined_data["否"]
        combined_data["Total"] = combined_data["Total"].astype(int)
        combined_data = combined_data.reset_index().rename(
            columns={"index": "Experience Type"}
        )
        combined_data.reset_index(inplace=True)
        combined_data.rename(columns={"index": "Experience Type"}, inplace=True)
        fig = px.bar(
            combined_data,
            x="問卷學歷",
            y=["是", "否"],
            barmode="stack",
            title="學歷分佈圖",
            labels={"value": "人數", "variable": "關鍵TA?"},
            color_discrete_sequence=["red", "blue"],
        )
        for i, total in enumerate(combined_data["Total"]):
            fig.add_annotation(
                x=combined_data["Experience Type"][i],
                y=total,
                text=str(total),
                showarrow=False,
                yshift=10,  # Adjust this value to position the annotation above the bar
            )
        st.plotly_chart(fig, use_container_width=True)

    def generate_job_consideration_histogram(self, df):
        my_df = df.copy()
        ta_df = my_df[my_df["關鍵TA"] == "T"]
        nonta_df = my_df[my_df["關鍵TA"] == "F"]
        jc_true = ta_df["求職考量"].str.split("、").explode().str.strip()
        jc_false = nonta_df["求職考量"].str.split("、").explode().str.strip()
        jc_counts_true = pd.Series(Counter(jc_true)).reset_index()
        jc_counts_false = pd.Series(Counter(jc_false)).reset_index()
        jc_counts_true.columns = ["求職考量", "人數"]
        jc_counts_false.columns = ["求職考量", "人數"]
        jc_counts_true["關鍵TA"] = "是"
        jc_counts_false["關鍵TA"] = "否"
        combined_counts_with_key_TA = pd.concat([jc_counts_true, jc_counts_false])
        total_counts = (
            combined_counts_with_key_TA.groupby("求職考量")["人數"].sum().reset_index()
        )
        fig = px.bar(
            combined_counts_with_key_TA,
            x="求職考量",
            y="人數",
            color="關鍵TA",
            title="求職考量分佈圖",
            labels={"人數": "人數", "求職考量": "求職考量", "關鍵TA": "關鍵TA"},
            color_discrete_sequence=["red", "blue"],
        )

        # Add text annotations for total counts
        for i, row in total_counts.iterrows():
            fig.add_annotation(
                x=row["求職考量"],
                y=row["人數"],
                text=str(row["人數"]),
                showarrow=True,
                font=dict(size=12, color="black"),
            )
        st.plotly_chart(fig, use_container_width=True)

    def generate_job_channel_histogram(self, df, title="問卷求職管道"):
        my_df = df.copy()
        ta_df = my_df[my_df["關鍵TA"] == "T"]
        nonta_df = my_df[my_df["關鍵TA"] == "F"]
        # Count job considerations for both subsets
        jc_true = ta_df["問卷求職管道"].str.split("、").explode().str.strip()
        jc_false = nonta_df["問卷求職管道"].str.split("、").explode().str.strip()
        jc_counts_true = pd.Series(Counter(jc_true)).reset_index()
        jc_counts_false = pd.Series(Counter(jc_false)).reset_index()
        jc_counts_true.columns = ["問卷求職管道", "人數"]
        jc_counts_false.columns = ["問卷求職管道", "人數"]
        jc_counts_true["關鍵TA"] = "是"
        jc_counts_false["關鍵TA"] = "否"
        combined_counts_with_key_TA = pd.concat([jc_counts_true, jc_counts_false])
        total_counts = (
            combined_counts_with_key_TA.groupby("問卷求職管道")["人數"].sum().reset_index()
        )

        fig = px.bar(
            combined_counts_with_key_TA,
            x="問卷求職管道",
            y="人數",
            color="關鍵TA",
            title=title,
            labels={"人數": "人數", "問卷求職管道": "問卷求職管道", "關鍵TA": "關鍵TA"},
            color_discrete_sequence=["red", "blue"],
        )

        # Add text annotations for total counts
        for i, row in total_counts.iterrows():
            fig.add_annotation(
                x=row["問卷求職管道"],
                y=row["人數"],
                text=str(row["人數"]),
                showarrow=True,
                font=dict(size=12, color="black"),
            )
        st.plotly_chart(fig, use_container_width=True)

    def generate_model_performance(self):
        inside_features = ["工作意願和動機", "學習動力", "基本溝通表達", "工作責任感", "解決問題意願"]
        outside_features = ["社群和社交活動", "自我身心照顧", "家人支持程度", "私人企業工作經驗", "量化求職考量"]

        @ignore_warnings(category=ConvergenceWarning)
        @st.cache_data
        def logistic_regression_bootstrap(
            df,
            features,
            target="關鍵TA",
            test_size=0.2,
            random_state=42,
            n_bootstraps=200,
            title="Bootstrapped",
        ):
            X_train, X_test, y_train, y_test = train_test_split(
                df[features], df[target], test_size=test_size, random_state=random_state
            )

            classifier = LogisticRegression(random_state=42)
            classifier.fit(X_train, y_train)

            # Predict on the testing data
            y_test_pred = classifier.predict(X_test)

            bootstrap_accuracies = []
            # bootstrapping
            for _ in range(n_bootstraps):
                # TODO: wrap try catch here in case we get all TA's
                X_train_boot, y_train_boot = resample(X_train, y_train)
                classifier.fit(X_train_boot, y_train_boot)
                # Predict on the original testing data
                y_test_pred_boot = classifier.predict(X_test)
                bootstrap_accuracies.append(accuracy_score(y_test, y_test_pred_boot))
            bootstrap_accuracies = np.array(bootstrap_accuracies)
            mean_accuracy = bootstrap_accuracies.mean()
            std_dev_accuracy = bootstrap_accuracies.std()

            # Plot the distribution of accuracies
            len_training = len(X_train)
            len_testing = len(X_test)
            st.text(f"跑{n_bootstraps}次，隨機選用{len_training}位訓練，指定{len_testing}位盲測")
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(bootstrap_accuracies, kde=True)
            plt.title(title)
            plt.xlabel("Accuracy")
            plt.ylabel("Frequency")
            plt.axvline(
                x=mean_accuracy,
                color="red",
                linestyle="--",
                label=f"Mean Accuracy: {mean_accuracy:.2f}",
            )
            plt.legend()
            st.pyplot(fig)
            return classifier, bootstrap_accuracies

        @st.cache_data
        def svm_bootstrap(
            df,
            features,
            target="關鍵TA",
            test_size=0.2,
            random_state=42,
            n_bootstraps=200,
            title="Bootstrapped Accuracies Distribution",
        ):
            X_train, X_test, y_train, y_test = train_test_split(
                df[features], df[target], test_size=test_size, random_state=random_state
            )

            classifier = SVC(kernel="rbf", random_state=random_state)
            classifier.fit(X_train, y_train)

            # Predict on the testing data
            y_test_pred = classifier.predict(X_test)

            bootstrap_accuracies = []
            # bootstrapping
            for _ in range(n_bootstraps):
                # TODO: wrap try catch here in case we get all TA's
                X_train_boot, y_train_boot = resample(X_train, y_train)
                classifier.fit(X_train_boot, y_train_boot)
                # Predict on the original testing data
                y_test_pred_boot = classifier.predict(X_test)
                bootstrap_accuracies.append(accuracy_score(y_test, y_test_pred_boot))
            bootstrap_accuracies = np.array(bootstrap_accuracies)
            mean_accuracy = bootstrap_accuracies.mean()
            std_dev_accuracy = bootstrap_accuracies.std()

            # Plot the distribution of accuracies
            len_training = len(X_train)
            len_testing = len(X_test)
            st.text(f"跑{n_bootstraps}次，隨機選用{len_training}位訓練，指定{len_testing}位盲測")
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(bootstrap_accuracies, kde=True)
            plt.title(title)
            plt.xlabel("Accuracy")
            plt.ylabel("Frequency")
            plt.axvline(
                x=mean_accuracy,
                color="red",
                linestyle="--",
                label=f"Mean Accuracy: {mean_accuracy:.2f}",
            )
            plt.legend()
            st.pyplot(fig)
            return classifier, bootstrap_accuracies

        model_col1, model_col2 = st.columns(2)
        with model_col1:
            model = st.selectbox("選擇模型", ("Logistic Regression", "SVM"))
        with model_col2:
            features = st.selectbox("選擇特質", ("內在特質", "外在特質", "內+外在特質", "PPSS"))

        model_df = scores_df.copy()
        if model == "SVM":
            with st.spinner("訓練中..."):
                if features == "內在特質":
                    svm, bootstrap_accuracies = svm_bootstrap(
                        model_df,
                        inside_features,
                        n_bootstraps=500,
                        title="SVM Model Using Inside Features",
                    )
                if features == "外在特質":
                    svm, bootstrap_accuracies = svm_bootstrap(
                        model_df,
                        outside_features,
                        n_bootstraps=500,
                        title="SVM Model Using Outside Features",
                    )
                if features == "內+外在特質":
                    svm, bootstrap_accuracies = svm_bootstrap(
                        model_df,
                        inside_features + outside_features,
                        n_bootstraps=500,
                        title="SVM Model Using Inside and Outside Features",
                    )
                if features == "PPSS":
                    svm, bootstrap_accuracies = svm_bootstrap(
                        model_df,
                        ppss_features,
                        n_bootstraps=500,
                        title="SVM Model Using PPSS",
                    )
        if model == "Logistic Regression":
            with st.spinner("訓練中..."):
                if features == "內在特質":
                    lrm, bootstrap_accuracies = logistic_regression_bootstrap(
                        model_df,
                        inside_features,
                        n_bootstraps=500,
                        title="LR Model Using Inside Features",
                    )
                if features == "外在特質":
                    lrm, bootstrap_accuracies = logistic_regression_bootstrap(
                        model_df,
                        outside_features,
                        n_bootstraps=500,
                        title="LR Model Using Outside Features",
                    )
                if features == "內+外在特質":
                    lrm, bootstrap_accuracies = logistic_regression_bootstrap(
                        scores_df,
                        inside_features + outside_features,
                        n_bootstraps=500,
                        title="LR Model Using Both Inside and Outside Features",
                    )
                if features == "PPSS":
                    lrm, bootstrap_accuracies = logistic_regression_bootstrap(
                        scores_df,
                        ppss_features,
                        n_bootstraps=500,
                        title="LR Model Using PPSS",
                    )


# dashboard title
st.title(":potable_water: :blue[若水]身障就業資料分析")
st.markdown("『_創造多元共﻿融環境是為了每一個人_』，我們希望透過商業力量，協助企業和身障人才有效銜接，改善身障就業問題！")


scores_df = read_data(dataset_url)

placeholder = st.empty()
with placeholder.container():
    fig_col1, fig_col2, fig_col3 = st.columns(3, gap="large")
    with fig_col1:
        st.header(":dart: 定位")
        option = st.selectbox(
            "視覺化六大特質",
            ("外部關鍵TA的特質常態分佈", "內外部關鍵TA的特質雷達圖", "內外部關鍵TA的特質中間值", "特質訪談文字雲"),
            index=None,
            placeholder="選擇視覺化圖表",
        )
        radar_features = ["工作意願和動機", "學習動力", "基本溝通表達", "工作責任感", "解決問題意願", "自我身心照顧"]
        if option == "特質訪談文字雲":
            wc_col1, wc_col2 = st.columns(2)
            with wc_col1:
                feature_selection = st.selectbox(
                    "選擇特質",
                    ("工作責任感", "工作意願", "自我身心照顧", "學習動機", "溝通表達", "工作意願", "解決問題意願"),
                )
            feature_data = {
                "工作責任感": 工作責任感,
                "工作意願": 工作意願,
                "自我身心照顧": 自我身心照顧,
                "學習動機": 學習動機,
                "溝通表達": 溝通表達,
                "解決問題意願": 解決問題意願,
            }

            if feature_selection in feature_data:
                with st.spinner("製圖中..."):
                    wordcloud = Visualization(
                        "wordcloud", feature_data[feature_selection]
                    )
        if option == "外部關鍵TA的特質常態分佈":
            distribution = Visualization("distribution", scores_df, radar_features)
            st.markdown("### 結論")
            st.markdown("1. 依照六大特質加總結果，定義出判別關鍵 TA 的標準線，為總分18分以上(滿分23分)。")
            st.markdown("2. 建議持續優化評測指標並建立評估表，在徵才階段有效判斷六大特質。")

        if option == "內外部關鍵TA的特質雷達圖":
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                inside_outside_filter = st.selectbox(
                    "選擇內外部", pd.unique(scores_df["內外部"])
                )
            if inside_outside_filter == "外部":
                with filter_col2:
                    ta_filter = st.selectbox("選擇關鍵TA", pd.unique(scores_df["關鍵TA"]))
                radar_df = scores_df.copy()
                radar_df = radar_df[
                    radar_df["內外部"] == inside_outside_filter
                ].reset_index(drop=True)
                radar_df = radar_df[radar_df["關鍵TA"] == ta_filter].reset_index(
                    drop=True
                )

                radar_charts = Visualization(
                    "personality",
                    radar_df,
                    radar_features,
                    2,
                )
            else:
                radar_df = scores_df.copy()
                radar_df = radar_df[
                    radar_df["內外部"] == inside_outside_filter
                ].reset_index(drop=True)
                radar_charts = Visualization(
                    "personality",
                    radar_df,
                    radar_features,
                    2,
                )
        if option == "內外部關鍵TA的特質中間值":
            median_radar_chart = Visualization("median", scores_df, radar_features)

    with fig_col2:
        st.header(":pinching_hand: 篩選")
        option = st.selectbox(
            "視覺化建模與特質篩選",
            (
                "特質相關矩陣",
                "利用特質建模預測關鍵TA",
            ),
            index=None,
            placeholder="選擇視覺化圖表",
        )

        if option == "特質相關矩陣":
            corr_features = inside_features + outside_features
            correlation_matrix = Visualization("correlation", scores_df, corr_features)
            st.markdown("_0.2以下不相關，0.2 − 0.39 是弱相關， 0.4 − 0.59 是中度相關，0.6 − 0.79 是強相關。_")
            st.markdown("### 結果")
            st.markdown("1. 自我身心照顧與每項內在特質有強度相關性。")
            st.markdown("2. 量化求職考量與每項內在特質有中、強度相關性。")
            st.markdown("3. 家人支持程度與、私人企業工作經驗、社群社交活動與幾項內在特質有弱度相關性。")
            st.markdown("4. 先天後天沒有和內在特質有相關度")
            st.markdown("### 結論")
            st.markdown("1. 自我身心照顧和每項內在特質皆有高度相關，為最重要的外在特質！")
            st.markdown("2. 求職考量*和基本溝通表達有高度相關，和其他四項有中度相關，為次重要的關鍵外在特質。")
            st.markdown("3. 先天後天與關鍵TA很明顯沒有相關性，所以可以不用考慮。")
        if option == "利用特質建模預測關鍵TA":
            model_performance = Visualization("models")
            st.markdown("### 實驗")
            st.text("我們透過內在特質、外在特質、內+外在特質、PPSS特質建立了關鍵 TA 的預測模型。")
            st.text("但因為『先天後天』指標沒有明確的關聯繫，所以我們沒有將其納入模型。")
            st.markdown("### 結果")
            st.text("1. 透過內在特質建模，預測關鍵 TA 的最高平均準確率為 0.97。")
            st.text("2. 透過外在特質建模，預測關鍵 TA 的最高平均準確率為 0.89。")
            st.text("3. 透過內+外在特質，預測關鍵 TA 的最高平均準確率為 0.95。")
            st.text("4. 透過PPSS特質，預測關鍵 TA 的最高平均準確率為 0.61。")
            st.markdown("### 結論")
            st.text("內在特質才是最準確的評測指標，但只看外在特質也有不錯的表現！")

    with fig_col3:
        st.header(":books: 管道")
        option = st.selectbox(
            "視覺化求職管道與考量",
            (
                "訪談者問卷學歷",
                "訪談者障別分析",
                "訪談者求職考量",
                "問卷求職管道",
            ),
            index=None,
            placeholder="選擇視覺化圖表",
        )
        if option == "訪談者問卷學歷":
            education_dist = Visualization("education", scores_df)
        if option == "訪談者障別分析":
            disability_types = Visualization("disability", scores_df)
            st.markdown("### 結果")
            st.markdown(
                "1. 訪談者障別分析發現，脊椎損傷、腦性麻痺、肌肉萎縮為三大主要障別有高度關鍵TA。但這也有可能是因為這三種障別的訪談者較多，所以有待進一步驗證。"
            )
            st.markdown("### 結論")
            st.markdown(
                "1. 外部訪談發現，肌肉萎縮的受訪者皆為關鍵人才 。因肌肉萎縮為罕見疾病, 較無法大規模主動觸及。企業可考慮將肌肉萎縮納入徵才階段的障別考量參考，並持續驗證。"
            )
        if option == "訪談者求職考量":
            job_consideration = Visualization("job_consideration", scores_df)
            st.markdown("### 結論")
            st.markdown("1. 關鍵人才重視的求職考量要素排序：工作性質與內容 > 無障礙環境 > 經濟需求")
            st.markdown("2. 非關鍵人才重視的求職考量要素排序：經濟需求 > 無障礙環境 > 交通距離")
        if option == "問卷求職管道":
            job_channel = Visualization("job_channel", scores_df)
            st.markdown("### 結果")
            st.markdown("1. 無論是求職管道總人數(20人)，以及招募管道有效性(12人，60%)皆以網路人力銀行為最高")
            st.markdown("2. 招募管效性次高為社群貼文（4人，57%）")

            st.markdown("### 結論")
            st.markdown(
                "1. 根據實際訪談與求職管道比對後，發現關鍵人才並非集中存在「與該障別直接相關」的傷友支持社群或非營利組織。相對來說，他們多聚集於自我挑戰導向的活動Line社群，例：輪椅夢公園群組。未來可強化連結同性質社群，例：身心障礙潛水協會"
            )

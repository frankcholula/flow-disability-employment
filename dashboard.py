# import pandas as pd
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

dataset_url = "https://raw.githubusercontent.com/frankcholula/flow-disability-employment/main/data/scores.csv"

# 內在指標 + 外在指標
inside_features = ["工作意願和動機", "學習動力", "基本溝通表達", "工作責任感", "解決問題意願"]
outside_features = ["社群和社交活動", "自我身心照顧", "家人支持程度", "私人企業工作經驗", "量化求職考量", "先天後天"]
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
            "personality": self.generate_all_radar_charts,
            "distribution": self.generate_distribution,
            "median": self.generate_median_radar_chart,
            "correlation": self.generate_correlation_matrix,
            "disability": self.generate_disability_histogram,
            "education": self.generate_education_histogram,
            "job_consideration": self.generate_job_consideration_histogram,
            "job_channel": self.generate_job_channel_histogram,
        }

        init_func = dispatcher.get(vis)
        if not init_func:
            raise ValueError(f"Uknown visualization type: {vis}")
        else:
            init_func(*args)

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
        for feature in features:
            num_bins = int(dist_df[feature].max() - dist_df[feature].min() + 1)

            fig = px.histogram(
                dist_df,
                x=feature,
                marginal="box",
                title=f"外部關鍵TA vs 外部非關鍵的{feature}常態分佈",
                nbins=num_bins,
                color="關鍵TA",
                color_discrete_sequence=["red", "blue"],
            )

            fig.update_layout(
                xaxis=dict(tickmode="linear", tick0=dist_df[feature].min(), dtick=1)
            )

            fig.update_traces(opacity=0.75)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 結論")
        st.text("依照六大特質加總結果，定義出判別關鍵 TA 的標準線，為總分18分以上。（滿分23分）")

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
        st.markdown("_0.2以下不相關，0.2 − 0.39 是弱相關， 0.4 − 0.59 是中度相關，0.6 − 0.79 是強相關。_")
        st.markdown("### 結論")
        st.text("1. 自我身心照顧與每向內在特質有高度相關性，所以很重要的外在指標")
        st.text("2. 量化求職考量與解決問題意願和基本溝通表達是高相關")
        st.text("3. 家人支持程度與美向內在特質是中度相關")

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
        st.markdown("### 結論")
        st.text("1. 無論是求職管道總人數(20人)，以及招募管道有效性(12人，60%)皆以網路人力銀行為最高")
        st.text("2. 招募管效性次高為社群貼文（4人，57%）")
        st.text("3. 根據實際訪談與求職管道比對後，發現關鍵 TA 並非集中存在「與該障別直接相關」的傷友支持社群或非營利組織")
        st.text("4. 相對來說，他們多聚集於興趣、自我挑戰導向的私密社群，例：輪椅夢公園群組")
        st.text("5. 未來可強化連結同性質社群，提升關鍵TA觸及率，例：身心障礙潛水協會")


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
            ("外部關鍵TA的特質常態分佈", "內外部關鍵TA的特質雷達圖", "內外部關鍵TA的特質中間值"),
            index=None,
            placeholder="選擇視覺化圖表",
        )
        radar_features = ["工作意願和動機", "學習動力", "基本溝通表達", "工作責任感", "解決問題意願", "自我身心照顧"]
        if option == "外部關鍵TA的特質常態分佈":
            distribution = Visualization("distribution", scores_df, radar_features)
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

        if option == "利用特質建模預測關鍵TA":
            pass

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
        if option == "訪談者求職考量":
            job_consideration = Visualization("job_consideration", scores_df)
        if option == "問卷求職管道":
            job_channel = Visualization("job_channel", scores_df)

# import pandas as pd
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    def __init__(self, vis, *args, **kwargs):
        self.vis = vis
        dispatcher = {
            "personality": self.generate_radar_charts,
            "distribution": self.generate_distribution,
        }

        init_func = dispatcher.get(vis)
        if not init_func:
            raise ValueError(f"Uknown visualization type: {vis}")
        else:
            if vis == "distribution":
                init_func(*args)
            else:
                init_func(*args)

    def generate_radar_charts(
        self,
        df,
        features,
        charts_per_row,
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
        # Create a subplot layout
        fig = make_subplots(
            rows=n_rows,
            cols=charts_per_row,
            specs=[[{"type": "polar"}] * charts_per_row] * n_rows,
            subplot_titles=(df.受訪者),
        )

        layout_update = {}
        features_closed = features[0:] + [features[0]]

        for index, row in df.iterrows():
            row_normalized = {col: row[col] / MAX_VALUES[col] for col in features[0:]}
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
            **layout_update,
            margin=dict(t=50, b=50, l=100, r=100),
            height=1000,
        )
        fig.update_polars(radialaxis=dict(range=[0, 1]))

        st.plotly_chart(fig, use_container_width=True)

        def get_median_df(df):
            try:
                if df["關鍵TA"].unique()[0] == "關鍵TA":
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

        st.markdown("### 內在指標中間值")
        is1, is2, is3, is4, is5, is6 = st.columns(6)
        metrics = get_median_df(scores_df)
        is1.metric("工作意願和動機", metrics["工作意願和動機"].values[0])
        is2.metric("學習動力", metrics["學習動力"].values[0])
        is3.metric("基本溝通表達", metrics["基本溝通表達"].values[0])
        is4.metric("工作責任感", metrics["工作責任感"].values[0])
        is5.metric("解決問題意願", metrics["解決問題意願"].values[0])

        st.markdown("### 外在指標中間值")
        os1, os2, os3, os4, os5, os6 = st.columns(6)
        os1.metric("社群和社交活動", metrics["社群和社交活動"].values[0])
        os2.metric("自我身心照顧", metrics["自我身心照顧"].values[0])
        os3.metric("家人支持程度", metrics["家人支持程度"].values[0])
        os4.metric("私人企業工作經驗", metrics["私人企業工作經驗"].values[0])
        os5.metric("量化求職考量", metrics["量化求職考量"].values[0])
        os6.metric("先天後天", metrics["先天後天"].values[0])

    def generate_distribution(self, df: pd.DataFrame, features: List[str]):
        dist_df = df.copy()
        color_dict = {"T": "red", "F": "blue"}
        for feature in features:
            num_bins = int(dist_df[feature].max() - dist_df[feature].min() + 1)

            fig = px.histogram(
                dist_df,
                x=feature,
                marginal="box",
                title=f"外部關鍵TA vs 外部非關鍵的{feature}常態分佈",
                nbins=num_bins,
                color="關鍵TA",
                color_discrete_map=color_dict,
            )

            fig.update_layout(
                xaxis=dict(tickmode="linear", tick0=dist_df[feature].min(), dtick=1)
            )

            fig.update_traces(opacity=0.75)
            st.plotly_chart(fig, use_container_width=True)


# dashboard title
st.title(":potable_water: :blue[_若水_]身障就業資料分析")


scores_df = read_data(dataset_url)

placeholder = st.empty()
with placeholder.container():
    fig_col1, fig_col2, fig_col3 = st.columns(3, gap="large")
    with fig_col1:
        st.markdown("## :dart: 定位")
        option = st.selectbox(
            "視覺化圖表",
            (
                "外部關鍵TA的特質常態分佈",
                "內外部關鍵TA的特質雷達圖",
            ),
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

    with fig_col2:
        st.markdown("## :pinching_hand: 篩選")
        option = st.selectbox(
            "視覺化圖表",
            ("foo",),
            index=None,
            placeholder="選擇視覺化圖表",
        )

    with fig_col3:
        st.markdown("## :books: 管道")
        option = st.selectbox(
            "視覺化圖表",
            ("bar",),
            index=None,
            placeholder="選擇視覺化圖表",
        )

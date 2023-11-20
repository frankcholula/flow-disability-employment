# import pandas as pd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import time
from typing import List

dataset_url = "https://raw.githubusercontent.com/frankcholula/flow-disability-employment/main/data/scores.csv"

# 內在指標 + 外在指標
inside_features = ["工作意願和動機", "學習動力", "基本溝通表達", "工作責任感", "解決問題意願"]
outside_features = ["社群和社交活動", "自我身心照顧", "家人支持程度", "私人企業工作經驗", "量化求職考量", "先天後天"]

# page setup
st.set_page_config(
    page_title="🚰若水身障就業資料分析",
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


scores_df = read_data(dataset_url)


class Visualization:
    """_summary_"""

    def __init__(self, vis, *args, **kwargs):
        self.vis = vis
        dispatcher = {
            "personality": self.generate_radar_chart(),
        }

        init_func = dispatcher.get(vis)
        if not init_func:
            raise ValueError(f"Uknown visualization type: {vis}")

    def generate_radar_chart(self, df, max_values, cahrts_per_row):
        return


# dashboard title
st.title("若水身障就業資料分析")


# top-level filters
inside_outside_filter = st.selectbox("選擇內外部", pd.unique(scores_df["內外部"]))
ta_filter = st.selectbox("選擇關鍵TA", pd.unique(scores_df["關鍵TA"]))
scores_df = scores_df[scores_df["內外部"] == inside_outside_filter].reset_index(drop=True)
scores_df = scores_df[scores_df["關鍵TA"] == ta_filter].reset_index(drop=True)

scores_df
# creating a single-element container
placeholder = st.empty()

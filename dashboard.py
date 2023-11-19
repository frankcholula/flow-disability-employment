# import pandas as pd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from typing import List

dataset_url = "https://raw.githubusercontent.com/frankcholula/flow-disability-employment/main/data/scores.csv"

# from plotly.graph_objs import Margin
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go


# # 內在指標
# INSIDE_COLUMNS = ["工作意願和動機", "學習動力", "基本溝通表達", "工作責任感", "解決問題意願"]

# # 外在指標
# OUTSIDE_COLUMNS = ["社群和社交活動", "自我身心照顧", "家人支持程度", "私人企業工作經驗", "量化求職考量", "先天後天"]


st.set_page_config(
    page_title="若水身障就業資料分析",
    page_icon="🚰",
    layout="wide",
)


@st.cache_data
def read_data(file_path) -> List[pd.DataFrame]:
    """
    Read data from csv file and return 3 dataframes
    """
    scores_df = pd.read_csv(file_path)
    ta_df = scores_df[scores_df["關鍵TA"] == "T"]
    ta_df.reset_index(drop=True, inplace=True)
    nonta_df = scores_df[scores_df["關鍵TA"] == "F"]
    nonta_df.reset_index(drop=True, inplace=True)

    return scores_df, ta_df, nonta_df


scores_df, ta_df, nonta_df = read_data(dataset_url)


# scores_df, ta_df, nonta_df = read_data(FILE_PATH)


# class Visualization:
#     """_summary_"""

#     def __init__(self, vis, *args, **kwargs):
#         self.vis = vis
#         dispatcher = {
#             "personality": self.generate_radar_chart(),
#         }

#         init_func = dispatcher.get(vis)
#         if not init_func:
#             raise ValueError(f"Uknown visualization type: {vis}")

#     def generate_radar_chart(self, df, max_values, cahrts_per_row):
#         return

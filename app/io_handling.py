import streamlit as st
from pathlib import Path
import pandas as pd
import time
import os


def save_data():
    data_path = Path(__file__).parent / "data.csv"
    st.session_state["data"].to_csv(data_path, index=False)


def delete_data():
    data_path = Path(__file__).parent / "data.csv"

    os.rename(
        data_path,
        data_path.with_name(f"{data_path.stem}_{time.time()}{data_path.suffix}"),
    )
    st.session_state["data"] = pd.DataFrame(
        {
            "Parameter": pd.Series(dtype="int"),
            "Value": pd.Series(dtype="float"),
        }
    )
    st.session_state.pop("campaign")
    st.session_state["done"] = False


def load_data():
    data_path = Path(__file__).parent / "data.csv"
    if data_path.is_file():
        st.session_state["data"] = pd.read_csv(data_path)
    else:
        st.session_state["data"] = pd.DataFrame(
            {
                "Parameter": pd.Series(dtype="int"),
                "Value": pd.Series(dtype="float"),
            }
        )

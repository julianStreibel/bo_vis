import streamlit as st
import pandas as pd
import numpy as np

from config import (
    MIN_PARAM,
    MAX_PARAM,
)
from plotting import plot
from recommender import recommend, setup_campaign
from io_handling import save_data, load_data, delete_data


def show_recommendation():
    def update_data():
        if (st.session_state["recommendation"] is None) or (
            st.session_state["observed_value"] is None
        ):
            st.error("Recommendation and Observed Value can't be None")
        else:
            st.session_state["data"].loc[
                len(st.session_state["data"]), ["Parameter", "Value"]
            ] = [
                st.session_state["recommendation"],
                float(st.session_state["observed_value"]),
            ]
            save_data()
            if "campaign" not in st.session_state:
                setup_campaign()
            st.session_state["campaign"].add_measurements(
                pd.DataFrame(
                    {
                        "Parameter": [st.session_state["recommendation"]],
                        "Value": st.session_state["observed_value"],
                    }
                )
            )
            recommend()

    st.subheader("Next Experiment")
    st.number_input("Recommendation", key="recommendation", step=1)
    st.number_input("Observed Value", key="observed_value")
    st.button("Submit Experiment", on_click=update_data)


def show_data():
    st.subheader("Experiments")
    st.dataframe(st.session_state["data"], hide_index=True)
    st.button("Delete Campaign", on_click=delete_data)


def init_session():
    st.set_page_config(page_title="Bayesian Optimization App", layout="wide")
    st.session_state["init"] = True
    st.session_state["observed_value"] = None
    load_data()
    if st.session_state["data"].empty:
        st.session_state["recommendation"] = np.random.randint(MIN_PARAM, MAX_PARAM + 1)
    else:
        setup_campaign()
        recommend()


def main():
    if not st.session_state.get("init", False):
        init_session()

    if st.session_state.get("done", False):
        st.balloons()
        n = min(5, len(st.session_state["data"]))
        st.subheader(f"Your top {n} valued parameter settings are:")
        st.dataframe(
            st.session_state["data"].sort_values(
                "Value", ascending=False, ignore_index=True
            )[:n]
        )
        st.button("Restart", on_click=delete_data)

    else:
        st.header("Bayesian Optimization App")
        if not st.session_state["data"].empty:
            st.plotly_chart(
                plot(
                    st.session_state["posterior_mean_np"],
                    st.session_state["posterior_variance_np"],
                    st.session_state["acquisition_samples_np"],
                    st.session_state["x_values_np"],
                    st.session_state["design_points_np"],
                    st.session_state["design_values_np"],
                    st.session_state["candidate_np"],
                    st.session_state["candidate_acqu_np"],
                ),
                use_container_width=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            show_recommendation()
        with col2:
            show_data()


main()

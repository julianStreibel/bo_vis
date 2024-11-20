import streamlit as st
import numpy as np

from config import (
    MIN_PARAM,
    MAX_PARAM,
    N_SAMPLES_VIZ,
    VALUES,
)


def recommend():
    import torch
    from baybe.exceptions import NotEnoughPointsLeftError

    sample_x = torch.linspace(MIN_PARAM, MAX_PARAM, steps=N_SAMPLES_VIZ).reshape(
        -1, 1, 1
    )

    campaign = st.session_state["campaign"]
    recommender = campaign.recommender
    searchspace = campaign.searchspace
    objective = campaign.objective
    try:
        df = campaign.recommend(batch_size=1)
    except NotEnoughPointsLeftError:
        st.session_state["done"] = True
        return

    posterior = recommender.get_surrogate(
        searchspace, objective, st.session_state["data"]
    )._posterior_comp(sample_x)
    acquisition_samples = recommender._botorch_acqf(sample_x.reshape(-1, 1, 1))

    candidate = df.iloc[0]["Parameter"]
    acq_value = recommender._botorch_acqf(torch.tensor(candidate).reshape(-1, 1, 1))

    # Convert to NumPy arrays
    posterior_mean_np = posterior.mean.detach().numpy().reshape(-1)
    posterior_variance_np = posterior.variance.detach().numpy().reshape(-1)
    acquisition_samples_np = acquisition_samples.detach().numpy().reshape(-1)
    x_values_np = sample_x.reshape(-1)
    design_points_np = st.session_state["data"]["Parameter"].values.reshape(-1)
    design_values_np = st.session_state["data"]["Value"].values.reshape(-1)
    candidate_np = np.array([candidate])
    candidate_acqu_np = acq_value.detach().numpy()

    st.session_state["posterior_mean_np"] = posterior_mean_np
    st.session_state["posterior_variance_np"] = posterior_variance_np
    st.session_state["acquisition_samples_np"] = acquisition_samples_np
    st.session_state["x_values_np"] = x_values_np
    st.session_state["design_points_np"] = design_points_np
    st.session_state["design_values_np"] = design_values_np
    st.session_state["candidate_np"] = candidate_np
    st.session_state["candidate_acqu_np"] = candidate_acqu_np

    st.session_state["recommendation"] = candidate_np[0]
    st.session_state["observed_value"] = None


def setup_campaign():
    from baybe.targets import NumericalTarget
    from baybe.objectives import SingleTargetObjective
    from baybe.parameters import NumericalDiscreteParameter
    from baybe.searchspace import SearchSpace
    from baybe import Campaign
    from baybe.recommenders import BotorchRecommender

    target = NumericalTarget(
        name="Value",
        mode="MAX",
    )
    objective = SingleTargetObjective(target=target)

    parameters = [
        NumericalDiscreteParameter(name="Parameter", values=VALUES),
    ]

    searchspace = SearchSpace.from_product(parameters)

    recommender = BotorchRecommender(
        allow_recommending_already_measured=False,
        allow_repeated_recommendations=False,
    )

    campaign = Campaign(searchspace, objective, recommender)
    campaign.add_measurements(st.session_state["data"])
    st.session_state["campaign"] = campaign

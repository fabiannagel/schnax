import torch
import schnetpack as spk



defaults_representation = {
    "cutoff": 5.0,
    "n_interactions": 2,
    "n_atom_basis": 128,
    "n_filters": 128,
    "trainable_gaussians": False,
    "normalize_filter": False,
}

defaults_atomwise = {
    "mean": 0.0,
    "stddev": 1.0,
    "n_layers": 2,
    "n_neurons": None,
}


def load_model(file, cutoff, device):
    dct = torch.load(file)
    representation = dct["config"]["model"]["representation"]
    atomwise = dct["config"]["model"]["atomwise"]
    state = dct["state"]

    config_representation = {
        **defaults_representation,
        **representation,
    }
    assert config_representation["cutoff"] == cutoff
    representation = spk.SchNet(**config_representation)

    config_atomwise = {
        **{"n_in": representation.n_atom_basis},
        **defaults_atomwise,
        **atomwise,
    }
    atomwise = get_atomwise(**config_atomwise)

    model = spk.atomistic.model.AtomisticModel(representation, [atomwise])
    model.load_state_dict(state)
    model.to(device)
    return model


def get_atomwise(n_in, mean, stddev, n_layers, n_neurons):

    return spk.atomistic.Atomwise(
        n_in=n_in,
        mean=torch.tensor(mean),
        stddev=torch.tensor(stddev),
        n_layers=n_layers,
        n_neurons=n_neurons,
        negative_dr=True,
        property="energy",
        derivative="forces",
        # stress=None,
        contributions="energies",
    )

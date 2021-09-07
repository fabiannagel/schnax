from schnetpack.data.atoms import AtomsConverter
from schnetpack.environment import AseEnvironmentProvider


def get_converter(cutoff, device="cuda"):
    return AtomsConverter(
        environment_provider=AseEnvironmentProvider(cutoff=cutoff),
        device=device,
    )

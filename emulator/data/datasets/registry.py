from emulator.data.datasets.climate_model import ClimateModelDataset
from emulator.data.datasets.timescale_aligned import TimescaleAlignedClimateModelDataset


DATASET_CLASS_ALIASES = {
    "ClimateModelDataset": ClimateModelDataset,
    "TimescaleAlignedClimateModelDataset": TimescaleAlignedClimateModelDataset,
}


def resolve_climate_dataset_class(name: str):
    try:
        return DATASET_CLASS_ALIASES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_CLASS_ALIASES))
        raise ValueError(f"Unknown dataset type {name!r}. Expected one of: {valid}") from exc


def is_climate_dataset_name(name: str) -> bool:
    return name in DATASET_CLASS_ALIASES


def canonical_climate_dataset_name(name: str) -> str:
    cls = resolve_climate_dataset_class(name)
    if cls is ClimateModelDataset:
        return "ClimateModelDataset"
    if cls is TimescaleAlignedClimateModelDataset:
        return "TimescaleAlignedClimateModelDataset"
    raise ValueError(f"No canonical dataset name registered for {name!r}.")

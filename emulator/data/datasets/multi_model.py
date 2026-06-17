import torch
from torch.utils.data import ConcatDataset

from emulator.data.datasets.registry import resolve_climate_dataset_class

class MultiClimateModelDataset(torch.utils.data.Dataset):
    """Concatenate one or more climate-model datasets from metadata entries."""

    def __init__(
        self,
        dataset,
        climate_model_metadata,
        testing_dataset=False,
        **kwargs,
    ):
        self.climate_models = []
        self.climate_model_datasets = []

        for metadata in climate_model_metadata:
            kwargs_copy = kwargs.copy()

            climate_model = metadata["model"]

            optional_keys = [
                "scenarios",
                "num_ensemble_member",
                "output_data_dir",
                "IB",
                "align_to_decade",
                "min_start_year",
                "max_start_year",
                "member_variant_labels",
            ]

            for key in optional_keys:
                if key in metadata:
                    kwargs_copy[key] = metadata[key]

            DatasetClass = resolve_climate_dataset_class(dataset)

            # Split scenarios into datasets for testing.
            if testing_dataset:
                for scenario in kwargs_copy["scenarios"]:
                    kwargs_modified = kwargs_copy.copy()
                    kwargs_modified.pop("scenarios", None)
                    self.climate_model_datasets.append(
                        DatasetClass(
                            climate_model=climate_model,
                            scenarios=[scenario],
                            **kwargs_modified,
                        )
                    )
            else:
                self.climate_model_datasets.append(
                    DatasetClass(
                        climate_model=climate_model,
                        **kwargs_copy,
                    )
                )
            self.climate_models.append(climate_model)

        self.ds = ConcatDataset(self.climate_model_datasets)

    def __getitem__(self, index):
        return self.ds.__getitem__(index)

    def __len__(self):
        return len(self.ds)

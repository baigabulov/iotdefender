import enum
import pathlib


CORE_ROOT_PATH = pathlib.Path(__file__).parent.resolve()
DATASET_DIR_PATH = CORE_ROOT_PATH / 'dataset'


class DatasetType(enum.Enum):
    NETWORK = 'NETWORK'
    MALWARE = 'MALWARE'
    OTHER = 'OTHER'

    @property
    def types(self):
        return [
            self.NETWORK,
            self.MALWARE,
        ]

    @classmethod
    def get_dataset_path(cls, dataset_name: 'DatasetType'):
        dataset_name = '.'.join([dataset_name.value.lower(), 'csv'])
        return (DATASET_DIR_PATH / dataset_name).resolve()

    @classmethod
    def get_dataset_model_path(cls, dataset_name: 'DatasetType'):
        model_name = '.'.join([dataset_name.value.lower(), 'joblib'])
        return (DATASET_DIR_PATH / model_name).resolve()

    def get_dataset_paths(self):
        paths = []
        for dataset_type in self.types:
            paths.append((dataset_type.value.lower(), '.'.join([dataset_type.value, 'csv'])))

        return paths

    def get_dataset_model_names(self):
        model_names = []
        for dataset_type in self.types:
            model_names.append((dataset_type.value.lower(), '.'.join([dataset_type.value, 'joblib'])))

        return model_names

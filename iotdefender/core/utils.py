import joblib

from core import DatasetType


def load_model(dataset_type: DatasetType):
    model_path = DatasetType.get_dataset_model_path(dataset_type)
    model = joblib.load(model_path)
    return model


def dump_model(model, dataset_type: DatasetType):
    model_path = DatasetType.get_dataset_model_path(dataset_type)
    joblib.dump(model, model_path)

from typing import Optional

import pandas as pd

from core import DatasetType
from core import utils
from core.dataclasses import PredictionInput
from core.ml.dataset import DataSetMixin


class Predictor(DataSetMixin):

    def __init__(self, dataset_type: DatasetType, output_feature: Optional[str] = None):
        super().__init__(dataset_type=dataset_type, output_feature=output_feature)
        self._model = utils.load_model(self._dataset_type)

    def predict(self, prediction_input: PredictionInput):

        trained_dataset_keys = list(self._data.keys())
        trained_dataset_keys.remove(self._output_feature)

        formatted_params = {param.field_name: param.field_value for param in prediction_input.parameters}

        data = pd.Series(formatted_params, index=trained_dataset_keys)

        predictions = self._model.predict([data])

        return predictions

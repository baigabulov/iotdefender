from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class PredictorParameter:
    field_name: str
    field_value: Any


@dataclass
class PredictionInput:
    parameters: List[PredictorParameter] = field(default_factory=list)

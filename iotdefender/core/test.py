from sklearn.metrics import accuracy_score

from core import DatasetType
from core.dataclasses import PredictionInput, PredictorParameter
from core.ml.trainer import Trainer
from core.ml.predictor import Predictor


# Train
trainer = Trainer(DatasetType.OTHER, output_feature='Species')
trainer.train()


# Predict
prediction_input = PredictionInput(parameters=[
    PredictorParameter('SepalLengthCm', 5.5),
    PredictorParameter('SepalWidthCm', 3.3),
    PredictorParameter('PetalLengthCm', 1.6),
    PredictorParameter('PetalWidthCm', 0.3),
])

predictor = Predictor(DatasetType.OTHER, output_feature='Species')
prediction = predictor.predict(prediction_input)


# Make assurance and estimate clarity of prediction
accuracy = accuracy_score(['Iris-setosa'], prediction)
print(accuracy)

import abc
import dataclasses
import json
import random
from pathlib import Path
from typing import List, Dict
import sklearn
from openai import OpenAI
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from src.base_data import ClientData

@dataclasses.dataclass
class Result:
    accuracy: float
    recall: float
    precision: float
    f1_score: float

class Predictor(abc.ABC):


    def __init__(self):
        self.fitted_data = None

    def fit(self, train_data: List[ClientData], validation_data: List[ClientData], **kwargs):
        return self

    def predict(self, data: List[ClientData] | ClientData = None, *args, add_output_to_client_data = False, **kwargs) -> List[int] | int:
        #kwargs should contain arguments for additional data for in context e.g.
        #should return the client data containing the prediction

        use_parameter_data = data is not None
        input_is_list = isinstance(data, list)
        if use_parameter_data:
            if not input_is_list:
                data = [data]

        preds = self._predict(data or self.fitted_data, *args, **kwargs)

        if use_parameter_data and add_output_to_client_data:
            for data, pred in zip(data, preds):
                data.label = pred

        if use_parameter_data and not input_is_list:
            preds = preds[0]

        return preds

    def _predict(self, data: List[ClientData], *args,  **kwargs) -> List[int]:
        raise NotImplementedError()


    def get_scores(self, y_pred: List[ClientData | int], y_true: List[ClientData | int]) -> Result:
        if isinstance(y_pred[0], ClientData):
            y_pred = [client.label for client in y_pred]
        if isinstance(y_true[0], ClientData):
            y_true = [client.label for client in y_true]


        return Result(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred),
            recall=recall_score(y_true, y_pred),
            f1_score=f1_score(y_true, y_pred)
        )


class RandomPredictor(Predictor):

    def _predict(self, data: List[ClientData], *args, **kwargs) -> List[int]:
        return [random.choice([0,1]) for _ in range(len(data))]



class OpenAIPredictor(Predictor):

    def __init__(self, rulebook_path: str | Path):
        super().__init__()

        with open(rulebook_path, "r") as f:
            self.rules = f.read()

    def _predict(self, data: List[ClientData], *args, **kwargs) -> List[int]:

        PROMPT = """
            HERE is a client data, that we would like to verify that has no inconsistencies.
            We would like to reject the application if something does not add up, or misses a field.
            - Compare fields across documents
            - Check if the description of the client adds up with the numbers and backstories.
            - You can reason for yourself shortly.
            - last line of your response should be a json {'reject': true/false}.
            - Most importanyl reject only if the document breaks one of these rules:
            - {rules}
            
            HERE is the data: {data}
        """
        result = []
        for client in data:
            client = OpenAI()

            response = client.responses.create(
                model="o3-mini",
                input=PROMPT.format(rules=self.rules, data=json.dumps(client)),
            )

            print(response)







import pytest
import torch
import torch.nn as nn
import json
from fastapi.testclient import TestClient
import app as app_module

class FakeModel(nn.Module):
    def forward(self, x):
        logits = torch.zeros((x.shape[0], 10))
        logits[:, 2] = 5.0
        return logits

def fake_load_model(weights_path, classes_path):
    model = FakeModel()
    with open(classes_path) as f:
        class_names = json.load(f)
    return model, class_names


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "load_model", fake_load_model)
    with TestClient(app_module.app) as c:
        yield c




#
# @pytest.fixture(scope="session")
# def client(target_module, name, new_value):
#     model, class_names = fake_load_model(None, "../models/classes.json")
#     with pytest.MonkeyPatch.context() as mp:
#         mp.setattr(app_module.app, "model", model)
#         yield mp
#


# model = FakeModel()
# x = torch.randn(2, 2342)
# y = model(x)
# print(y)
# print(y.shape)



from src.models import build_model
from src.features import ALL_FEATURES

def test_pipeline_builds():
    model = build_model("logreg")
    assert model is not None

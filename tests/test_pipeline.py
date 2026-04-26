"""
Offline tests for MoEPipeline, Grad-CAM, and FastAPI health endpoint.

All tests inject fake checkpoints via tmp_path so no HF Hub downloads occur.
"""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "demo"))

from models.expert import ExpertModel  # noqa: E402
from models.gating import LogitGating  # noqa: E402

EXPERT_NAMES = ["sd15", "sd21", "sdxlbase", "sd35", "flux"]


def _make_fake_checkpoints(tmp_path: Path) -> None:
    expert = ExpertModel(num_classes=2)
    expert_sd = {"state_dict": {"model." + k: v for k, v in expert.state_dict().items()}}

    for name in EXPERT_NAMES:
        ckpt_dir = tmp_path / "experts" / name
        ckpt_dir.mkdir(parents=True)
        torch.save(expert_sd, ckpt_dir / "best-fake.ckpt")

    gating = LogitGating(num_experts=5)
    gating_sd = {
        "state_dict": {"model.gating." + k: v for k, v in gating.state_dict().items()}
    }
    moe_dir = tmp_path / "moe" / "logit"
    moe_dir.mkdir(parents=True)
    torch.save(gating_sd, moe_dir / "best-fake.ckpt")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("checkpoints")
    _make_fake_checkpoints(tmp_path)

    from demo.pipeline import MoEPipeline

    return MoEPipeline(device="cpu", strategy="logit", checkpoints_dir=str(tmp_path))


@pytest.fixture
def dummy_image():
    return Image.new("RGB", (512, 512), color=(128, 64, 32))


def test_predict_schema(pipeline, dummy_image):
    result = pipeline.predict(dummy_image)

    assert set(result.keys()) == {"prediction", "confidence", "alpha_weights", "attributed_source"}
    assert result["prediction"] in {"real", "synthetic"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["alpha_weights"].keys()) == set(EXPERT_NAMES)
    assert abs(sum(result["alpha_weights"].values()) - 1.0) < 1e-4


def test_gradcam_returns_image(pipeline, dummy_image):
    cam_img = pipeline.gradcam(dummy_image)

    assert isinstance(cam_img, Image.Image)
    assert cam_img.size == (256, 256)


def test_api_health(pipeline):
    import demo.api as api_module  # noqa: E402, I001
    from fastapi.testclient import TestClient  # noqa: E402

    api_module.PIPELINE = pipeline

    client = TestClient(api_module.app, raise_server_exceptions=True)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

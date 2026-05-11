"""
test_mlops_artifacts.py — MLOps artifact integrity tests
=========================================================
Tests that ALL critical MLOps JSON files exist, are valid,
and contain required fields. Runs in <5 seconds (no model loading).

Run:
  pytest backend/tests/test_mlops_artifacts.py -v
"""

import json
import os
import pytest
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent.parent
BACKEND     = REPO_ROOT / "backend"
ML          = BACKEND / "ml"
DATASETS    = ML / "datasets"
EXPERIMENTS = ML / "experiments"
MODELS      = ML / "models"


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_json(path: Path) -> dict:
    assert path.exists(), f"Missing file: {path}"
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  1. config.yaml
# ══════════════════════════════════════════════════════════════════════════════
class TestConfig:
    def test_config_exists(self):
        assert (ML / "config.yaml").exists(), "config.yaml missing"

    def test_config_is_valid_yaml(self):
        import yaml
        # with open(ML / "config.yaml") as f:
        with open(ML / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None

    def test_config_has_required_sections(self):
        import yaml
        # with open(ML / "config.yaml") as f:
        with open(ML / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for section in ["shared", "data", "transformer", "lstm", "cnn", "vae", "promotion"]:
            assert section in cfg, f"config.yaml missing section: {section}"

    def test_config_shared_values(self):
        import yaml
        # with open(ML / "config.yaml") as f:
        with open(ML / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        s = cfg["shared"]
        assert s["n_features"] == len(s["features"]), \
            "n_features must equal len(features)"
        assert s["input_len"] > 0
        assert s["horizon"] > 0

    def test_config_data_splits_coherent(self):
        import yaml
        from datetime import datetime
        # with open(ML / "config.yaml") as f:
        with open(ML / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        d = cfg["data"]
        start  = datetime.fromisoformat(d["date_start"])
        end    = datetime.fromisoformat(d["date_end"])
        t_end  = datetime.fromisoformat(d["train_end"])
        v_start= datetime.fromisoformat(d["val_start"])
        assert start < t_end < v_start < end, \
            "Data splits must be chronologically ordered: start < train_end < val_start < end"

    def test_promotion_gate_positive(self):
        import yaml
        # with open(ML / "config.yaml") as f:
        with open(ML / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["promotion"]["min_improvement"] > 0


# ══════════════════════════════════════════════════════════════════════════════
#  2. datasets/manifest.json
# ══════════════════════════════════════════════════════════════════════════════
class TestManifest:
    @pytest.fixture(autouse=True)
    def manifest(self):
        self.data = load_json(DATASETS / "manifest.json")

    def test_manifest_has_cities(self):
        assert "cities" in self.data
        assert len(self.data["cities"]) == 6

    def test_all_parquet_files_exist(self):
        for city, info in self.data["cities"].items():
            p = DATASETS / info["file"]
            assert p.exists(), f"Parquet file missing: {p}"

    def test_parquet_byte_sizes_match(self):
        for city, info in self.data["cities"].items():
            p = DATASETS / info["file"]
            actual_bytes = p.stat().st_size
            assert actual_bytes == info["bytes"], \
                f"{city}.parquet: manifest says {info['bytes']} bytes, got {actual_bytes}"

    def test_manifest_features_match_config(self):
        import yaml
        # with open(ML / "config.yaml") as f:
        with open(ML / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert self.data["features"] == cfg["shared"]["features"], \
            "manifest.json features must match config.yaml shared.features"

    def test_manifest_total_rows(self):
        cities     = self.data["cities"]
        n_cities   = len(cities)
        rows_each  = list(cities.values())[0]["rows"]
        expected   = n_cities * rows_each
        assert self.data["totals"]["total_rows"] == expected

    def test_manifest_date_range(self):
        assert self.data["date_range"]["start"] == "2023-01-01"
        assert self.data["date_range"]["end"]   == "2024-12-31"

    def test_scaler_stats_all_features(self):
        import yaml
        # with open(ML / "config.yaml") as f:
        with open(ML / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        features = cfg["shared"]["features"]
        for feat in features:
            assert feat in self.data["scaler_stats"], \
                f"scaler_stats missing feature: {feat}"
            assert "mean" in self.data["scaler_stats"][feat]
            assert "std"  in self.data["scaler_stats"][feat]
            assert self.data["scaler_stats"][feat]["std"] > 0, \
                f"{feat} std must be positive"


# ══════════════════════════════════════════════════════════════════════════════
#  3. experiments/runs.json
# ══════════════════════════════════════════════════════════════════════════════
class TestRunsLog:
    @pytest.fixture(autouse=True)
    def runs(self):
        self.data = load_json(EXPERIMENTS / "runs.json")["runs"]

    def test_all_models_have_runs(self):
        models = {r["model"] for r in self.data}
        for m in ["transformer", "lstm", "cnn", "vae"]:
            assert m in models, f"No runs found for model: {m}"

    def test_each_run_has_required_fields(self):
        required = ["run_id", "model", "status", "promoted", "started_at",
                    "finished_at", "config", "epoch_logs", "best_epoch", "metrics"]
        for run in self.data:
            for field in required:
                assert field in run, f"Run {run.get('run_id','?')} missing field: {field}"

    def test_epoch_logs_are_non_empty(self):
        for run in self.data:
            assert len(run["epoch_logs"]) > 0, \
                f"Run {run['run_id']} has empty epoch_logs"

    def test_loss_generally_decreases(self):
        """Val loss must fall over the first half of training."""
        for run in self.data:
            logs = run["epoch_logs"]
            if len(logs) < 4:
                continue
            key = "val_mse" if "val_mse" in logs[0] else "val_recon"
            first_half_avg = sum(e[key] for e in logs[:len(logs)//2]) / (len(logs)//2)
            second_half_avg= sum(e[key] for e in logs[len(logs)//2:]) / (len(logs) - len(logs)//2)
            assert second_half_avg < first_half_avg, \
                f"Run {run['run_id']}: val loss did not decrease (first_half={first_half_avg:.4f} second_half={second_half_avg:.4f})"

    def test_promoted_runs_exist_per_model(self):
        for model in ["transformer", "lstm", "cnn", "vae"]:
            model_runs = [r for r in self.data if r["model"] == model]
            promoted   = [r for r in model_runs if r["promoted"]]
            assert len(promoted) >= 1, f"No promoted run found for {model}"

    def test_exactly_one_promoted_run_per_model(self):
        """Only the final best run should be promoted."""
        for model in ["transformer", "lstm", "cnn", "vae"]:
            promoted = [r for r in self.data if r["model"] == model and r["promoted"]]
            assert len(promoted) == 1, \
                f"Expected exactly 1 promoted run for {model}, got {len(promoted)}"

    def test_run_ids_are_unique(self):
        ids = [r["run_id"] for r in self.data]
        assert len(ids) == len(set(ids)), "Duplicate run_ids found"


# ══════════════════════════════════════════════════════════════════════════════
#  4. experiments/best_metrics.json
# ══════════════════════════════════════════════════════════════════════════════
class TestBestMetrics:
    @pytest.fixture(autouse=True)
    def best(self):
        self.data = load_json(EXPERIMENTS / "best_metrics.json")
        self.runs = load_json(EXPERIMENTS / "runs.json")["runs"]

    def test_all_models_have_champion(self):
        for m in ["transformer", "lstm", "cnn", "vae"]:
            assert m in self.data, f"best_metrics.json missing model: {m}"
            assert "champion_run" in self.data[m]

    def test_champion_run_ids_exist_in_runs(self):
        run_ids = {r["run_id"] for r in self.runs}
        for m in ["transformer", "lstm", "cnn", "vae"]:
            champ = self.data[m]["champion_run"]
            assert champ in run_ids, \
                f"Champion {champ} for {m} not found in runs.json"

    def test_model_files_exist(self):
        for m in ["transformer", "lstm", "cnn", "vae"]:
            model_file = ML / self.data[m]["model_file"]
            assert model_file.exists(), f"Model file missing: {model_file}"

    def test_model_file_sizes_match(self):
        for m in ["transformer", "lstm", "cnn", "vae"]:
            model_file = ML / self.data[m]["model_file"]
            expected   = self.data[m]["model_bytes"]
            actual     = model_file.stat().st_size
            assert actual == expected, \
                f"{m}/model.pt: best_metrics says {expected} bytes, got {actual}"

    def test_promotion_gate_respected(self):
        """beat_previous_by must exceed min_improvement for all non-first promotions."""
        gate = self.data["min_improvement"]
        for entry in self.data.get("promotion_history", []):
            if entry["action"] == "PROMOTED" and "Δ=" in entry.get("reason", ""):
                # Extract delta from reason string
                import re
                m = re.search(r"Δ=([\d.]+)", entry["reason"])
                if m:
                    delta = float(m.group(1))
                    assert delta > gate, \
                        f"Promotion {entry['run']} Δ={delta} did not exceed gate={gate}"

    def test_transformer_beats_lstm_beats_cnn(self):
        """Sanity check: transformer should have lower val_mse than lstm and cnn."""
        txf  = self.data["transformer"]["metrics"]["val_mse"]
        lstm = self.data["lstm"]["metrics"]["val_mse"]
        cnn  = self.data["cnn"]["metrics"]["val_mse"]
        assert txf < lstm, f"Transformer ({txf}) should beat LSTM ({lstm})"
        assert lstm < cnn, f"LSTM ({lstm}) should beat CNN ({cnn})"


# ══════════════════════════════════════════════════════════════════════════════
#  5. experiments/comparison_table.json  (generated by compare_models.py)
# ══════════════════════════════════════════════════════════════════════════════
class TestComparisonTable:
    @pytest.fixture(autouse=True)
    def table(self):
        self.data = load_json(EXPERIMENTS / "comparison_table.json")

    def test_winner_is_transformer(self):
        assert self.data["winner"] == "transformer", \
            f"Expected transformer to win, got: {self.data['winner']}"

    def test_rankings_present(self):
        assert len(self.data["rankings"]) == 4, \
            "Should have 4 models ranked"

    def test_rank_1_has_lowest_val_mse(self):
        forecasters = [r for r in self.data["rankings"] if r["rank"] == 1]
        assert len(forecasters) == 1
        winner_mse = forecasters[0]["primary_value"]
        for r in self.data["rankings"]:
            if isinstance(r["rank"], int) and r["rank"] > 1:
                assert winner_mse <= r["primary_value"], \
                    f"Rank 1 ({winner_mse}) should have lowest val_mse, but rank {r['rank']} has {r['primary_value']}"


# ══════════════════════════════════════════════════════════════════════════════
#  6. Model .pt files exist and are non-empty
# ══════════════════════════════════════════════════════════════════════════════
class TestModelFiles:
    @pytest.mark.parametrize("model", ["transformer", "lstm", "cnn", "vae"])
    def test_model_pt_exists(self, model):
        pt = MODELS / model / "model.pt"
        assert pt.exists(), f"{model}/model.pt not found"

    @pytest.mark.parametrize("model", ["transformer", "lstm", "cnn", "vae"])
    def test_model_pt_non_empty(self, model):
        pt = MODELS / model / "model.pt"
        assert pt.stat().st_size > 0, f"{model}/model.pt is empty"

    def test_vae_anomaly_threshold_exists(self):
        threshold_file = MODELS / "vae" / "anomaly_threshold.json"
        assert threshold_file.exists(), "vae/anomaly_threshold.json not found"

    def test_vae_anomaly_threshold_valid(self):
        data = load_json(MODELS / "vae" / "anomaly_threshold.json")
        # Should have some threshold key
        assert len(data) > 0, "anomaly_threshold.json is empty"


# ══════════════════════════════════════════════════════════════════════════════
#  7. Data pipeline smoke test (import only — no data download)
# ══════════════════════════════════════════════════════════════════════════════
class TestDataPipeline:
    def test_data_pipeline_importable(self):
        import sys
        sys.path.insert(0, str(BACKEND))
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "data_pipeline", ML / "data_pipeline.py"
            )
            mod = importlib.util.module_from_spec(spec)
            # Don't exec — just check it can be found
            assert spec is not None
        except Exception as e:
            pytest.fail(f"data_pipeline.py could not be loaded: {e}")

    def test_features_constant_matches_config(self):
        import yaml
        # with open(ML / "config.yaml") as f:
        with open(ML / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        expected_features = cfg["shared"]["features"]
        # Read FEATURES from data_pipeline.py as text
        src = (ML / "data_pipeline.py").read_text()
        for feat in expected_features:
            assert feat in src, \
                f"Feature '{feat}' from config not found in data_pipeline.py"

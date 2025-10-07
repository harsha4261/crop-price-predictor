"""Predictor loader with robust lazy-loading for Gunicorn/Django.

- Avoids loading pickled models at import time (prevents __main__ path issues)
- If saved pickle is incompatible, it is renamed and retrained automatically
"""

# predictor/loader.py
import os
import time
from datetime import datetime
import sys

from apis.predictor.utils.ml_model import DeepLearningCropPredictor
import threading

# Resolve important paths
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(UTILS_DIR, "models", "best_deep_learning_model")
CSV_PATH = os.path.join(UTILS_DIR, "enam price data.csv")

# Global variable to store the predictor (lazy loading)
_PREDICTOR = None
_TRAINING_LOCK = threading.Lock()
_TRAINING_IN_PROGRESS = False


def _rename_incompatible_model(path: str, reason: str) -> None:
    """Rename incompatible model files so future loads don't hit the same error."""
    try:
        ts = time.strftime("%Y%m%d-%H%M%S")
        
        # Handle deep learning model files (.h5 and _components.pkl)
        h5_path = f"{path}.h5"
        components_path = f"{path}_components.pkl"
        
        if os.path.exists(h5_path):
            new_h5_path = f"{h5_path}.incompatible.{ts}"
            os.rename(h5_path, new_h5_path)
            print(f"[loader] Renamed incompatible model: {h5_path} -> {new_h5_path}. Reason: {reason}")
            
        if os.path.exists(components_path):
            new_components_path = f"{components_path}.incompatible.{ts}"
            os.rename(components_path, new_components_path)
            print(f"[loader] Renamed incompatible components: {components_path} -> {new_components_path}. Reason: {reason}")
            
    except Exception as e:
        print(f"[loader] Failed to rename incompatible model files '{path}': {e}")


def _start_background_training(csv_path: str):
    """Kick off model training in a background thread if not already running."""
    global _TRAINING_IN_PROGRESS, _PREDICTOR

    def _train():
        global _TRAINING_IN_PROGRESS, _PREDICTOR
        try:
            predictor = DeepLearningCropPredictor()
            predictor.train(csv_path)
            # Persist trained model for fast future loads
            try:
                predictor.save_model('best_deep_learning_model')
                print(f"[loader] Saved trained model to {MODEL_PATH}")
            except Exception as save_err:
                print(f"[loader] Warning: failed to save trained model to {MODEL_PATH}: {save_err}")
            _PREDICTOR = predictor
            print("[loader] Background training completed; model ready.")
        except Exception as e:
            print(f"[loader] Background training failed: {e}")
        finally:
            with _TRAINING_LOCK:
                _TRAINING_IN_PROGRESS = False

    with _TRAINING_LOCK:
        if not _TRAINING_IN_PROGRESS:
            _TRAINING_IN_PROGRESS = True
            t = threading.Thread(target=_train, daemon=True)
            t.start()


def get_predictor():
    """Get the predictor instance with lazy loading and robust fallbacks."""
    global _PREDICTOR

    if _PREDICTOR is None:
        # Always create a fresh instance first
        predictor = DeepLearningCropPredictor()
        try:
            # Try to load the saved model (safe path inside class)
            predictor.load_model('best_deep_learning_model')
            print("[loader] Deep Learning model loaded successfully from:", MODEL_PATH)
            print(f"[loader] Model fitted: {predictor.is_fitted}")
            _PREDICTOR = predictor
        except Exception as e:
            # Known scenario under gunicorn: pickled under different __main__
            err_msg = str(e)
            print(f"[loader] Error loading saved model: {err_msg}")
            _rename_incompatible_model(MODEL_PATH, err_msg)

            # Fallback: start background retraining and return fast
            print("[loader] Starting background training; requests will temporarily return 'warming up'.")
            _start_background_training(CSV_PATH)
            raise RuntimeError(
                "Model is warming up (training). Please retry shortly."
            )

    return _PREDICTOR


class LazyPredictor:
    def __getattr__(self, name):
        predictor = get_predictor()
        return getattr(predictor, name)

    def predict(self, *args, **kwargs):
        predictor = get_predictor()
        return predictor.predict(*args, **kwargs)


# Initialize lazy predictor (no heavy work at import time)
PREDICTOR = LazyPredictor()


def get_status() -> dict:
    """Return the current loader/model status."""
    return {
        "ready": _PREDICTOR is not None,
        "training": _TRAINING_IN_PROGRESS,
        "model_path": MODEL_PATH,
        "csv_path": CSV_PATH,
    }

# # predictor/loader.py
# import os
# import joblib
# from datetime import datetime
# import tensorflow as tf
# import sys


# # Limit TF threading to avoid conflicts
# try:
#     tf.config.threading.set_intra_op_parallelism_threads(1)
#     tf.config.threading.set_inter_op_parallelism_threads(1)
# except Exception:
#     pass

# from apis.predictor.utils.ml_model import CropPriceRandomForestPredictor

# MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/best_rf_model.pkl")

# try:
#     PREDICTOR = joblib.load(MODEL_PATH)
#     print("Model loaded from:", MODEL_PATH)
# except Exception as e:
#     raise RuntimeError(f"Failed to load model at {MODEL_PATH}. Error: {e}")

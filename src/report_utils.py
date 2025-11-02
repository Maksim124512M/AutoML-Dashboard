import joblib

from visual_utils import best_model, best_name

joblib.dump(best_model, f"models/{best_name}.pkl")
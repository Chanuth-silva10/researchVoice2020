import os
from app.config import settings
from app.services.predict_emotion_from_heart_stress import (
    PredictEmotionFromHeartAndStress,
)


class HeartStressDetectionFunction:
    def predict_heart_stress(data):
        model_path = os.path.join(
            settings.ROOT_DIR, "util", "heart_stress_model", "random_forest_model.pkl"
        )
        PEFHS = PredictEmotionFromHeartAndStress(model_path)
        return PEFHS.prediction(data)

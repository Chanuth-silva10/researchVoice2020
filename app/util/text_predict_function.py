import os
from app.config import settings
from app.services.predict_emotion_from_text import PredictEmotionFromText


class TextPredictionFunction:
    def predict_emo_and_suicidal(text):
        folder_path = os.path.join(settings.ROOT_DIR, "util", "text_model")
        text_emo_model_path = os.path.join(folder_path, "emotions.keras")
        text_emo_tokenizer_path = os.path.join(folder_path, "tokenizer.json")
        text_suicidal_model_path = os.path.join(folder_path, "suicidal.keras")
        PEFT = PredictEmotionFromText(
            text_emo_model_path, text_emo_tokenizer_path, text_suicidal_model_path
        )

        text_emotion = PEFT.emo_prediction(text)
        text_suicidal = PEFT.suicidal_prediction(text)
        response = {
            "text_emotion_state": text_emotion,
            "text_suicidal_state": text_suicidal,
        }
        return response

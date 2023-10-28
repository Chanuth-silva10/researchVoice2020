import os
from app.config import settings
from keras.models import load_model
from app.services.predict_emotion_from_face import *
import shutil


class FaceRecordFunc:
    def predict_emotion(file):
        upload_dir = os.path.join(settings.ROOT_DIR, "util", "upload_face")

        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        dest = os.path.join(upload_dir, file.filename)

        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        emotion_detection_model = os.path.join(
            settings.ROOT_DIR,
            "util",
            "face_model",
            "emotion_detection_model_10epochs.h5",
        )

        model = load_model(emotion_detection_model)

        PEFF = PredictEmotionFromFace
        emotion = PEFF.prediction(dest, model)

        stress = ""

        response = {"emotion": emotion, "stress": stress}
        return response

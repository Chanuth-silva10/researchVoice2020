from app.services.predict_emotion_from_voice_cnn import *
from app.config import settings

import os
from tensorflow.keras.models import Sequential, model_from_json

class VoiceRecordFuncCNN:
    def record_audio():
        PEFC = PredictEmotionFromVoiceCNN()

        rec_duration = 10

        rec_sub_dir = os.path.join(
            settings.ROOT_DIR,"util","tmp",
            "voice_recording.wav",
        )

        # PEFC.voice_recording(rec_sub_dir, duration=rec_duration)
        return "Recording Successfully"
    
    def predict_emotion(audio):
        json_file_path = os.path.join(
            settings.ROOT_DIR,"util","train_model",
            "CNN_model.json",
        )

        load_weight_path_for_new_model = os.path.join(
            settings.ROOT_DIR,"util","train_model",
            "best_model1_weights.h5",
        )

        scaler_path = os.path.join(
            settings.ROOT_DIR,"util","train_model",
            "scaler2.pickle",
        )

        encoder_path = os.path.join(
            settings.ROOT_DIR,"util","train_model",
            "encoder2.pickle",
        )

        PEFC = PredictEmotionFromVoiceCNN(
            json_file_path, load_weight_path_for_new_model, scaler_path, encoder_path
        )

        rec_sub_dir = os.path.join(
            settings.ROOT_DIR,"util","tmp",
            "voice_recording.wav",
        )

        emotion = PEFC.prediction(audio)
        print(emotion)


        response = {"emotion": emotion}
        return response

import time
import os
import numpy as np

# import pyaudio
import wave
import librosa
from tensorflow.keras.models import model_from_json
import pickle


class PredictEmotionFromVoiceCNN:
    def __init__(
        self,
        json_file_path=None,
        load_weight_path_for_new_model=None,
        scaler_path=None,
        encoder_path=None,
    ):
        if (
            json_file_path is not None
            and load_weight_path_for_new_model is not None
            and scaler_path is not None
            and encoder_path is not None
        ):
            self.json_file = open(json_file_path, "r")
            self.loaded_model_json = self.json_file.read()
            self.json_file.close()
            self.loaded_model = model_from_json(self.loaded_model_json)
            # load weights into new model
            self.loaded_model.load_weights(load_weight_path_for_new_model)

            print("Loaded model from disk")

            with open(scaler_path, "rb") as f:
                self.scaler2 = pickle.load(f)

            with open(encoder_path, "rb") as f:
                self.encoder2 = pickle.load(f)

            print("Done")

    def extract_features(slef, data, sr=22050, frame_length=2048, hop_length=512):
        result = np.array([])

        def zcr(data, frame_length, hop_length):
            zcr = librosa.feature.zero_crossing_rate(
                data, frame_length=frame_length, hop_length=hop_length
            )
            return np.squeeze(zcr)

        def rmse(data, frame_length=2048, hop_length=512):
            rmse = librosa.feature.rms(
                y=data, frame_length=frame_length, hop_length=hop_length
            )
            return np.squeeze(rmse)

        def rmse(data, frame_length=2048, hop_length=512):
            rmse = librosa.feature.rms(
                y=data, frame_length=frame_length, hop_length=hop_length
            )
            return np.squeeze(rmse)

        def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
            mfcc = librosa.feature.mfcc(y=data, sr=sr)
            return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

        result = np.hstack(
            (
                result,
                zcr(data, frame_length, hop_length),
                rmse(data, frame_length, hop_length),
                mfcc(data, sr, frame_length, hop_length),
            )
        )
        return result

    def get_predict_feat(self, path):
        d, s_rate = librosa.load(path, duration=2.5, offset=0.5)
        res = self.extract_features(d)
        result = np.array(res)
        result = np.reshape(result, newshape=(1, 2376))
        i_result = self.scaler2.transform(result)
        final_result = np.expand_dims(i_result, axis=2)

        return final_result

    def prediction(self, path1):
        res = self.get_predict_feat(path1)
        predictions = self.loaded_model.predict(res)
        y_pred = self.encoder2.inverse_transform(predictions)
        return y_pred[0][0]

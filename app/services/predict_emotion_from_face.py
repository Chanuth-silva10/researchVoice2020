import numpy as np
from keras_preprocessing import image


import warnings

warnings.filterwarnings("ignore")


class PredictEmotionFromFace:
    def prediction(file_path, model):
        try:
            img = image.load_img(
                file_path, target_size=(48, 48), color_mode="grayscale"
            )
            img = np.array(img)

            label_dict = {
                0: "Angry",
                1: "Disgust",
                2: "Fear",
                3: "Happy",
                4: "Neutral",
                5: "Sad",
                6: "Surprise",
            }

            img = np.expand_dims(img, axis=0)  # makes image shape (1,48,48)
            img = img.reshape(1, 48, 48, 1)
            result = model.predict(img)
            result = list(result[0])
            img_index = result.index(max(result))
            return label_dict[img_index]
        except Exception:
            return "Error"

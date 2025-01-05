import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

class MSLModel:
    def __init__(self, model_path: str):
        # Load the pre-trained model and labels dictionary
        model_dict = pickle.load(open(model_path, 'rb'))
        self.model = model_dict['model']
        self.labels_dict = model_dict['labels_dict']

        # Initialize MediaPipe hands module
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        self.last_prediction_time = time.time()

    def processFrame(self, frame: str) -> str:
        """
        Process a single frame to detect sign language gesture.
        Args:
        - frame (str): Path to the frame file to be processed.
        
        Returns:
        - str: The detected sign or gesture.
        """
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        predicted_character = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                               self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                               self.mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            if time.time() - self.last_prediction_time >= 2:
                my_array = [np.asarray(data_aux)]
                if my_array[0].shape[0] <= 42:
                    prediction = self.model.predict(my_array)
                    predicted_character = self.labels_dict[prediction]
                    self.last_prediction_time = time.time()

        return predicted_character

    def processVideo(self, video: str) -> str:
        """
        Process a video file to detect sign language gestures in multiple frames.
        Args:
        - video (str): Path to the video file to process.
        
        Returns:
        - str: A message about the processed video.
        """

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError("Error opening video stream or file.")

        result_sentence = ""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            predicted_char = self.processFrame(frame)
            if predicted_char:
                result_sentence += predicted_char

        cap.release()
        return result_sentence

    def convertTextToSign(self, text: str) -> str:
        """
        Convert text into sign language animation.
        Args:
        - text (str): The text to convert to sign language.
        
        Returns:
        - str: A message indicating the conversion is done.
        """
        # This is a placeholder for the text-to-sign animation conversion logic.
        # The actual implementation should generate or simulate sign language animations.
        # For now, we simulate the conversion.
        return text

# Example Usage
if __name__ == "__main__":
    model = MSLModel('./model/model.pickle')  # Load model

    # Process a single frame (example)
    # frame_path = "./media/sign_language_image.jpg" 
    # frame_path = "./media/ain.jpg" 
    # detected_sign = model.processFrame(frame_path)
    # print(f"Detected sign: {detected_sign}")

    # # Process a video (example)
    video_path = "./media/sign_language_video.mp4"
    result = model.processVideo(video_path)
    print(result)

    # # Convert text to sign (example)
    # text = "Hello"
    # animation_result = model.convertTextToSign(text)
    # print(animation_result)

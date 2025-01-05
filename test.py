import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

model_dict = pickle.load(open('./model/model.pickle', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = model_dict['labels_dict']

sentence = ""
last_prediction_time = time.time()
predicted_character = ""

# Create a new window for the sentence display
cv2.namedWindow("Sentence", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (1920, 1080))
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        if time.time() - last_prediction_time >= 2:
            my_array = [np.asarray(data_aux)]
            if my_array[0].shape[0] <= 42:     
                prediction = model.predict(my_array)
                predicted_character = prediction[0]
                if predicted_character == "space":
                    sentence += " "
                elif predicted_character == "del":
                    sentence = sentence[:-1]
                elif predicted_character == "W":
                    sentence=""
                else :
                    sentence += predicted_character
                last_prediction_time = time.time()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)

    # Display the sentence in a separate window
    sentence_frame = np.zeros((100, 800, 3), np.uint8)
    sentence_frame.fill(255)  # Set frame background to white

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  # Adjust the font size as desired
    font_thickness = 2

    text_size, _ = cv2.getTextSize(sentence, font, font_scale, font_thickness)

    text_x = int((sentence_frame.shape[1] - text_size[0]) / 2)
    text_y = int((sentence_frame.shape[0] + text_size[1]) / 2)

    cv2.putText(sentence_frame, sentence, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    cv2.imshow("Sentence", sentence_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2
import mediapipe as mp
import time



class GESTURES():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        retval = self.cap = cv2.VideoCapture(0)

        self.hands = mp.solutions.hands.Hands(model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        print("Gestures Initialized")
        

    def get_image(self):
        if not self.cap.isOpened():
            print("Cap closed")
            self.cap.open()

        retval, image = self.cap.read()

        #if not retval:
        #    print("Ignoring empty camera frame.")

        return image

    
    def process_image(self, image):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                print("----------------------------")
                print(hand_landmarks)
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        
        return image


    def disploy_output(self, image):
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            exit()

    def run(self):
        while True:
            t_curr = time.perf_counter()
            img = self.get_image()
            img = self.process_image(img)
            self.disploy_output(img)
            t_compute = time.perf_counter() - t_curr
            print(f"Compute time: {t_compute:.3f}\tFPS: {1/t_compute:.2f}")
            


def main():
    print("Start Gestures")
    gestures = GESTURES()
    gestures.run()
    
if __name__ == "__main__":
    main()
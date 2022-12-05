import copy
import numpy as np
import cv2
import mediapipe as mp
import time
import itertools


class MEDIAHANDS():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        #retval = self.cap = cv2.VideoCapture(0)

        self.hands = mp.solutions.hands.Hands(model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        print("Gestures Initialized")


    def process_image(self, image):
        """Detects hand landmarks

        Args:
            image (numpy): Image to process

        Returns:
            image: Image with points drown
            results: Final landmark points
        """
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.hands.process(image)

        if results.multi_hand_landmarks:
            landmark_list_nom_return = [] # Return variable
            landmark_list_return = []
            for hand_landmarks in results.multi_hand_landmarks:
                #print("----------------------------")
                landmark_list = self.process_landmarks(image, hand_landmarks)
                landmark_list_return.append(landmark_list)
                #print(f"list: {landmark_list}")
                landmark_list_nom = self.normalize_landmarks(image, landmark_list)
                landmark_list_nom_return.append(landmark_list_nom) # Append to return variable
                #print(f"nom: {landmark_list_nom}")
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
            return image, landmark_list_nom_return, landmark_list_return
        else:
            return image, None, None


    # Inspired by: https://github.com/kinivi/hand-gesture-recognition-mediapipe/blob/0e737bb8c45ea03f6fafb1f5dbfe9246c34a8003/app.py#L215
    def process_landmarks(self, image, landmarks):
        """Takes raw media hand results and converts it into a list of image points

        Args:
            image (numpy): origional cv image
            landmarks (list): mediahands results

        Returns:
            list: List of landmark points
        """
        image_width, image_height = image.shape[1], image.shape[0]
        
        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    # https://github.com/kinivi/hand-gesture-recognition-mediapipe/blob/0e737bb8c45ea03f6fafb1f5dbfe9246c34a8003/app.py#L231
    def normalize_landmarks(self, image, landmarks):
        """Takes a list of landmarks and converts them to relative distances

        Args:
            image (numpy): origional cv image
            landmarks (list): mediahands results

        Returns:
            list: List of relative landmark points
        """
        temp_landmark_list = copy.deepcopy(landmarks)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        
        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        #print(temp_landmark_list)
        max_value = max(list(map(abs, temp_landmark_list)))
        #print(max_value)

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        # Round results to significant digits + a few
        def round_(n):
            return round(n, 5)

        temp_landmark_list = list(map(round_, temp_landmark_list))

        return temp_landmark_list



    def run(self):
        """Main gesture function
        """
        while True:
            t_curr = time.perf_counter()

            img = self.get_image()
            img, hand_results, hand_points = self.process_image(img)
            img = self.draw_boxinfo(img, hand_points)
            self.display_output(img)

            t_compute = time.perf_counter() - t_curr
            time.sleep(.3)
            #print(f"Compute time: {t_compute:.3f}\tFPS: {1/t_compute:.2f}")


    def run_once(self, img):
        """Main gesture funtion
        """
        t_curr = time.perf_counter()
        img, hand_results, hand_points = self.process_image(img)

        t_compute = time.perf_counter() - t_curr
        #print(f"Compute time: {t_compute:.3f}\tFPS: {1/t_compute:.2f}")

        return hand_results, hand_points


def main():
    print("Start hands")
    hands = MEDIAHANDS()
    hands.run()
    
if __name__ == "__main__":
    main()

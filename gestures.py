import csv
import keyboard
import cv2
import numpy as np
import pickle
from actions import ACTION_CONTROLLER, KEYBOARD_ACTION
from classifier import CLASSIFIER
from mediahands import MEDIAHANDS
from mykb import KEYBOARD_LOG


hands = MEDIAHANDS()
classifier = CLASSIFIER()

class GESTURES():
    def __init__(self):
        self.hands = MEDIAHANDS()
        self.classifier = CLASSIFIER()
        self.mykb = KEYBOARD_LOG()
        
        retval = self.cap = cv2.VideoCapture(0)
        
        self.gesture_table = ["Fist", "One", "Two", "Three", "Four", "Five", "Rock", "Gunk", "L"]
        
    def get_image(self):
        """Capture image frame

        Returns:
            numpy: Image frame
        """
        if not self.cap.isOpened():
            print("Cap closed")
            self.cap.open()

        retval, image = self.cap.read()

        #if not retval:
        #    print("Ignoring empty camera frame.")

        return image
    
    
    def display_output(self, image):
        """Display image on screen

        Args:
            image (numpy): Input image
        """
        cv2.imshow('MediaPipe Hands', image)#, cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            exit()
    
    
    def draw_boxinfo(self, image, hand_points, message="-"):
        # Find extreams
        minx, miny, maxx, maxy = 0, 0, 0, 0
        if hand_points is not None:
            for hand in hand_points:
                hand = np.array(hand)
                # for pair in hand:
                #     print(pair)
                maxx = max(hand[:,0])
                maxy = max(hand[:,1])
                minx = min(hand[:,0])
                miny = min(hand[:,1])

            cv2.rectangle(image, (minx, miny), (maxx, maxy), (255,0,0), 2)

            cv2.rectangle(image, (minx, miny-40), (maxx, miny), (255,30,0), -1)          
            cv2.putText(image, message, (minx, miny-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (36, 255,12), 1)
        
        return image


def detect_gestures():
    print("Start Gestures")
    gesture = GESTURES()
    actions = ACTION_CONTROLLER()
    actions.add(KEYBOARD_ACTION('k'))

    while True:
        image = gesture.get_image()
        hand_results, hand_points = gesture.hands.run_once(image)

        if hand_results is not None:
            print(hand_results)
            result, result_all = classifier.categorize(hand_results)
            actions.call(result)  # Perform gesture action
        else:
            result = "-"

        if isinstance(result, int):
            if len(gesture.gesture_table) > result:
                result = gesture.gesture_table[result]
        image_an = gesture.draw_boxinfo(image, hand_points, message=str(result))
        gesture.display_output(image_an)


def train_gestures():
    gesture = GESTURES()
    mykb = KEYBOARD_LOG()
    key = mykb.pop_key()
    train_data = []
    print("Make a gestures and select key number (0-9). q to quit")
    
    # Get training data
    train_index = 0 # Adds an index to each training data
    while key != "q":
        image = gesture.get_image()
        hand_results, hand_points = gesture.hands.run_once(image)
        gesture.display_output(image)

        if isinstance(key, str):
            if key.isdigit() and hand_results is not None:
                for item in hand_results:
                    train_data.append((train_index, item, int(key)))
                    train_index += 1
        
        #key = keyboard.read_key(suppress=True)
        key = mykb.pop_key()
        if key is not None:
            print(key)
    
    print(f"Training data\n{train_data}")
    
    # Remove any None instances
    print("\nCleaning training data of empties")
    train_data2 = []
    for i, line in enumerate(train_data):
        print(line)
        if line[0] is None:
            continue
        train_data2.append(line)

    # Save data to file
    with open('training_data.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',') #, quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # Generate labels for each point
        fields = ["index"]
        for i in range(42): # for each point in the 21 hand point pairs
            point_no = i//2
            if i%2: # if odd
                fields.append(f"{point_no}b")
            else:
                fields.append(f"{point_no}a")
        fields.append("label")
        writer.writerow(fields)

        # Write data
        for item in train_data2:
            row_data = [item[0]]
            row_data.extend(item[1])
            row_data.append(item[2])
            writer.writerow(row_data)

    # Submit training data to classifier
    classifier.train()

    return


def train_from_file():
    classifier.train()


def main():

    print("Press key for mode: ")
    key = keyboard.read_key()
    print(f"You pressed {key}")

    if key == "t":
        print("Selected Training\n")
        train_gestures()
    elif key == "y":
        print("Selected training with past data\n")
        train_from_file()
    else:
        print("Selected gesture detection\n")
        detect_gestures()

if __name__ == "__main__":
    main()

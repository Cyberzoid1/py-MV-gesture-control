import keyboard
from classifier import CLASSIFIER
from mediahands import MEDIAHANDS


hands = MEDIAHANDS()
classifier = CLASSIFIER()


def detect_gestures():
    print("Start Gestures")
    while True:
        hands_results = hands.run_once()
        print(hands_results)
        if hands_results is not None:
            pass


def train_gestures():
    pass


def main():
    print("Press key for mode: ")
    key = keyboard.read_key()
    print(f"You pressed {key}")

    if key == "t":
        print("Entering Training")
        train_gestures()
    else:
        print("Detecting gestures")
        detect_gestures()

if __name__ == "__main__":
    main()

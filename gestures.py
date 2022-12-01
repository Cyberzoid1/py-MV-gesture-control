import keyboard
import pickle
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
    key = "-"
    train_data = []
    print("Make a gestures and select key number (0-9). q to quit")
    
    # Get training data
    while key != "q":
        #num = int(key)
        hands_results = hands.run_once()
        if isinstance(key, str):
            if key.isdigit():
                for item in hands_results:
                    train_data.append((item, int(key)))
        
        key = keyboard.read_key(suppress=True)
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
    with open('training_data.pickle', 'wb') as f:
        pickle.dump(train_data2, f)
    
    with open('training_data.txt', 'w') as f:
        f.write(str(train_data2))

    # Submit training data to classifier

    return

def train_from_file():
    # Read from file
    with open('training_data.pickle', 'rb') as f:
        training_data = pickle.load(f)

    no_classes = 5
    classifier.train(no_classes, training_data)



def main():
    train_from_file()
    return

    print("Press key for mode: ")
    key = keyboard.read_key()
    print(f"You pressed {key}")

    if key == "t":
        print("Entering Training")
        train_gestures()
    elif key == "y":
        print("Retraining with past data")
        train_from_file()
    else:
        print("Detecting gestures")
        detect_gestures()

if __name__ == "__main__":
    main()

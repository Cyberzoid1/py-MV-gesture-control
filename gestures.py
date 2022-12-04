import csv
import keyboard
import pickle
from classifier import CLASSIFIER
from mediahands import MEDIAHANDS
from mykb import KEYBOARD_LOG


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
    mykb = KEYBOARD_LOG()
    key = mykb.pop_key()
    train_data = []
    print("Make a gestures and select key number (0-9). q to quit")
    
    # Get training data
    train_index = 0 # Adds an index to each training data
    while key != "q":
        hands_results = hands.run_once()
        if isinstance(key, str):
            if key.isdigit():
                for item in hands_results:
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
    with open('training_data.pickle2', 'wb') as f:
        pickle.dump(train_data2, f)

    with open('training_data2.txt', 'w') as f:
        f.write(str(train_data2))
    
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
    no_classes = 5
    classifier.train(no_classes, train_data2)

    return

def train_from_file():
    # Read from file
    with open('training_data.pickle', 'rb') as f:
        training_data = pickle.load(f)

    no_classes = 5
    classifier.train(no_classes, training_data)



def main():

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

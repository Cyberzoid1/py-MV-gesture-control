import keyboard

class KEYBOARD_LOG():
    def __init__(self):
        self.keys = []

        # ---------> hook event handler
        keyboard.on_press(self.onkeypress)
        #keyboard.on_release(self.onkeyrelease) # Doesn't seem to support multiple callbacks

    def list_current_keys(self):
        print(f"Current keys: {self.keys}")
        return self.keys

    def onkeypress(self, event):
        #print(f"Adding character event is: {event}")
        if event.name not in self.keys:
            self.keys.append(event.name)

    def onkeyrelease(self, event):
        print(f"Removing character: {event}")
        if event.name in self.keys:
            index = self.keys.index(event.name)
            print("index ", index)
            self.keys.pop(index)

    def pop_key(self):
        """Returns first character pressed from keyboard then clears list

        Returns:
            string: key press
        """
        if len(self.keys):
            val = self.keys[0]
            self.keys = []
            return val
        else:
            return None



def main():
    import time
    mykb = KEYBOARD_LOG()

    while True:  # making a loop
        time.sleep(1)
        #mykb.list_current_keys()
        print(mykb.pop_key())

if __name__ == "__main__":
    main()

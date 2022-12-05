import time
import keyboard
from time import sleep

class ACTION_CONTROLLER():
    def __init__(self):
        self.actions = {}
    
    def add(self, index, action):
        self.actions[index] = action
    
    def call(self, n):
        try:
            self.actions[n]() # Call action
        except IndexError:
            pass
            #print(f"No action registered for {n}")
        except KeyError:
            pass
            #print(f"No action registered for {n}")
        except Exception as e:
            print("An exception of type {0} occurred. Arguments:\n{1!r}".format(type(e).__name__, e.args))

class KEYBOARD_ACTION():
    def __init__(self, key) -> None:
        self.key = key
        self.lastcalled = 0
        self.cooldown = 2
    
    def __call__(self) -> None:
        """Presses key on keyboard

        Args:
            None

        Returns:
            None
        """
        if (self.lastcalled + self.cooldown) > time.time():
            return # skip

        print(f"Pressing key {self.key}")
        keyboard.press(self.key)
        self.lastcalled = time.time()
        return


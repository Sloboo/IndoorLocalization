from collections import Counter

class LowPassFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.window = []

    def update(self, value):
        # Add the new value to the window
        self.window.append(value)
        
        # Ensure the window doesn't exceed the specified size
        if len(self.window) > self.window_size:
            self.window.pop(0)

        # Return the most common value in the window
        return self._majority_element(self.window)

    def _majority_element(self, inputs):
        if not inputs:
            return None
        count = Counter(inputs)
        return count.most_common(1)[0][0]

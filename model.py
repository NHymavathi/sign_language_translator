# model.py
import numpy as np

# Output classes (order matters)
SIGNS = ["HELLO", "YES", "NO", "THANKYOU", "ILOVEYOU"]

class RuleBasedModel:
    """
    Rule-based placeholder for an LSTM model.
    Uses MediaPipe hand landmarks.
    """

    def predict(self, x):
        """
        x shape: (1, 1, features)
        returns: probability array
        """

        lm = x[0, 0]

        # Fingertip Y-coordinates (MediaPipe)
        thumb  = lm[4 * 3 + 1]
        index  = lm[8 * 3 + 1]
        middle = lm[12 * 3 + 1]
        ring   = lm[16 * 3 + 1]
        pinky  = lm[20 * 3 + 1]

        # Lower Y = finger UP
        fingers_up = [
            thumb < index,
            index < middle,
            middle < ring,
            ring < pinky
        ]

        # ---------------- SIGN RULES ---------------- #

        # 🤟 I LOVE YOU → thumb + index + pinky up
        if thumb < index and pinky < ring and middle > index:
            probs = [0, 0, 0, 0, 1]

        # ✋ HELLO → all fingers up
        elif all(fingers_up):
            probs = [1, 0, 0, 0, 0]

        # 👍 YES → thumb up only
        elif thumb < index and index > middle and middle > ring:
            probs = [0, 1, 0, 0, 0]

        # ✊ NO → fist (all fingers down)
        elif thumb > index and index > middle and middle > ring:
            probs = [0, 0, 1, 0, 0]

        # 🙏 THANK YOU → fingers together (index≈middle≈ring)
        elif abs(index - middle) < 0.02 and abs(middle - ring) < 0.02:
            probs = [0, 0, 0, 1, 0]

        else:
            probs = [1, 0, 0, 0, 0]  # default HELLO

        return np.array(probs, dtype=np.float32)


# Model instance
model = RuleBasedModel()




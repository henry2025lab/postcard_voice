
import queue
import threading
import tkinter as tk

from recorder1 import AudioRecorder
from algorithm1 import EmotionClassifier
from gui1 import EmotionGUI


def recognition_worker(audio_q, result_q, recognizer):
    while True:
        item = audio_q.get()
        if item is None:
            break
        samples, sr, ts = item
        try:
            label, score = recognizer.predict(samples, sr)
            if label and score is not None and score >= recognizer.threshold:
                result_q.put((ts, label, score))
        except Exception as e:gROUP 6
            print("Recognition error:", e)
        finally:
            audio_q.task_done()

def main():
    audio_q = queue.Queue()
    result_q = queue.Queue()

    recognizer = EmotionClassifier()

    def handle_segment(samples, sr, ts):
        audio_q.put((samples, sr, ts))

    recorder = AudioRecorder(handle_segment)

    worker = threading.Thread(
        target=recognition_worker,
        args=(audio_q, result_q, recognizer),
        daemon=True,
    )
    worker.start()

    root = tk.Tk()
    app = EmotionGUI(root, recorder, result_q)
    try:
        app.run()
    finally:
        recorder.stop_recording()
        audio_q.put(None)
        worker.join()



if __name__ == "__main__":
    main()


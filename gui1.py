# import queue
# import tkinter as tk
# from tkinter.scrolledtext import ScrolledText
# import csv
# import tkinter.filedialog
#
#
# class EmotionGUI:
#     def __init__(self, root: tk.Tk, recorder, result_queue: "queue.Queue"):
#         self.root = root
#         self.recorder = recorder
#         self.result_queue = result_queue
#         self.root.title("Emotion Recorder")
#
#         self.results_log = []  # 存储识别结果用于导出
#
#         self.start_btn = tk.Button(root, text="Start Recording", command=self.start)
#         self.start_btn.pack(pady=5)
#         self.stop_btn = tk.Button(
#             root, text="Stop Recording", state=tk.DISABLED, command=self.stop
#         )
#         self.stop_btn.pack(pady=5)
#
#         self.output = ScrolledText(root, height=10, width=50)
#         self.output.pack(padx=5, pady=5)
#         self.output.insert(tk.END, "Ready\n")
#
#         self.export_btn = tk.Button(
#             root, text="Export to CSV", command=self.export_to_csv
#         )
#         self.export_btn.pack(pady=5)
#
#         self._poll_results()
#
#     def start(self):
#         self.recorder.start_recording()
#         self.start_btn.config(state=tk.DISABLED)
#         self.stop_btn.config(state=tk.NORMAL)
#
#     def stop(self):
#         self.recorder.stop_recording()
#         self.start_btn.config(state=tk.NORMAL)
#         self.stop_btn.config(state=tk.DISABLED)
#
#     def _poll_results(self):
#         try:
#             while True:
#                 ts, label, score = self.result_queue.get_nowait()
#                 self.output.insert(tk.END, f"{ts:.2f}s: {label} ({score:.2f})\n")
#                 self.output.see(tk.END)
#                 self.results_log.append((ts, label, score))  # 记录结果
#         except queue.Empty:
#             pass
#         self.root.after(100, self._poll_results)
#
#     def export_to_csv(self):
#         if not self.results_log:
#             print("No data to export.")
#             return
#
#         file_path = tkinter.filedialog.asksaveasfilename(
#             defaultextension=".csv",
#             filetypes=[("CSV files", "*.csv")],
#             title="Save CSV File"
#         )
#         if not file_path:
#             return
#
#         with open(file_path, mode='w', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow(["Timestamp (s)", "Emotion Label", "Score"])
#             writer.writerows(self.results_log)
#
#         print(f"Data exported to {file_path}")
#
#     def run(self):
#         self.root.mainloop()

import os
import queue
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import csv
import tkinter.filedialog
import platform

# On Windows, use winsound to play WAV async; else fallback to playsound
if platform.system() == 'Windows':
    try:
        import winsound
    except ImportError:
        winsound = None
    playsound = None
else:
    try:
        from playsound import playsound
    except ImportError:
        playsound = None

class EmotionGUI:
    def __init__(self, root: tk.Tk, recorder, result_queue: "queue.Queue"):
        self.root = root
        self.recorder = recorder
        self.result_queue = result_queue
        self.root.title("Emotion Recorder")

        self.results_log = []  # 存储识别结果用于导出

        # 录音控制按钮
        self.start_btn = tk.Button(root, text="Start Recording", command=self.start)
        self.start_btn.pack(pady=5)
        self.stop_btn = tk.Button(root, text="Stop Recording", state=tk.DISABLED, command=self.stop)
        self.stop_btn.pack(pady=5)

        # 日志输出区
        self.output = ScrolledText(root, height=10, width=50)
        self.output.pack(padx=5, pady=5)
        self.output.insert(tk.END, "Ready\n")

        # 导出按钮
        self.export_btn = tk.Button(root, text="Export to CSV", command=self.export_to_csv)
        self.export_btn.pack(pady=5)

        # 匹配音频播放按钮
        self.match_btn = tk.Button(root, text="Play Matched Audio", command=self.play_matched_audio)
        self.match_btn.pack(pady=5)

        self._poll_results()

    def start(self):
        self.recorder.start_recording()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop(self):
        self.recorder.stop_recording()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _poll_results(self):
        try:
            while True:
                ts, label, score = self.result_queue.get_nowait()
                self.output.insert(tk.END, f"{ts:.2f}s: {label} ({score:.2f})\n")
                self.output.see(tk.END)
                self.results_log.append((ts, label, score))  # 记录结果
        except queue.Empty:
            pass
        self.root.after(100, self._poll_results)

    def export_to_csv(self):
        if not self.results_log:
            self.output.insert(tk.END, "No data to export.\n")
            return

        file_path = tkinter.filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save CSV File"
        )
        if not file_path:
            return

        with open(file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp (s)", "Emotion Label", "Score"])
            writer.writerows(self.results_log)

        self.output.insert(tk.END, f"Data exported to {file_path}\n")

    def play_matched_audio(self):
        """
        播放与脚本同级目录下名为 'matched_audio.wav' 的音频文件。
        Windows 下使用 winsound；其他平台尝试 playsound。
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = 'matched_audio.wav'
        file_path = os.path.join(script_dir, file_name)
        if not os.path.exists(file_path):
            self.output.insert(tk.END, f"Matched audio file '{file_name}' not found.\n")
            return

        # Windows 平台优先 winsound
        if platform.system() == 'Windows' and winsound:
            try:
                winsound.PlaySound(file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except RuntimeError as e:
                self.output.insert(tk.END, f"winsound error: {e}\n")
        else:
            if playsound:
                threading.Thread(target=playsound, args=(file_path,), daemon=True).start()
            else:
                self.output.insert(tk.END, "Error: No suitable playback method available.\n")

    def run(self):
        self.root.mainloop()

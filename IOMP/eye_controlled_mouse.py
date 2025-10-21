import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
import threading
import sys
# Redirect stdout to GUI
class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, text):
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
    def flush(self):
        pass
# Eye-controlled logic in a thread
class EyeMouseController(threading.Thread):
    def __init__(self, output_widget):
        super().__init__()
        self.running = False
        self.output_widget = output_widget
    def run(self):
        cam = cv2.VideoCapture(0)
        face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        screen_w, screen_h = pyautogui.size()
        cursor_history = []
        calibrated = False
        calib_blink_distances = []
        dragging = False
        blink_start_time = None
        blink_duration = 1  # seconds
        SMOOTHING_WEIGHT = 0.2
        MAX_DELTA = 40
        left_blink_threshold = 0.004
        right_blink_threshold = 0.004
        face_timeout = 5
        last_face_time = time.time()
        print("🟡 Calibrating... Please blink 5 times normally (both eyes)")
        while self.running:
            _, frame = cam.read()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = face_mesh.process(rgb_frame)
            frame_h, frame_w, _ = frame.shape
            if output.multi_face_landmarks:
                last_face_time = time.time()
                landmarks = output.multi_face_landmarks[0].landmark
                left_eye = landmarks[468]
                right_eye = landmarks[473]
                eye_x = (left_eye.x + right_eye.x) / 2
                eye_y = (left_eye.y + right_eye.y) / 2
                x = int(eye_x * frame_w)
                y = int(eye_y * frame_h)
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
                if not cursor_history:
                    cursor_history.append((screen_x, screen_y))
                else:
                    last_x, last_y = cursor_history[-1]
                    smooth_x = last_x * (1 - SMOOTHING_WEIGHT) + screen_x * SMOOTHING_WEIGHT
                    smooth_y = last_y * (1 - SMOOTHING_WEIGHT) + screen_y * SMOOTHING_WEIGHT
                    cursor_history = [(smooth_x, smooth_y)]
                    dx = np.clip(smooth_x - pyautogui.position()[0], -MAX_DELTA, MAX_DELTA)
                    dy = np.clip(smooth_y - pyautogui.position()[1], -MAX_DELTA, MAX_DELTA)
                    pyautogui.moveRel(dx, dy)
                left_upper = landmarks[159]
                left_lower = landmarks[145]
                left_blink = abs(left_upper.y - left_lower.y)
                right_upper = landmarks[386]
                right_lower = landmarks[374]
                right_blink = abs(right_upper.y - right_lower.y)
                if not calibrated:
                    calib_blink_distances.append((left_blink, right_blink))
                    cv2.putText(frame, f"Calibrating... Blink {5 - len(calib_blink_distances)} more times",
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if len(calib_blink_distances) >= 5:
                        left_blink_threshold = np.mean([x[0] for x in calib_blink_distances]) * 0.7
                        right_blink_threshold = np.mean([x[1] for x in calib_blink_distances]) * 0.7
                        calibrated = True
                        print(f"✅ Calibration done.")
                        print(f"🔹 Left blink threshold: {left_blink_threshold:.6f}")
                        print(f"🔹 Right blink threshold: {right_blink_threshold:.6f}")
                else:
                    left_closed = left_blink < left_blink_threshold
                    right_closed = right_blink < right_blink_threshold
                    if left_closed:
                        if blink_start_time is None:
                            blink_start_time = time.time()
                        elif time.time() - blink_start_time >= blink_duration:
                            if not dragging:
                                pyautogui.mouseDown()
                                dragging = True
                                print("🟠 Drag started")
                    else:
                        blink_start_time = None
                        if dragging:
                            pyautogui.mouseUp()
                            dragging = False
                            print("⚪ Drag ended")
                    if left_closed and not right_closed and not dragging:
                        pyautogui.click()
                        cv2.putText(frame, "Left Click", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        time.sleep(0.3)
                    if right_closed and not left_closed:
                        pyautogui.rightClick()
                        cv2.putText(frame, "Right Click", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                        time.sleep(0.3)
                    nose = landmarks[1]
                    nose_y = int(nose.y * frame_h)
                    if nose_y < frame_h * 0.3:
                        pyautogui.scroll(40)
                        print("⬆️ Scroll Up Triggered")
                    elif nose_y > frame_h * 0.7:
                        pyautogui.scroll(-40)
                        print("⬆️ Scroll Down Triggered")
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(frame, "Dragging" if dragging else "", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if time.time() - last_face_time > face_timeout:
                    print("⏱ No face detected for too long. Exiting...")
                    break
            cv2.imshow("Eye Controlled Mouse", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
    def stop(self):
        self.running = False
# GUI Setup
class EyeMouseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Controlled Mouse")
        self.controller = None
        self.start_btn = tk.Button(root, text="Start", command=self.start_tracking, bg="green", fg="white", font=("Arial", 12))
        self.start_btn.pack(pady=10)
        self.stop_btn = tk.Button(root, text="Stop", command=self.stop_tracking, bg="red", fg="white", font=("Arial", 12))
        self.stop_btn.pack(pady=10)
        self.output = scrolledtext.ScrolledText(root, width=60, height=15, font=("Courier", 10))
        self.output.pack(padx=10, pady=10)
        sys.stdout = TextRedirector(self.output)
    def start_tracking(self):
        if not self.controller or not self.controller.is_alive():
            self.controller = EyeMouseController(self.output)
            self.controller.running = True
            self.controller.start()
            print("▶️ Tracking started")
    def stop_tracking(self):
        if self.controller and self.controller.is_alive():
            self.controller.stop()
            print("⏹ Tracking stopped")
# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = EyeMouseApp(root)
    root.mainloop()
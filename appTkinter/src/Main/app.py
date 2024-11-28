# app.py
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os
from pathlib import Path
import time
from Main.Image.image_emotion_detector import EmotionDetector
from Main.Text.text_emotion_detector import EmotionDetector_text
from Main.Audio.audio_emotion_detector import EmotionDetector_audio
import numpy as np

# Mapa de emociones
emotion_map = {0: 'Enojado', 1: 'Asco', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}
emotion_map_audio_model = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}

class FacialEmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facialex")
        self.root.geometry("900x500")
        self.color = {"background":'#17181A', "green":'#11CBC8', "darkgreen":'#01403F'}
        self.root.configure(bg=self.color["background"])

        self.detector = EmotionDetector()
        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()

        self.setup_ui()
        self.update_camera()
        self.update_time()

    def setup_ui(self):
        SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
        OUTPUT_PATH = Path(SCRIPT_DIR)
        ASSETS_PATH = OUTPUT_PATH / "src" / "btn"

        def relative_to_assets(path: str) -> Path:
            return ASSETS_PATH / Path(path)

        # Load Images
        save_image = ImageTk.PhotoImage(Image.open(relative_to_assets("save.png")).resize((210, 30)))
        close_image = ImageTk.PhotoImage(Image.open(relative_to_assets("close.png")).resize((210, 30)))
        voice_image_btn = ImageTk.PhotoImage(Image.open(relative_to_assets("voice.png")).resize((100, 30)))
        face_image_btn = ImageTk.PhotoImage(Image.open(relative_to_assets("face.png")).resize((100, 30)))

        # Main Frame
        main_frame = tk.Frame(self.root, bg=self.color["background"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure((0, 1), weight=1)

        # Left Frame
        left_frame = tk.Frame(main_frame, bg=self.color["background"])
        left_frame.grid(row=0, column=0, sticky="nsew", padx=50, pady=10)
        left_frame.grid_rowconfigure((0, 1), weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # Camera Frame
        camera_frame = tk.Frame(left_frame, bg="white", width=400, height=300)
        camera_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.camera_label = tk.Label(camera_frame, bg="white")
        self.camera_label.pack(expand=True, fill=tk.BOTH)

        # Info Frame
        info_frame = tk.Frame(left_frame, bg=self.color["background"])
        info_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        for i in range(3):
            info_frame.grid_columnconfigure(i, weight=1)

        self.time_label = tk.Label(info_frame, text="00:00", font=("Arial", 16), bg=self.color["background"], fg="#00E3C8")
        self.time_label.grid(row=0, column=0, padx=20)
        tk.Label(info_frame, text="Tiempo Transcurrido", font=("Arial", 10), bg=self.color["background"], fg="white").grid(row=1, column=0)

        self.emotion_label = tk.Label(info_frame, text="Enojo", font=("Helvetica", 16), bg=self.color["background"], fg="#00E3C8")
        self.emotion_label.grid(row=0, column=1, padx=20)
        tk.Label(info_frame, text="Emoción Predominante", font=("Helvetica", 10), bg=self.color["background"], fg="white").grid(row=1, column=1)

        self.confidence_label = tk.Label(info_frame, text="98%", font=("Helvetica", 16), bg=self.color["background"], fg="#00E3C8")
        self.confidence_label.grid(row=0, column=2, padx=20)
        tk.Label(info_frame, text="Nivel de Confianza", font=("Helvetica", 10), bg=self.color["background"], fg="white").grid(row=1, column=2)

        # Right Frame
        right_frame = tk.Frame(main_frame, bg=self.color["background"])
        right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=4)
        right_frame.grid_rowconfigure(2, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Mode Buttons
        mode_label = tk.Label(right_frame, text="Modo", font=("Helvetica", 14), bg=self.color["background"], fg="white")
        mode_label.pack(pady=5)

        btn_frame = tk.Frame(right_frame, bg=self.color["background"])
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, image=face_image_btn, bg=self.color["background"], activebackground=self.color["background"], border=0, command=self.mode_face).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, image=voice_image_btn, bg=self.color["background"], activebackground=self.color["background"], border=0, command=self.mode_voice).grid(row=0, column=1, padx=5)

        # Statistics
        stats_label = tk.Label(right_frame, text="Estadísticas", font=("Helvetica", 14), bg=self.color["background"], fg="white")
        stats_label.pack(pady=5)

        fig = Figure(figsize=(5, 4), tight_layout=True, facecolor=self.color["background"])
        self.ax = fig.add_subplot(111)
        self.setup_graph()

        self.canvas = FigureCanvasTkAgg(fig, master=right_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH, pady=5)

        # Buttons
        tk.Button(right_frame, image=save_image, bg=self.color["background"], activebackground=self.color["background"], border=0, width=210, height=30, command=self.save_info).pack(pady=2)
        tk.Button(right_frame, image=close_image, bg=self.color["background"], activebackground=self.color["background"], border=0, width=210, height=30, command=self.close_app).pack(pady=2)

        # Footer
        tk.Label(self.root, text="v1.0.0", font=("Helvetica", 8), bg=self.color["background"], fg="white").pack(side=tk.BOTTOM, pady=5)

    def setup_graph(self):
        self.ax.set_facecolor(self.color["background"])
        for spine in self.ax.spines.values():
            spine.set_color('white')
        self.ax.set_title("Emociones", color="white", fontsize=10)
        self.ax.set_ylabel("Confianza", color="white")
        self.ax.yaxis.set_label_position("right")
        self.ax.tick_params(colors='white')

    def update_graph(self, predictions):
        self.ax.clear()
        self.setup_graph()
        emotions = list(emotion_map.values())
        self.ax.bar(emotions, predictions[0], color=self.color["green"])
        self.ax.set_xticklabels(emotions, rotation=90, color='white')
        self.canvas.draw()

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detect_faces(gray)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                self.detector.frame_counter += 1

                if self.detector.frame_counter % 15 == 0:
                    emotion, confidence, predictions = self.detector.predict_emotion(roi_gray)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f'Emocion: {emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                    self.emotion_label.config(text=emotion)
                    self.confidence_label.config(text=f"{confidence}%")
                    self.update_graph(predictions)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (400, 300))
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.camera_label.config(image=img)
            self.camera_label.image = img

        self.root.after(20, self.update_camera)

    def update_time(self):
        elapsed = int(time.time() - self.start_time)
        minutes, seconds = divmod(elapsed, 60)
        self.time_label.config(text=f"{minutes:02}:{seconds:02}")
        self.root.after(1000, self.update_time)

    def save_info(self):
        print("Guardar información presionado")

    def close_app(self):
        self.cap.release()
        self.root.destroy()

    def mode_face(self):
        pass  # Implement mode change

    def mode_voice(self):
        pass  # Implement mode change

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialEmotionApp(root)
    root.mainloop()
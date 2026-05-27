import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from tensorflow.keras.models import load_model

# =====================================
# LOAD MODEL
# =====================================
model = load_model("flower_model.keras")

print("MODEL INPUT SHAPE:", model.input_shape)

# =====================================
# LOAD CLASS NAMES
# =====================================
with open("classes.txt", "r") as f:
    class_names = [x.strip() for x in f.readlines()]

# =====================================
# CONFIG
# =====================================
IMG_SIZE = 64

# =====================================
# GUI
# =====================================
root = tk.Tk()
root.title("Flower Detector")
root.geometry("1000x700")

camera_label = tk.Label(root)
camera_label.pack()

result_label = tk.Label(
    root,
    text="Prediction: None",
    font=("Arial", 20)
)
result_label.pack(pady=10)

# =====================================
# VARIABLES
# =====================================
cap = None
current_frame = None
frame_counter = 0
is_predicting = False

# =====================================
# DETECT FLOWER
# =====================================
def detect_flower():

    global current_frame
    global is_predicting

    if current_frame is None:
        return

    is_predicting = True

    try:

        # copy frame
        img = current_frame.copy()

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize đúng input model
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # normalize
        img = img.astype(np.float32) / 255.0

        # shape = (1,64,64,3)
        img = np.expand_dims(img, axis=0)

        print("INPUT SHAPE =", img.shape)

        # predict
        pred = model.predict(img, verbose=0)

        print("PRED =", pred)

        # class index
        class_id = np.argmax(pred)

        # confidence
        confidence = pred[0][class_id]

        # flower name
        flower = class_names[class_id]

        # update GUI
        result_label.config(
            text=f"{flower} ({confidence*100:.2f}%)"
        )

    except Exception as e:

        print("ERROR =", e)

        result_label.config(
            text=str(e)
        )

    is_predicting = False

# =====================================
# CAMERA LOOP
# =====================================
def show_frame():

    global current_frame
    global frame_counter
    global is_predicting

    if cap is None:
        return

    ret, frame = cap.read()

    if ret:

        # lưu frame gốc
        current_frame = frame.copy()

        # convert để hiển thị
        display_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # resize màn hình
        display_frame = cv2.resize(
            display_frame,
            (800, 500)
        )

        # convert PIL
        img = Image.fromarray(display_frame)

        # convert Tkinter
        imgtk = ImageTk.PhotoImage(image=img)

        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

        # auto detect mỗi 15 frame
        frame_counter += 1

        if frame_counter >= 15:

            frame_counter = 0

            if not is_predicting:

                threading.Thread(
                    target=detect_flower,
                    daemon=True
                ).start()

    # loop camera
    camera_label.after(10, show_frame)

# =====================================
# OPEN CAMERA
# =====================================
def open_camera():

    global cap

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        result_label.config(
            text="Không mở được camera"
        )

        return

    show_frame()

# =====================================
# CLOSE APP
# =====================================
def close_app():

    global cap

    if cap is not None:
        cap.release()

    root.destroy()

# =====================================
# BUTTONS
# =====================================
btn_open = tk.Button(
    root,
    text="Open Camera",
    command=open_camera,
    font=("Arial", 16),
    width=20
)

btn_open.pack(pady=5)

btn_detect = tk.Button(
    root,
    text="Detect Flower",
    command=lambda: threading.Thread(
        target=detect_flower,
        daemon=True
    ).start(),
    font=("Arial", 16),
    width=20
)

btn_detect.pack(pady=5)

btn_exit = tk.Button(
    root,
    text="Exit",
    command=close_app,
    font=("Arial", 16),
    width=20
)

btn_exit.pack(pady=5)

# =====================================
# WINDOW CLOSE EVENT
# =====================================
root.protocol(
    "WM_DELETE_WINDOW",
    close_app
)

# =====================================
# RUN APP
# =====================================
root.mainloop()
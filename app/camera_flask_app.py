import datetime
import os
import time
from threading import Thread

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from keras.models import load_model

global capture, rec_frame, switch, neg, face, rec, out, predicted_data
capture = 0
neg = 0
face = 0
switch = 1
rec = 0
predicted_data = [0, 0, 0, 0, 0, 0, 0]

model_path = "./saved_model/emotions_model.h5"
model = load_model(model_path)

# make shots directory to save pics
try:
    os.mkdir("./shots")
except OSError as error:
    pass

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe(
    "./saved_model/deploy.prototxt.txt",
    "./saved_model/res10_300x300_ssd_iter_140000.caffemodel",
)

# instatiate flask app
app = Flask(__name__, template_folder="./templates")


camera = cv2.VideoCapture(0)


def record(out):
    global rec_frame
    while rec:
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


def get_emotion(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resized_image_init = cv2.resize(img, (48, 48))
    img = np.expand_dims(resized_image_init, axis=0)
    resized_image = resized_image_init.reshape(1, 48, 48, 1)
    predicted_values = model.predict(resized_image)
    predicted_values = list(predicted_values[0])
    predicted_values = predicted_values
    predicted_values = [np.float32(value) for value in predicted_values]
    predicted_values = [float(value) for value in predicted_values]
    global capture
    if capture:
        capture = 0
        now = datetime.datetime.now()
        p = os.path.sep.join(
            ["shots", "KEEEEEEk{}.png".format(str(now).replace(":", ""))]
        )
        cv2.imwrite(p, resized_image_init)

    return predicted_values


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame, predicted_data
    predicted_data = [0, 0, 0, 0, 0, 0, 0]
    while True:
        success, frame = camera.read()
        if success:
            with app.app_context():
                app.config["PREDICTED_DATA"] = predicted_data
            if face:
                frame = detect_face(frame)

            if neg:
                frame = cv2.bitwise_not(frame)
            if capture:
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(
                    ["shots", "shot_{}.png".format(str(now).replace(":", ""))]
                )
                cv2.imwrite(p, frame)

            if rec:
                rec_frame = frame
                frame = cv2.putText(
                    cv2.flip(frame, 1),
                    "Recording...",
                    (0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    4,
                )
                frame = cv2.flip(frame, 1)
            predicted_data = get_emotion(frame)
            try:
                ret, buffer = cv2.imencode(".jpg", cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            except Exception as e:
                pass

        else:
            pass


@app.route("/")
def index():
    predicted_data = app.config.get("PREDICTED_DATA", [])
    return render_template("index.html", predicted_data=predicted_data)


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/predicted_data")
def get_predicted_data():
    return jsonify(predicted_data)


@app.route("/requests", methods=["POST", "GET"])
def tasks():
    global switch, camera
    if request.method == "POST":
        if request.form.get("click") == "Capture":
            global capture
            capture = 1
        elif request.form.get("neg") == "Negative":
            global neg
            neg = not neg
        elif request.form.get("face") == "Face Only":
            global face
            face = not face
            if face:
                time.sleep(4)
        elif request.form.get("stop") == "Stop/Start":
            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get("rec") == "Start/Stop Recording":
            global rec, out
            rec = not rec
            if rec:
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out = cv2.VideoWriter(
                    "vid_{}.avi".format(str(now).replace(":", "")),
                    fourcc,
                    20.0,
                    (640, 480),
                )
                # Start new thread for recording the video
                thread = Thread(
                    target=record,
                    args=[
                        out,
                    ],
                )
                thread.start()
            elif rec is False:
                out.release()

    elif request.method == "GET":
        predicted_data = app.config.get("PREDICTED_DATA", [])
        return render_template("index.html", predicted_data=predicted_data)
    predicted_data = app.config.get("PREDICTED_DATA", [])
    return render_template("index.html", predicted_data=predicted_data)


if __name__ == "__main__":
    app.run()

camera.release()
cv2.destroyAllWindows()

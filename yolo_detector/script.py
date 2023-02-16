import cv2
from flask import Flask, request, jsonify
import requests
import json
import threading
import time
import numpy as np

app = Flask(__name__)
net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")
sem = threading.Semaphore()

def detect_cars(image, blob):
    sem.acquire()
    start_time = time.time()
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    predictions = net.forward(output_layers)
    sem.release()
    boxes = []
    confidences = []
    class_ids = []
    centers = []
    for prediction in predictions:
        for detection in prediction:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            # Filter out weak detections
            if confidence > 0.5:
                # Get detection coordinates
                x, y, w, h = (detection[0:4] * np.array(
                    [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype("int")
                x = int(x - w / 2)
                y = int(y - h / 2)

                center_x = int(x + w // 2)
                center_y = int(y + h // 2)

                centers.append((center_x, center_y))
                boxes.append([int(x), int(y), int(x + w), int(y + h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # color = (0, 255, 0)
                # thickness = 2
                # cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
                # center = (x + w // 2, y + h // 2)
                #
                # # Increment number of cars
                # print(class_id)
                # num_cars += 1
                # # if classes[class_id] == "car":
                # #    num_cars += 1

    print("Time taken: "+str(time.time()-start_time))
    
    return boxes, confidences, class_ids, centers

@app.route("/detect", methods=["POST"])
def detect():
    image = np.array(request.json["image"]).astype(int)
    blob = np.array(request.json["blob"])
    bs, cnfs, cids, ces = detect_cars(image, blob)
    resp = jsonify(
        boxes=bs,
        confidences=cnfs,
        class_ids=cids,
        centers=ces
    )
    print(resp)
    return resp

@app.route("/")
def index():
    print("test")
    return "test_route"

if __name__ == "__main__":
    print("Starting flask app")
    app.run(host='0.0.0.0', port=5001)

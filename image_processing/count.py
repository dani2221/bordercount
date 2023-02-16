import math

from PIL import Image
import cv2
import numpy as np
import requests
import json
import time
import sys
import base64

classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
           "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
           "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
           "baseball glove",
           "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
           "bowl",
           "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
           "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
           "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
           "vase",
           "scissors", "teddy bear", "hair drier", "toothbrush"]

url_yolo = 'yolo:5001' if sys.argv[2] == 'prod' else 'localhost:5001'
url_api = 'api' if sys.argv[2] == 'prod' else 'localhost:5290'

def read_stream():
    f = open('sources.json')
    data = json.load(f)
    return data


def preprocess(image):
    resized_image = cv2.resize(image, (416, 416))
    blob = cv2.dnn.blobFromImage(resized_image, 1 / 255, (416, 416), swapRB=True, crop=False)
    return resized_image, blob


def NMS(boxes, confidences, threshold):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    scores = confidences
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


# def detect_cars(image, blob):
#     net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
#     net.setInput(blob)
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#     predictions = net.forward(output_layers)
#     boxes = []
#     confidences = []
#     class_ids = []
#     centers = []
#     for prediction in predictions:
#         for detection in prediction:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             # Filter out weak detections
#             if confidence > 0.5:
#                 # Get detection coordinates
#                 x, y, w, h = (detection[0:4] * np.array(
#                     [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype("int")
#                 x = int(x - w / 2)
#                 y = int(y - h / 2)

#                 center_x = (x + w // 2)
#                 center_y = (y + h // 2)

#                 centers.append((center_x, center_y))
#                 boxes.append([x, y, x + w, y + h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#                 # color = (0, 255, 0)
#                 # thickness = 2
#                 # cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
#                 # center = (x + w // 2, y + h // 2)
#                 #
#                 # # Increment number of cars
#                 # print(class_id)
#                 # num_cars += 1
#                 # # if classes[class_id] == "car":
#                 # #    num_cars += 1

#     return boxes, confidences, class_ids, centers

def detect_cars_api(image, blob):
    response = requests.post('http://'+url_yolo+'/detect',json={"image": image.tolist(), "blob": blob.tolist()})
    boxes = response.json()['boxes']
    confidences = response.json()['confidences']
    class_ids = response.json()['class_ids']
    centers = response.json()['centers']
    return boxes, confidences, class_ids, centers


def visualize(boxes, centers, class_ids, keep, image, colors, lanes, ccs_with_pos, border_name):
    for lane in lanes:
        cv2.polylines(image, [np.array(lane).reshape((-1, 1, 2))], False, (0,0,255), 2)
    for i, (box, c, ci) in enumerate(zip(boxes, centers, class_ids)):
        if i not in keep:
            continue
        class_list = ['car', 'motorbike', 'bus', 'truck', 'person']
        if classes[ci] in class_list:
            color = colors[class_list.index(classes[ci])]
            thickness = 2
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (w, h), color, thickness)

    if ccs_with_pos is not None:
        prev_vehs = list([None for el in range(len(lanes))])
        for i in range(416, 0, -1):
            for car in ccs_with_pos:
                if car[0][1] == i and car[1] != -1:
                    if prev_vehs[car[1]] is None:
                        prev_vehs[car[1]] = car
                        cv2.putText(image, 'lane ' + str(car[1]+1), (car[0][0]-10, car[0][1]+10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.line(image, car[0], prev_vehs[car[1]][0], (255,0,0), 2)
                        prev_vehs[car[1]] = car
    #cv2.imshow("", image)
    post_image(image, border_name)


def get_car_centers(centers, class_ids, keep):
    car_centers = []
    for index, (c, i) in enumerate(zip(centers, class_ids)):
        if index not in keep:
            continue
        if i == 2:
            car_centers.append(c)
    return car_centers


def get_displacement(old_center, new_center, width=416, x_weight=20):
    if old_center == new_center:
        return np.inf
    if old_center[0] < width / 2 and new_center[0] < width / 2:
        if new_center[0] - old_center[0] < 0 and new_center[1] > old_center[1]:
            x_displacement = abs(old_center[0] - new_center[0]) * x_weight / 2
        else:
            x_displacement = abs(old_center[0] - new_center[0]) * x_weight * 2
    else:
        if old_center[0] > width / 2 and new_center[0] > width / 2:
            if new_center[0] - old_center[0] > 0 and new_center[1] > old_center[1]:
                x_displacement = abs(old_center[0] - new_center[0]) * x_weight / 2
            else:
                x_displacement = abs(old_center[0] - new_center[0]) * x_weight * 2
        else:
            x_displacement = abs(old_center[0] - new_center[0]) * x_weight * 2

    y_displacement = new_center[1] - old_center[1]
    if y_displacement < 0:
        return np.inf
    dist = x_displacement + y_displacement
    if dist > 1000:
        return np.inf
    return x_displacement + y_displacement


def get_x_at_y(line, y):
    loc = -1
    for segment in range(len(line)-1):
        if line[segment][1] <= y <= line[segment + 1][1] or line[segment][1] >= y >= line[segment + 1][1]:
            loc = segment
    if loc == -1:
        return None
    slope = (line[segment+1][1] - line[segment][1]) / (line[segment+1][0]-line[segment][0])
    return (y-line[segment+1][1])/slope + line[segment+1][0]


def track_lanes(car_centers, image, lanes):
    if len(car_centers) < 2:
        return []
    ccs_with_pos = []
    for cc in car_centers:
        xs = []
        for lane in lanes:
            xs.append(get_x_at_y(lane, cc[1]))
        dists = []
        if None in xs:
            ccs_with_pos.append((cc, -1))
            continue
        for x in xs:
            dists.append(x-cc[0])
        if dists[0]>0 or dists[-1]<0:
            ccs_with_pos.append((cc, -1))
            continue
        for i in range(len(dists)-1):
            if dists[i] < 0 < dists[i + 1]:
                ccs_with_pos.append((cc, i))
                break
    return ccs_with_pos


def check_movement(frame, prev_frame, check_points):
    for i, point in enumerate(check_points):
        roi_new = frame[point[0]-20:point[0]+20, point[1]-20:point[1]+20]
        roi_old = prev_frame[point[0]-20:point[0]+20, point[1]-20:point[1]+20]
        mse = np.square(np.subtract(roi_new, roi_old)).mean()
        #print("lane: "+str(i+1), "mse: "+str(mse), "point: "+str(point))


def optical_flow(image, prev_image, features_to_track, lanes, cars_in_lanes):
    if len(cars_in_lanes) == 0:
        return list([-1 for el in range(len(lanes))])
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=20,
                          blockSize=7)
    lk_params = dict(winSize=(30, 30),
                     maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_gray = cv2.convertScaleAbs(prev_image)
    prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
    p0 = np.array(features_to_track).reshape((-1, 1, 2))
    p0 = np.float32(p0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

    # Filter good points
    good_new = p1[st == 1]
    good_prev = p0[st == 1]

    # Draw the optical flow tracks
    lane_speeds = list([0 for el in range(len(lanes))])
    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()
        xs_new = []
        xs_old = []
        for lane in lanes:
            xs_new.append(get_x_at_y(lane, y_new))
            xs_old.append(get_x_at_y(lane, y_prev))
        dists_new = []
        dists_old = []
        if None in xs_new or None in xs_old:
            continue
        for x in xs_new:
            dists_new.append(x - x_new)
        for x in xs_old:
            dists_old.append(x - x_prev)
        if dists_new[0] > 0 or dists_new[-1] < 0 or dists_old[0] > 0 or dists_old[-1] < 0:
            continue
        lane_new = None
        lane_old = None
        for j in range(len(dists_new) - 1):
            if dists_new[j] < 0 < dists_new[j + 1]:
                lane_new = j
                break
        for j in range(len(dists_old) - 1):
            if dists_old[j] < 0 < dists_old[j + 1]:
                lane_old = j
                break
        lane_speeds[lane_new] += math.dist([x_new, y_new],[x_prev, y_prev])

    for i, ls in enumerate(lane_speeds):
        cnt = 0
        for car in cars_in_lanes:
            if car[1] == i:
                cnt += 1
        if cnt == 0:
            lane_speeds[i] = -1
            continue
        lane_speeds[i] = lane_speeds[i]/cnt

    return lane_speeds


def calc_movement(new_movement, old_movement, sum_movement):
    for i in range(len(new_movement)):
        if new_movement[i] > old_movement[i] + 1:
            sum_movement[i] += 1
    return sum_movement


def cars_per_lane(ccs, sum_cars):
    for i in range(len(sum_cars)):
        for car in ccs:
            if car[1] == i:
                sum_cars[i]+=1
    return sum_cars


def post_data(movement, cars, border, elapsed_minutes):
    response = requests.post('http://'+url_api+'/api/BorderInformationInterval',
                             json={
                                  "border": border,
                                  "speedLanes": movement,
                                  "carLanes": cars,
                                  "prevMinutes": elapsed_minutes
                                })
    print("data", response)


def post_image(img, border):
    _, jpeg_data = cv2.imencode('.jpg', img)
    base64_data = base64.b64encode(jpeg_data).decode()
    response = requests.post('http://'+url_api+'/api/BorderImage',
                             json={
                                 "border": border,
                                 "image": base64_data,
                             })
    print("img", response)


def loop_frames(brd):
    # cv2.namedWindow("Detected Objects")
    # cv2.setMouseCallback("Detected Objects", click_event)
    colors = [[200, 255, 0],
              [50, 240, 50],
              [125, 110, 240],
              [80, 130, 160],
              [180, 40, 220]]

    border = read_stream()[int(brd)]
    old_frame = None
    prev_movement = list([-1 for el in range(len(border['lines']))])
    sum_movement = list([-1 for el in range(len(border['lines']))])
    sum_cars = list([0 for el in range(len(border['lines']))])
    start_time = time.time()
    while True:
        cap = cv2.VideoCapture(border['url'])
        ret, frame = cap.read()
        if not ret:
            break

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break

        image, blob = preprocess(frame)

        if old_frame is not None:
            mse = np.mean((image - old_frame) ** 2)
            if mse < 0.001:
                old_frame = image
                continue
        boxes, confs, class_ids, centers = detect_cars_api(image, blob)
        keep = NMS(boxes, np.array(confs), 0.2)
        if len(border['lines']) > 0:
            cars_lanes = track_lanes(get_car_centers(centers, class_ids, keep), image, border['lines'])
            if old_frame is not None:
                # check_movement(frame, old_frame, border['movementPoints'])
                new_movement = optical_flow(image, old_frame, centers, border['lines'], cars_lanes)
                sum_movement = calc_movement(new_movement, prev_movement, sum_movement)
                prev_movement = new_movement

            sum_cars = cars_per_lane(cars_lanes, sum_cars)
            visualize(boxes, centers, class_ids, keep, image, colors, border['lines'], cars_lanes, border['name'])
        else:
            visualize(boxes, centers, class_ids, keep, image, colors, border['lines'], None, border['name'])

        old_frame = image

        if time.time() - start_time > 360:
            post_data(sum_movement, sum_cars, border['name'], math.ceil((time.time() - start_time)/60))
            start_time = time.time()
            sum_movement = list([-1 for el in range(len(border['lines']))])
            sum_cars = list([0 for el in range(len(border['lines']))])


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('[' + str(x) + ',' + str(y) + ']')


if __name__ == "__main__":
    print(cv2.__version__)
    loop_frames(sys.argv[1])

# import the necessary packages
import numpy as np
import imutils
import time
import cv2
import os
import glob
import math
from sort import Sort

# Clear the output directory
files = glob.glob('output/*.png')
for f in files:
    os.remove(f)

# Initialize the SORT tracker
tracker = Sort()
memory = {}
line1 = [(100, 400), (1000, 400)]
counter1 = 0
counter2 = 0

# Set paths directly
input_path = 'input/input_video.mp4'
output_path = 'output/output_video.avi'
yolo_base_path = 'yolo-coco'
confidence_threshold = 0.35
nms_threshold = 0.25

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# Load the COCO class labels YOLO model was trained on
labelsPath = os.path.sep.join([yolo_base_path, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

# Derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([yolo_base_path, "yolov3.weights"])
configPath = os.path.sep.join([yolo_base_path, "yolov3.cfg"])

# Load YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize the video stream, pointer to output video file, and frame dimensions
vs = cv2.VideoCapture(input_path)
writer = None
(W, H) = (None, None)

frameIndex = 0

try:
    prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    frame = adjust_gamma(frame, gamma=1.5)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    center = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                center.append(int(centerY))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    dets = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x + w, y + h, confidences[i]])

    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []

    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                cv2.line(frame, p0, p1, color, 3)

                y_pix_dist = int(y + (h - y) / 2) - int(y2 + (h2 - y2) / 2)
                x_pix_dist = int(x + (w - x) / 2) - int(x2 + (w2 - x2) / 2)
                final_pix_dist = math.sqrt((y_pix_dist * y_pix_dist) + (x_pix_dist * x_pix_dist))
                speed = np.round(1.5 * y_pix_dist, 2)
                text_speed = "{} km/h".format(speed)
                cv2.putText(frame, text_speed, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                if intersect(p0, p1, line1[0], line1[1]):
                    counter1 += 1

            i += 1

    cv2.line(frame, line1[0], line1[1], (0, 255, 255), 4)

    note_text = "NOTE: Vehicle speeds are calibrated only at yellow line. Speed of cars are more stable."
    cv2.putText(frame, note_text, (20, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
    counter_text = "counter:{}".format(counter1)
    cv2.putText(frame, counter_text, (20, 150), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 5)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, 15, (frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    writer.write(frame)
    frameIndex += 1

print("[INFO] cleaning up...")
writer.release()
vs.release()

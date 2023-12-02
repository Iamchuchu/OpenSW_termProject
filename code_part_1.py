import cv2
import numpy as np

# read image
img = cv2.imread("./image/puppies.jpg") # 사진 제대로 넣기
img = cv2.resize(img, (1024, 600)) # reduce some size, also outline of rectangle look more clear. no need to modity the number in cv2.rectangle
height, width, channel = img.shape

print('original image shape:', height, width, channel)

# get blob from image
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
print('blob shape:', blob.shape)

# read coco object names
with open("coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print('number of classes =', len(classes))

# load pre-trained yolo model from configuration and weight files
net = cv2.dnn.readNetFromDarknet('yolov3.cfg.txt', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# set output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print('output layers:', output_layers)

# detect objects
net.setInput(blob)
outs = net.forward(output_layers)

# get bounding boxes and confidence sccres
class_ids = []
confidence_scores = []
boxes = []

for out in outs: # for each detected object

    for detection in out: # for each bounding box

        scores = detection[5:] # scores (confidence) for all classes
        class_id = np.argmax(scores) # class id with the maximum score (confidence)
        confidence = scores[class_id] # the maximum score

        if confidence > 0.5:
            # bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidence_scores.append(float(confidence))
            class_ids.append(class_id)

print('number of detected objects =', len(boxes))

# non maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, 0.5, 0.4)
print('number of final objects =', len(indices))

# draw bounding boxes with labels on image
color1 = (0, 0, 255) # 색깔 여기서 수정 가능해 보임, 빨간색
color2= (255, 0, 0) # 색깔 여기서 수정 가능해 보임, 파란색
font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(f'class {label} detected at {x}, {y}, {w}, {h}')
        if (label == 'Person'):
            color = color1
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

        else:
            color = color2
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

cv2.imshow('Objects', img)
cv2.waitKey()
cv2.destroyAllWindows()
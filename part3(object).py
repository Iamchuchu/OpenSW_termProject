import cv2
import numpy as np

# COCO 데이터 세트의 클래스 이름 파일 로드
class_names_path = "path/to/coco.names"  # coco.names 파일의 경로로 수정
with open(class_names_path, "r") as f:
  class_names = f.read().strip().split("\n")

# YOLO 모델과 가중치 파일 로드
net = cv2.dnn.readNetFromDarknet("path/to/yolov3.cfg", "path/to/yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 사진 파일 경로 입력
image_path = "path/to/image.jpg"

# 이미지 불러오기
image = cv2.imread(image_path)

# 이미지 전처리: 크기 조정 및 정규화
height, width, channels = image.shape
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# 각각의 탐지 결과를 처리
for out in outs:
  for detection in out:
    scores = detection[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]

    if confidence > 0.5:
      # COCO 데이터 세트의 클래스 중 'person'이 아니면 사물로 간주하고 빨간색 네모를 그립니다.
      if class_names[class_id] != "person":
        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)

        x = int(center_x - w / 2)
        y = int(center_y - h / 2)

        # 빨간색 네모 그리기
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 화면에 출력
cv2.imshow("Object Detection", image)

# 'q' 키를 누르면 종료
if cv2.waitKey(0) & 0xFF == ord('q'):
  cv2.destroyAllWindows()

import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

video_path = "data_ext/sanh.mp4"
video_save = "output/output.mp4"
conf_threshold = 0.5
tracking_class = 0

# Khởi tạo DeepSORT
tracker = DeepSort(max_age=50)

# Lựa chọn CPU hoặc GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo YOLOv9
model = DetectMultiBackend(weights = "Weights/yolov9-c.pt", device=device, fuse=True)
model = AutoShape(model)

#load classnames tu file classes.names
classes_path = "data_ext/classes.names"
with open(classes_path, "r") as f:
    class_name = f.read().strip().split('\n')

np.random.seed(42)
colors = np.random.randint(0,255, size=(len(class_name),3))
tracks = []

#Khoi tao VideoCapture de doc tu file video
if video_path.isdigit():
    video_path = int(video_path)
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('Error: Không thể mở video!')
frame_with = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(video_save, fourcc, fps, (frame_with, frame_height))


#Tien hanh doc tung frame tu video
while True:
    # Doc
    ret, frame = cap.read()
    if not ret:
        break
    # Dua qua model de detect
    results = model(frame)

    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if tracking_class is None:
            if confidence < conf_threshold:
                continue
            else:
                if class_id != tracking_class or confidence < conf_threshold:
                    continue

            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    #Cap nhat, gan ID bang DeepSORT
    tracks = tracker.update_tracks(detect, frame = frame)
    #Ve len man hinh cac khung chu nhat kem ID
    for track in tracks:
        if not track.is_confirmed():
            continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)
            label = "{}-{}".format(class_name[class_id], track_id)

            cv2.rectangle(frame,(x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label)*12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show hình ảnh lên màn hinh
    cv2.imshow("YOLOv9 Object Tracking", frame)
    writer.write(frame)
    # Bấm Q thì thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
writer.release()
if __name__ == '__main__':
  try:
      app.run(main)
  except SystemExit:
      pass


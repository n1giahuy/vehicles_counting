# Import Neccessary Library
import warnings 
warnings.filterwarnings('ignore')

from ultralytics import YOLO
import cv2
import numpy as np

def draw_lanes(frame, polygons, colors, alpha=0.3):
    overlay=frame.copy()
    for i, polygon in enumerate(polygons):
        cv2.fillPoly(overlay, [polygon], colors[i])
        cv2.polylines(frame, [polygon], isClosed=True, color=colors[i], thickness=2)
    return cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

def position_checking(point, polygons):
    for i, polygon in enumerate(polygons):
        if cv2.pointPolygonTest(polygon, point, measureDist= False) >=0:
            return i
    return None

def traffic_counting():
    # Video and Model Paths
    mp4_in='videos/test.mp4'
    mp4_out='out.mp4'
    model=YOLO('runs/yolo11/weights/best.pt')

    # Because I get annotations from a 1920x1080 image so we have to scale coordinates. 
    anno_width=1920
    anno_height=1080

    #Unscaled annotations from Roboflow
    raw_polygons=[
        np.array([[867, 63], [873, 233], [576, 377], [524, 310]]),
        np.array([[875, 235], [1173, 444], [806, 660], [575, 379]]),
        np.array([[808, 668], [1494, 923], [1682, 595], [1179, 446]])
    ]
    lane_colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    cap=cv2.VideoCapture(mp4_in)
    frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS)

    scale_x=frame_width / anno_width
    scale_y=frame_height / anno_height

    scaled_polygons=[]
    for poly in raw_polygons:
        scaled_poly=poly.astype(np.float32)
        scaled_poly[:, 0] *= scale_x
        scaled_poly[:, 1] *= scale_y
        scaled_polygons.append(scaled_poly.astype(np.int32))

    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out= cv2.VideoWriter(mp4_out, fourcc, fps, (frame_width, frame_height))

    counts = {
        'lane_0': {'car': 0, 'truck': 0, 'van': 0},
        'lane_1': {'car': 0, 'truck': 0, 'van': 0},
        'lane_2': {'car': 0, 'truck': 0, 'van': 0}
    }
    counted_ids=set()
    while cap.isOpened():
        ret, frame= cap.read()
        if not ret:
            break
        fwl=draw_lanes(frame, scaled_polygons, lane_colors)
        results=model.track(frame, persist=True, tracker='bytetrack.yaml', conf=0.4, iou=0.5)

        if results[0].boxes.id is not None:
            detections=results[0].boxes
            for i in range(len(detections)):
                box=detections.xyxy[i]
                track_id=int(detections.id[i])
                class_id=int(detections.cls[i])
                class_name=model.names[class_id]

                if class_name not in ['car', 'truck', 'van']:
                    continue
                xmin, ymin, xmax, ymax=map(int, box)
                bottom_center_p=((xmin+xmax)/2, ymax)

                # Checking the point
                lane_id=position_checking(bottom_center_p, scaled_polygons)
                if lane_id is not None and track_id not in counted_ids:
                    lane_name=f'lane_{lane_id}'
                    counts[lane_name][class_name] +=1
                    counted_ids.add(track_id)

                cv2.rectangle(fwl, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                label=f'{class_name} ID:{track_id}'
                cv2.putText(fwl, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        y_offset=30
        for i in range(len(scaled_polygons)):
            lane_name=f'lane_{i}'
            lane_text=f"Lane {i}: Cars {counts[lane_name]['car']} Trucks {counts[lane_name]['truck']} Vans {counts[lane_name]['van']}"
            cv2.putText(fwl, lane_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, lane_colors[i], 2)
            y_offset+=30

        out.write(fwl)

    cap.release()
    out.release()
    return counts

if __name__ == '__main__':
    counts=traffic_counting()
    print('Final Counts:', counts)
    
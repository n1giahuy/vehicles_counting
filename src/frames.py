import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split

# Configuration
config=yaml.safe_load(open('src/config.yaml'))
video_path=config['video_path']
data_dir=config['data_dir']
class_names=config['class_names']
# coco_cls= {2: 0, 7: 1}

#Paths 
def create_dirs(data_dir):
    paths={
        'train_img': os.path.join(data_dir, 'train', 'images'),
        'train_lbl': os.path.join(data_dir, 'train', 'labels'),
        'valid_img': os.path.join(data_dir, 'valid', 'images'),
        'valid_lbl': os.path.join(data_dir, 'valid', 'labels')
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

dirs=create_dirs(data_dir)

# Process frames
def process_frame(model, frame, label_path):
    results=model.predict(frame, conf=config['conf'], verbose=False)
    with open(label_path, 'w') as f:
        boxes=results[0].boxes
        for i in range(len(boxes)):
            cc_class=int(boxes.cls[i])
            # if cc_class in coco_cls:
            # new_class = coco_cls[cc_class]
            cx, cy, w, h=boxes.xywhn[i]
            f.write(f'{cc_class} {cx} {cy} {w} {h}\n')

# Process video
def process_video(video_path, dirs):
    model=YOLO(config['model_name'])
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    frame_count=0
    frames=[]
    while True:
        ret, frame= cap.read()
        if not ret:
            break
        if frame_count % 5 ==0:
            frames.append(frame)
        frame_count +=1
    cap.release()

    train_frames, valid_frames=train_test_split(
        frames, 
        train_size=config['train_ratio'], 
        shuffle=True,
        random_state=42)
    
    # Process train frames
    for i, frame in enumerate(tqdm(train_frames, desc='Processing Train Frames ...' ,colour='green')):
        file_name=f'frame_{i:06d}'
        image_path=os.path.join(dirs['train_img'], f'{file_name}.jpg')
        label_path=os.path.join(dirs['train_lbl'], f'{file_name}.txt')
        
        cv2.imwrite(image_path, frame)
        process_frame(model, frame, label_path)
    # Process valid frames
    for i, frame in enumerate(tqdm(valid_frames, desc='Processing Valid Frames ...' ,colour='blue')):
        file_name=f'frame_{i:06d}'
        image_path=os.path.join(dirs['valid_img'], f'{file_name}.jpg')
        label_path=os.path.join(dirs['valid_lbl'], f'{file_name}.txt')
        
        cv2.imwrite(image_path, frame)
        process_frame(model, frame, label_path)

def create_yaml(data_dir, class_names):
    content=f"""train: {os.path.join(data_dir, 'train', 'images')}
valid: {os.path.join(data_dir, 'valid', 'images')}

nc: {len(class_names)}

names: {class_names}
"""
    yaml_path= os.path.join(data_dir, 'data.yaml')
    cvat_path= os.path.join(data_dir, 'obj.names')
    with open(yaml_path, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    process_video(video_path, dirs)
    create_yaml(data_dir, class_names)
    
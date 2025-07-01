import yaml
from ultralytics import YOLO
import os 

def train():
    # Load config.yaml file
    config=yaml.safe_load(open('src/config.yaml'))

    # Set up saving path
    run_dir= os.path.join('runs', 'yolo11')
    ckp_path= os.path.join(run_dir, 'weights', 'last.pt')

    # Saving checkpoint
    if os.path.exists(ckp_path):
        model = YOLO(ckp_path)
        model.train(resume=True)
    else:
        model= YOLO('yolo11n.pt')
        model.train(
            data='datasets/data.yaml',
            epochs= config['epochs'],
            workers= config['num_workers'],
            batch= config['batch_size'],
            device=[1],
            project='runs',
            name= 'yolo11',
        )


if __name__ == "__main__":
    train()
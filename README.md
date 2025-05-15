# Object Detection and Classification of Cars in Images and Videos Using the YOLOv8n

Link to Stanford Cars Dataset: https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset

Link to Stanford Cars Dataset, filtered car videos of ILSVRC2015 ImageNet Video Dataset, and annotations (Download and extract directly in the project folder): <br /> https://drive.google.com/file/d/1ZHV-10N9bWwZJLzNYhixNvlwm7sYmUEz/view?usp=sharing

Model Predictions on videos from ImageNet Video Dataset: <br />
https://youtu.be/5bL3Ujpr800 <br />
https://youtu.be/iPJqsfzfnVw <br />
https://youtu.be/cDQy8-8bbzU <br />
https://youtu.be/iKoYT9NRObA <br />


Model Prediction on video from YouTube: <br />
https://youtu.be/P94nyPQTjA4

## Instructions to set up project

Download files for this project. Then download the datasets.zip file from the link: https://drive.google.com/file/d/1ZHV-10N9bWwZJLzNYhixNvlwm7sYmUEz/view?usp=sharing. Extract this folder directly into the project directory. Open the ECE4990FinalProject.ipynb file in Jupyter Notebook and install and import all the necessary libraries in the first couple of cells. Everything else should be set up to start training the model. 

### Training the model
To start training, run the following cell: 

<pre>
  import os
  # Start training the model with YOLO v8 for 50 epochs
  # NEED THIS LINE
  # vvvvvvvvvvvvv
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  
  from ultralytics import YOLO
  
  # Load a pretrained model 
  model = YOLO("yolov8n.pt")
  
  # Train on your custom Stanford Cars data
  model.train(data="stanford.yaml", epochs=50, imgsz=640, batch=16, plots=False) 
</pre>

### Evaluating the model
After training, an evaluation can be done with the following cell:

<pre>
  # Validation Results on the Stanford Cars Dataset

  from ultralytics import YOLO
  import os
  
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  
  # Load trained model
  model = YOLO("runs/detect/train/weights/best.pt")
  
  # Run validation on your ImageNet val set
  metrics = model.val(
      data=r"stanford.yaml",  # YAML file
      imgsz=640, # Scales images to this size
      conf=0.25, # Confidence threshold of 25%
      iou=0.5, # Intersection over Union threshold of 50%
      save=True,  # Saves for visualization later
      save_txt=True # Saves the labels for predicted boxes
  )
  
  # Print evaluation results
  print("Validation Results:")
  print(metrics)
  print(f"Precision (mp):     {metrics.box.mp:.4f}")
  print(f"Recall (mr):        {metrics.box.mr:.4f}")
  print(f"mAP@0.5:            {metrics.box.map50:.4f}")
  print(f"mAP@0.5:0.95:       {metrics.box.map:.4f}")

</pre>

### Evaluating on the ImageNet Video Dataset
To evaluate this trained model on the ImageNet Video Dataset, run the following cell:

<pre>
  # Validation Results on the ImageNet Video Dataset

  from ultralytics import YOLO
  import os
  
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  
  # Load trained model
  model = YOLO("runs/detect/train/weights/best.pt")
  
  # Run validation on your ImageNet val set
  metrics = model.val(
      data=r"data_imagenet_eval.yaml",  # YAML file
      imgsz=640, # Scales images to this size
      conf=0.25, # Confidence threshold of 25%
      iou=0.5, # Intersection over Union threshold of 50%
      save=True,  # Saves for visualization later
      save_txt=True # Saves the labels for predicted boxes
  )
  
  # Print evaluation results
  print("Validation Results:")
  print(metrics)
  print(f"Precision (mp):     {metrics.box.mp:.4f}")
  print(f"Recall (mr):        {metrics.box.mr:.4f}")
  print(f"mAP@0.5:            {metrics.box.map50:.4f}")
  print(f"mAP@0.5:0.95:       {metrics.box.map:.4f}")

</pre>

# Objective:
Develop predictive model that can determine, given an image, the class and location (bounding box) of objects of 7 types (small truck, medium truck, large truck, bus, van, SUV, or car, labeled as 1–7), according to the labels assigned in the training/validation set.
# Approach:
I used YOLOv5 as the existing detection network that I retrained for the purpose of this project. Fast, precise, and easy to train, with its Pytorch implementation, YOLOv5 made the perfect sense for this task.
Methodology:
1. Setup the data and directories – YOLOv5 accepts labels indexed from 0. The labels.txt file provided to us had class labels from 1-7. So, as a first step, I changed the class range from 1-7 to 0-6. Next was data normalization. The file labels.txt had bounding box coordinates in pixels while Yolo accepts normalized relative coordinates. To do that, I divided bounding box cx, w values with image width and cy, h values with image height. As a last step for data preparation the training and validation labels need to be created per image, such that each image (named 00001.jpeg for example) in a folder called images has a corresponding label (00001.txt) file for it, in a folder called labels. Therefore, I created .txt files for each image id mentioned in labels.txt file. I did so by grouping the training_labels dataframe on img_id and create a corresponding txt file which has the same name as the image id and store it in a folder named labels. Note that the images and labels folder should be inside a folder named train and next to each other. Same process was followed for validation data.
2. Setup YAML files for training – Since I’m training the model on custom data, I created a file named findVehicles.yaml which directs the model as to where the training, validation and testing image folders are located. YOLOv5 can classify objects in 80 categories; in other words, its nc (number of classes) is 80. For the assignment, we are supposed to categorize images into 7 classes. So this custom yaml file also contains nc=7 and their names ('small truck', 'medium truck', 'large truck', 'bus', 'van', 'suv', 'car')
3. Training the model – I create the model using below command:
model = torch.hub.load('yolov5/', 'yolov5s', source='local', pretrained=True, force_reload=False, classes=7)
I use yolov5s as the pretrained model to start training from. It is the smallest and fastest model available. I use the pretrained weights from yolov5s.pt. Note that while working on HPC’s compute node and trying execute the above line, YOLOv5 by default tries to access github repo to download the latest version of yolov5s.pt. to overcome this issue, I manually uploaded the yolov5s.pt file on HPC and changed the source code in hubconf.py line 52 so that it accesses the yolov5s.pt file from my local and not the internet.
To begin training, I created a batch job and stored it in a file named objectDetection.sh. The command in the file that starts training is:
python yolov5/train.py --img 640 --batch 16 --epochs 300 --data yolov5/findVehicles.yaml --freeze 10 > detect.log
As suggested in Yolov5 documentation, I began training the model on default parameters and got decent results. To implement transfer learning, I froze Yolov5 layers to quickly retrain the model on new data. I chose to freeze the model’s backbone layers 0-9. This change reduces accuracy very slightly but speeds up the training.
A sample output image with bounding boxes is shown below:

![00005](https://user-images.githubusercontent.com/4620848/182739765-c7a41864-96d4-48cf-b1bd-33fce9a18d80.jpeg)

# Training/Val loss:
I get below results for input parameters as: epochs=300, batch_size=16, imgsz=640 (default values). I ran this batch job for 12 hours. It timed out at epoch = 83. Seeing the results I realized that running the training for lesser epochs might give similar results. This experiment gave me the best mAP = 0.7171.

<img width="591" alt="image" src="https://user-images.githubusercontent.com/4620848/182724515-440ce15c-4fef-4b17-84d0-2da3cfbc8616.png">

Next, I wanted to test it out for the entire 300 epochs. So I ran the job with input parameters as: epochs=300, batch_size=16, imgsz=640 – for 48 hours. The job ran for upto 198 epochs. This time my mAP score came down. This was because of overfitting. The results of this experiment are shown below:

<img width="511" alt="image" src="https://user-images.githubusercontent.com/4620848/182724583-2cb3ff2c-a5fc-46f9-9fae-b6f6f75323f2.png">

Seeing the above results it became clear that running the job for lesser epochs might give better results. Perhaps, increasing the input image_size, batch_size would make a difference. So I ran experiments for the same. However, img_size = 960 and batch_size = 32 failed with out of memory error. After that I ran the job with img_size = 960 and batch_size = 16 for 70 epochs.
python yolov5/train.py --img 960 --batch 16 --epochs 70 --data yolov5/findVehicles.yaml --freeze 10 > detect.log This gave me comparable results to my first experiment. My mAP was slightly down to 0.7082.
# Bias/Variance Analysis:
As described in the experiments above, when I increased training to 300 epochs, my data model began overfitting after about 100 epochs. I noticed that my train/cls_loss continued to decrease, while val/cls_loss began increasing. This is due to high variance of the model. To reduce this high variance, I ran training for lesser epochs (70, 50, etc.). This helped bring down variance.

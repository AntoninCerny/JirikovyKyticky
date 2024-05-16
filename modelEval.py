import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torchvision import transforms
from cjm_pil_utils.core import resize_img
import cv2
from PIL import Image
#from main import model

TIMESTAMP_FOLDER = "2024-05-16_14-34-12"
EVALUATION_IMAGE_PATH = "train\\images"
EVALUATION_ANOTATION_PATH = "train\\labels"

#EVALUATION_IMAGE_NAME = "RgbImage_2022-05-10_09-06-55-png_3_png.rf.d0c53515e1ce8cde3b43b0c2b81ff02a"
EVALUATION_IMAGE_NAME = "RgbImage_2022-05-10_09-05-11-png_3_png.rf.81af7aa58ce599e804a12e78af3b8095"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#test_img = Image.open(test_file).convert('RGB')


#img = mpimg.imread(os.path.join(EVALUATION_IMAGE_PATH,(EVALUATION_IMAGE_NAME+".jpg")))


img = Image.open(os.path.join(EVALUATION_IMAGE_PATH,(EVALUATION_IMAGE_NAME+".jpg"))).convert('RGB')
#.convert('RGB')
input_img = resize_img(img, divisor=1)








with open(os.path.join(EVALUATION_ANOTATION_PATH,EVALUATION_IMAGE_NAME+".txt")) as file:
    anotationDataString = file.read().split()
anotationData = [float(x) for x in anotationDataString]

transform = transforms.Compose([
    #transforms.Resize((224, 224)),  # Resize image to match model input size
    transforms.ToTensor(),           # Convert image to tensor
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

# Preprocess the image
input_image = transform(img).unsqueeze(0)  # Add batch dimension
input_image = input_image.to(device)

#TODO wrap model in own class in main.py for easy import and loss of need to have same code in two modules 
#start model here
num_keypoints = 3;
# Load a pre-trained model
#anchor generator - more options for default anchor - box with interesting area
anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)





# Load the saved parameters into the model
model.load_state_dict(torch.load(os.path.join(TIMESTAMP_FOLDER,"keypointrcnn_resnet50_fpn.pth")))
#send model to same device as input tensor
model.to(device)

model.eval();
#model.to(device);

# gradients are only usefull for learning 
with torch.no_grad():
    output = model(input_image)

bounding_boxes = output[0]["boxes"].tolist()  # Extract bounding box coordinates
keypoints = output[0]["keypoints"].tolist()  # Extract keypoint positions
#class_probabilities = output['class_probabilities']  # Extract class probabilities


def draw_predictions_matplotlib(image, bounding_boxes, keypoints):
    # Plot the image
    plt.imshow(image)
    ax = plt.gca()

    # Draw bounding boxes on the image
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox  # Extract coordinates
        box_width = x2 - x1
        box_height = y2 - y1
        bbox_rect = plt.Rectangle((x1, y1), box_width, box_height, fill=False, edgecolor='g', linewidth=2)
        ax.add_patch(bbox_rect)

    # Draw keypoints on the image
    for keypoint in keypoints:
        x, y = keypoint  # Extract coordinates
        plt.scatter(x, y, s=50, c='r', marker='o')

    # Show the image with bounding boxes and keypoints
    plt.show()

# Assuming output has the following structure:
# output = (bounding_boxes, keypoints)

# Assuming img is your input image
# Draw bounding boxes and keypoints on the image using Matplotlib
draw_predictions_matplotlib(img, bounding_boxes, keypoints)








"""
#TODO I dont really understand this yet 
# Ensure the model and input data are on the same device
model.to(device);
input_tensor = transforms.Compose([transforms.ToImage(), 
                                   transforms.ToDtype(torch.float32, scale=True)])(input_img)[None].to(device)
"""
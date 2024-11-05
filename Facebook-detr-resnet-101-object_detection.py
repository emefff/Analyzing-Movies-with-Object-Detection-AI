# -*- coding: utf-8 -*-
"""
Spyder-Editor

Dies ist eine temporäre Skriptdatei.
"""

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from time import time

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("torch:", TORCH_VERSION, "; cuda:", CUDA_VERSION, " device:", DEVICE)


# device = "cuda" if torch.cuda.is_available() else "cpu" 
# print(device)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image1 = Image.open(requests.get(url, stream=True).raw)
# image2 = Image.open("Zebras_Serengeti.JPG")
# image3 = Image.open("1920px-Serengeti_Elefantenherde1.jpg")
# image4 = Image.open("Stephansplatz-Fiaker-1382x2048.jpg")
# image5 = Image.open("Destructions_in_Kharkiv_after_Russian_attack,_2024-09-01_(12).jpg")
# image6 = Image.open("Matterhornwolke.jpg")
# image7 = Image.open("shibuya-scramble-retouched-14-of-31-1024x681.jpg")
# image8 = Image.open("shibuya-scramble-retouched-9-of-31-1024x681.jpg")
# image_list = [image2, image3, image4, image5, image6, image7, image8]

# let's open ten random files from test2015 folder:
import os, random
NUM_IMAGES = 10
PATH = "/media/drive2/PYTORCH_DATASETS/test2017/"

image_list = []

for i in range(NUM_IMAGES):
    image_str = random.choice(os.listdir(PATH))
    #print(image_str)
    image = Image.open(PATH+'/'+image_str) # warum notwendig kA
    image_list.append(image)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
model.to(DEVICE)

# save the model ONCE!
# torch.save(model.state_dict(), "/home/mario/pytorchProjects/Testing_HuggingFace_Models/Facebook-detr-resnet-101/Facebook-detr-resnet-101.pth")

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(20,13))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

# threshold for showing result
THRESHOLD = 0.7

durations_list = []
# we want to check more than one image at once:
for image in image_list:
    print("***************************")
    start_time = time()
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    
    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=THRESHOLD)[0]
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )
    
    # vizualize the results
    # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/DETR_minimal_example.ipynb
    import matplotlib.pyplot as plt
    
    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    
    # postprocess model outputs
    width, height = image.size
    postprocessed_outputs = processor.post_process_object_detection(outputs,
                                                                    target_sizes=[(height, width)],
                                                                    threshold=THRESHOLD)
    results = postprocessed_outputs[0]

    end_time = time() 
    durations_list.append(end_time-start_time)
    print(f"*** Inference took {end_time-start_time:.1f} seconds")         
    plot_results(image, results['scores'], results['labels'], results['boxes'])

print("################################################################################")
print("\n*** Average inference time was: ", sum(durations_list)/NUM_IMAGES, "seconds")
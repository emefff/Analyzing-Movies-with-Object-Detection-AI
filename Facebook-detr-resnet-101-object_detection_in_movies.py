# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:00:31 2024

@author: mario
"""

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
from time import time
import glob


###############################################################################
######################### GENERATE IMAGES WITH FFMPEG #########################
###############################################################################
# ffmpeg -i MOVIE.mkv -vf "fps=0.2" frame%04d.jpg # one frame every 5 seconds

# Analyzing 1200-1400 HD images takes ~6 minutes on a throttled (100W) GTX3090
# or 2m15s with 200W

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("torch:", TORCH_VERSION, "; cuda:", CUDA_VERSION, " device:", DEVICE)

image_list = []
labels_total_list = []

# let's open a folder with the jpgs to analyze:
# NUM_IMAGES = 20
#PATH = "/media/drive2/Four Weddings and a Funeral/fps1/"
PATH = "/media/drive2/Lost in Translation/fps1"
if PATH[-1] != '/': PATH = PATH + '/' # forgetting the '/' is annoying

MOVIE_TITLE = 'Lost in Translation'

# open some random images
# import os, random
# for i in range(NUM_IMAGES):
#     image_str = random.choice(os.listdir(PATH))
#     # print(image_str)
#     image = Image.open(PATH+'/'+image_str) 
#     image_list.append(image)

# open all jpgs in a folder
for filename in sorted(glob.glob(PATH+'*jpg')):  # WE NEED TO SORT THEM!!
    im=Image.open(filename)
    image_list.append(im)
print("Number of images: ", len(image_list))

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
model.to(DEVICE)

# save the model ONCE!
# torch.save(model.state_dict(), "/home/mario/pytorchProjects/Testing_HuggingFace_Models\
#                                     /Facebook-detr-resnet-101/Facebook-detr-resnet-101.pth")

# plots an image with detection box, label, score
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

# threshold for result
THRESHOLD = 0.7


###############################################################################
########################### DO INFERENCE ON IMAGES ############################
###############################################################################
# we want to check more than one image and plot it with detection box, label, score
durations_list = [] # we want to measure inference duration

for i, image in enumerate(image_list):
    # print("***************************")
    if i!= 0 and i%100 == 0: # a dot for every 100th frame
        print('.',end='')
    start_time = time()
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    
    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=THRESHOLD)[0]
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        #print(f"Detected {model.config.id2label[label.item()]} with confidence "
        #      f"{round(score.item(), 3)} at location {box}")
        labels_total_list.append((i, model.config.id2label[label.item()]))
    
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
    # print(f"*** Frame nr. {i} inference took {end_time-start_time:.1f} seconds")         
    # plot the image with detection boxes, labels, scores
    # plot_results(image, results['scores'], results['labels'], results['boxes'])

print("\n\n################################################################################")
print("*** Average inference time was: ", sum(durations_list)/len(image_list), "seconds.")
print("################################################################################")

# we need to find out which labels were detected in the movie, per image and also in total
labels_dict = {}
labels_occur_list = []
for i in range(len(image_list)):
    label_list = []
    for label in labels_total_list:
        if label[0]==i:
            #if label[1] not in label_list:
            label_list.append(label[1])
            if label[1] not in labels_occur_list:
                labels_occur_list.append(label[1])
    labels_dict[str(i)] = label_list
    
# print all detected labels
print("Detected labels:", labels_occur_list)


###############################################################################
################ LET'S SEARCH FOR SOME LABELS/ITEMS IN THE DATA ###############
###############################################################################

# choose some labels that are in the labels_occur_list
items_search_list = ['person', 'umbrella', 'tie', 'wine glass', 'bottle', 
                     'potted plant', 'car', 'bus', 'dining table']
items_num = len(items_search_list)

from itertools import cycle # we need to cycle through styles for plotting a few lines
# styles = ['b-', 'r-', 'g-', 'b-', 'c-', 'y-'] # for plot
colors = ['black', 'red', 'green', 'blue', 'cyan', 'yellow', 'magenta', 'royalblue',
          'orangered', 'gold'] # for bar
style_cycler = cycle(colors)

# let's plot a bar plot over all frames with the number of detected items per frame
for i, item in enumerate(items_search_list):
    data_x = []
    data_y = []
    offset = i / (items_num +1)
    for j in range(len(image_list)):
        labels_per_image = labels_dict[str(j)]
        item_num = labels_per_image.count(item)
        data_x.append(j + offset) # we need an offset in x otherwise we overwrite items with same occurences
        data_y.append(item_num)   # at the same frame number
    plt.bar(data_x, data_y, color = next(style_cycler), width = 0.5, label = item)
    data_x.clear()
    data_y.clear()
plt.xlabel('Frame number / 1')
plt.ylabel('Number of items / 1')
plt.legend() 
#plt.tight_layout()
plt.title(MOVIE_TITLE)
#plt.savefig(MOVIE_TITLE+'numimages'+str(len(labels_dict))+'analysis_thresh='+str(THRESHOLD)+'.jpg', dpi=600)
#plt.show()


###############################################################################
############################# SAVE DATA TO JSON ###############################
###############################################################################
# let's save the detected data to a JSON, we don't want to repeat inference, just
# because we don't like the plot.
import json
FILENAME = MOVIE_TITLE+'_numimages='+str(len(labels_dict))+'_analysis_thresh='+str(THRESHOLD)+'_labels_dict.json'
with open(FILENAME, "w") as f:
    json.dump(labels_dict, f, indent=4, sort_keys=False)



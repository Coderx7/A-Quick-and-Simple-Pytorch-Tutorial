# in the name of God the most compassionate the most merciful 
# https://huggingface.co/docs/transformers/en/model_doc/owlv2
# https://github.com/huggingface/notebooks/blob/main/examples/zeroshot_object_detection_with_owlvit.ipynb
# https://huggingface.co/docs/transformers/en/model_doc/owlvit
# https://medium.com/@Mert.A/zero-shot-object-detection-mit-owl-vit-und-huggingface-86c7e568c18a
# https://github.com/NVIDIA-AI-IOT/nanoowl

# For detection, I intend on covering a new trend, rather than going the traditional way. 
# that is, previously we have standalone models that were trained specifically for certain classes. 
# if we wanted a face detection, we would create a model for face detection, if we wanted to detect 
# cows! we would do so for cows, potatos! you guessed it, hardwares, the same thing!. 
# architectures such as retinaface, ssd (single shot detection), yolo(you only look once), etc are afew to name.
# As we saw advancements in the natural lanaguage processing domain, specifically the large language models and 
# their emergent capabilities, we are seeing more approaches that utilize these improvements.
# one of these trends is to use multi-modal representations and few/zero shot capabilities of these models. 
# transformers, and their fusion with a language model, has yielded impressive results. 
# so we are going to talk about a such a model here, for object detetcion that follows this trend and allows us
# to do object detection on a variety of classes/concepts in a zeroshot and single shot manner!
# we are going to see/use owl-vit and see it performs.
# There has two versions so far as Im writing this. 
# The original OWL-ViT model was introduced in May 2022 and is described in Simple Open-Vocabulary Object Detection with Vision Transformers
# paper(https://arxiv.org/abs/2205.06230).
# the second version, OWL-ViT v2, came out in june of 2023. Its the same architecture with a few imporvemnets. 
# like for example, it uses an improved architecture and training recipe that uses self-training on Web image-text
# data as described in Scaling Open-Vocabulary Object Detection paper(https://arxiv.org/abs/2306.09683).
# The core inference architecture of v2 is identical to v1, except that v2 adds an objectness prediction head which
# predicts the (query-agnostic) likelihood that a predicted box contains an object (as opposed to background). 
# The objectness score can be used to rank or filter predictions independently of text queries. 
# therefore the OWL-ViT v2 checkpoints are drop-in replacements for v1. 
#
#
# OWL-ViT is an open-vocabulary object detector. Given an image and one or multiple free-text queries, 
# it finds objects matching the queries in the image. Unlike traditional object detection models, 
# OWL-ViT is not trained on labeled object datasets and leverages multi-modal representations to 
# perform open-vocabulary detection.
# OWL-ViT uses CLIP with a ViT-like Transformer as its backbone to get multi-modal visual and text features. 
# To use CLIP for object detection, OWL-ViT removes the final token pooling layer of the vision model and 
# attaches a lightweight classification and box head to each transformer output token. 
# Open-vocabulary classification is enabled by replacing the fixed classification layer weights with the 
# class-name embeddings obtained from the text model. 
# The authors first train CLIP from scratch and fine-tune it end-to-end with the classification and box heads 
# on standard detection datasets using a bipartite matching loss. 
# One or multiple text queries per image can be used to perform zero-shot text-conditioned object detection.


# sidenote:
# **Bipartite Matching Loss:**
# The goal is to match predicted objects (from the model) 
# with ground-truth objects (from the labeled data).
# We want to find the best correspondence between predicted
# bounding boxes and their corresponding ground-truth boxes.
# Given a set of predicted bounding boxes and a set of 
# ground-truth bounding boxes, we perform bipartite matching.
# Each predicted box is associated with the closest ground-truth
# box based on some similarity metric (e.g., IoU - Intersection over Union).
# The Hungarian algorithm is commonly used to find this optimal matching.
# 
# Bipartite matching ensures that each predicted box corresponds to
# exactly one ground-truth box (and vice versa).
# By enforcing this one-to-one mapping, we avoid missing detections 
# or assigning multiple predictions to the same ground-truth object.
# In summary, bipartite matching loss aligns predicted and ground-truth bounding boxes, 
# improving the accuracy of object detection models.
#
#
# since we are planning to use pytorch (the original repo is in jax/tensorflow), 
# we use hugging face transformers that implemented it already! 
import transformers
# we also need opencv, and bunch of other libraries as well that we might use
import torch

import sys,os
import requests
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

print(f'{sys.version=}')
print(f'{torch.__version__=}')
print(f'{transformers.__version__=}')#transformers 4.37.2
# sys.version='3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0]'
# torch.__version__='2.2.0+cu118'
# transformers.__version__='4.42.3'
# 
# Load pre-trained model and processor
# Let's first apply the image preprocessing and tokenize the text queries using OwlViTProcessor. 
# The processor will resize the image(s), scale it between [0-1] range and normalize it across 
# the channels using the mean and standard deviation specified in the original codebase.
# Text queries are tokenized using a CLIP tokenizer and stacked to output tensors of shape
# [batch_size * num_max_text_queries, sequence_length].
# if you are inputting more than one set of (image, text prompt/s), num_max_text_queries is 
# the maximum number of text queries per image across the batch. 
# Input samples with fewer text queries are padded.
# from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Preprocess input image and text queries
# Let's use the image of astronaut Eileen Collins to test OWL-ViT. It's part of the NASA Great Images dataset.
# You can use one or multiple text prompts per image to search for the target object(s). 
# Let's start with a simple example where we search for multiple objects in a single image.
def get_image(url):
    if 'http' in url:
        return Image.open(requests.get(url=url,stream=True).raw)
    else:
        return Image.open(url).convert('RGB')
    
img_url = 'https://images.thalia.media/-/BF2000-2000/c745c2eb05804daabf8d3886d1ce9791/jujutsu-kaisen-the-official-anime-guide-season-1-taschenbuch-gege-akutami-englisch.jpeg'
img_url = 'https://upload.wikimedia.org/wikipedia/commons/8/88/Commander_Eileen_Collins_-_GPN-2000-001177.jpg'
# img_url = '/media/hossein/CodingStuffs/CodingStuff/Projects/OpenCVProjects/img/dice2.jpg'
# img_url = '/media/hossein/CodingStuffs/CodingStuff/Projects/OpenCVProjects/img/coffee.png'
img_url = 'https://mymodernmet.com/wp/wp-content/uploads/2019/09/100k-ai-faces-1.jpg'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device=}')
# to instruct our model to look for a specific object, we need to sent it in text format. 
# and to feed it to our model, we need to tokenize it. 
# so lets grab the tokenizer used for owlvit model
# in hf transformer, this tokenizer is wrapped in OWLViTProcessor class,
# which also wraps OwlViTImageProcessor aside from the CLIPTokenizer/CLIPTokenizerFast 
# that we are after. we can access each using "image_processor", "tokenizer" attributes. 
# since this processor interits both the image processor and tokenizer functionalities
# this makes our life much easier!
processor = transformers.OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
# what we want to seach for in our image
# note taht simply using 'face' will not give us anything! we have to specifically write human face!!
# note that if we have several images, we can create text-queries for each image by creating separate 
# nested list. like 
# text_queries = [["human face", "rocket", "nasa badge", "star-spangled banner"], ["coffee mug", "spoon", "plate"]]
# and we would have a list of images instead of a single image obviously!
text_queries = ["human face"]
# the image of interest in which we will be searching!
print(f'downloading image...')
image = get_image(img_url)
# image = Image.open('./nasa.png').convert('RGB')
# lets grab the result which is a dictionary of input_ids, attention_mask and pixel_values (image batch)
results = processor(text=text_queries, images=image, return_tensors='pt').to(device)
# Print input names and shapes
for key, val in results.items():
    print(f"{key}: {val.shape}")

# Forward pass
# Now we can pass the inputs to our OWL-ViT model to get object detection predictions.
# OwlViTForObjectDetection model outputs the prediction logits, boundary boxes and class embeddings,
# along with the image and text embeddings outputted by the OwlViTModel, which is the CLIP backbone.
model = transformers.OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
# model = transformers.Owlv2ForObjectDetection()
model = model.to(device)
# remember to model in evaluation mode
model.eval()
with torch.no_grad():
    outputs = model(**results)

for k, val in outputs.items():
    if k not in {"text_model_output", "vision_model_output"}:
        print(f"{k}: shape of {val.shape}")

print("\nText model outputs")
for k, val in outputs.text_model_output.items():
    print(f"{k}: shape of {val.shape}")

print("\nVision model outputs")
for k, val in outputs.vision_model_output.items():
    print(f"{k}: shape of {val.shape}") 

# logits: shape of torch.Size([1, 576, 4])
# pred_boxes: shape of torch.Size([1, 576, 4])
# text_embeds: shape of torch.Size([1, 4, 512])
# image_embeds: shape of torch.Size([1, 24, 24, 768])
# class_embeds: shape of torch.Size([1, 576, 512])

# Text model outputs
# last_hidden_state: shape of torch.Size([4, 16, 512])
# pooler_output: shape of torch.Size([4, 512])

# Vision model outputs
# last_hidden_state: shape of torch.Size([1, 577, 768])
# pooler_output: shape of torch.Size([1, 768])

# Draw predictions on image
# Let's draw the predictions / found objects on the input image. 
# Remember the found objects correspond to the input text queries.
#resize image we can use huggingface transformers ImageFeatureExtractionMixin
# to resize and preprocess the image 
# from transformers.image_utils import ImageFeatureExtractionMixin
# mixin = ImageFeatureExtractionMixin()
# image_size = model.config.vision_config.image_size
# image = mixin.resize(image, image_size)
# input_image = np.asarray(image).astype(np.float32) / 255.0
# or simply use opencv to resize the input image properly
img_size = model.config.vision_config.image_size
print(f'{img_size=}')
image = cv2.resize(np.asanyarray(image), dsize=(img_size,img_size))
# normalize image to 0-1 range
input_image = image/255

# Threshold to eliminate low probability predictions
# the accuracy of detection may not be high, in fact, in many cases, its pretty low
# and its not sufficient for some tasks. it works better for some usecases than others
# and its directly corollated with how good the underlying model is trained on similar concepts
probability_threshold = 0.1

# # Get prediction logits
logits = torch.max(outputs["logits"][0], dim=-1)
probs = torch.sigmoid(logits.values).cpu().detach().numpy()

# Get prediction labels and boundary boxes
labels = logits.indices.cpu().detach().numpy()
bboxes = outputs["pred_boxes"][0].cpu().detach().numpy()

# draw the bbox 
for prob,bbox,label in zip(probs,bboxes,labels):
    if prob < probability_threshold:
        continue
    # the bounding box is made of cx,cy and width and height normalized in 0-1 range
    # so in order to get the actual coordinates, we need to multiply them by the image
    # width and height
    cx,cy,w,h = bbox
    cx,cy = int(cx*input_image.shape[0]),int(cy*input_image.shape[1])
    w,h = int(w*input_image.shape[0]),int(h*input_image.shape[1])
    x,y = cx-w//2,cy-h//2
    text = text_queries[label]
    print(f'{prob=} {bbox=} {label=} {text=}')
    cv2.rectangle(input_image, (x,y), (x+w,y+h),color=(0,0,255),thickness=2)
    cv2.putText(input_image,text,org=(x,y),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(0,255,0),thickness=2)

cv2.imshow('input_image-bbox',input_image[...,::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()

# heres another implmentation using matplot lib(huggingface docs)
# def plot_predictions(input_image, text_queries, probs, bboxes, labels):
#     # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#     # ax.imshow(input_image, extent=(0, 1, 1, 0))
#     # ax.set_axis_off()
#     # setting extend is crucial for the image to align with the plot below so the bboxes 
#     # actuall are drawn properly.
#     # more specifically, the extent parameter specifies the spatial extent of our input image. 
#     # The four values (0, 1, 1, 0) correspond to the left, right, bottom, and top boundaries of
#     # the displayed image, respectively. 
#     # Essentially, it defines the region in the plot where the image will be shown. 
#     # The (0, 1) range represents the horizontal axis (x-axis), and the (1, 0) range 
#     # represents the vertical axis (y-axis). So, the image will be displayed within 
#     # the unit square (from (0, 0) to (1, 1)).
#     # here, this code snippet will display it within the specified extent on the plot.
#     # (note the 'plot' if we didnt have a plot afterwards, which draws the bbox, we wouldnt be needing this!)
#     plt.imshow(input_image, extent=(0, 1, 1, 0))

#     for prob, bbox, label in zip(probs, bboxes, labels):
#       if prob < probability_threshold:
#         continue

#       cx, cy, w, h = bbox
#       # draw the bboxes around the faces
#       plt.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
#               [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
      
#       plt.text(cx - w / 2, cy + h / 2 + 0.015,
#           f"{text_queries[label]}: {prob:1.2f}",
#           ha="left",
#           va="top",
#           color="red",
#           # draws a bbox around the text!
#           bbox={
#               "facecolor": "white",
#               "edgecolor": "red",
#               "boxstyle": "square,pad=.3"
#           })
    
# plot_predictions(input_image, text_queries, probs, bboxes, labels)
# plt.show()

# Post-processing model predictions
# As we have just seen, the OWL-ViT outputs are normalized box coordinates in [cx, cy, w, h] format 
# which is based on the assumption that all input image sizes are fixed. 
# aside from what we did to convert them back to normal coordinates, We can use the OwlViTProcessor's 
# convenient post_process() method as well to convert the model outputs to a COCO API format 
# and retrieve rescaled coordinates (with respect to the original image sizes) in [x0, y0, x1, y1] format.
# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
# target_sizes = torch.Tensor([img.size[::-1] for img in image]).to(device)
# Convert outputs (bounding boxes and class logits) to COCO API
# results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

# Batch processing
# We can also pass in multiple sets of images and text queries to search for different 
# (or same) objects in different images. Let's download an image of a coffee mug to process alongside the astronaut image.

# For batch processing, we need to input text queries as a nested list to OwlViTProcessor 
# and images as lists of (PIL images or PyTorch tensors or NumPy arrays).

# # Download the coffee mug image
# image = skimage.data.coffee()
# image = Image.fromarray(np.uint8(image)).convert("RGB")
# image

# # Preprocessing
# images = [skimage.data.astronaut(), skimage.data.coffee()]
# images = [Image.fromarray(np.uint8(img)).convert("RGB") for img in images]

# # Nexted list of text queries to search each image for
# text_queries = [["human face", "rocket", "nasa badge", "star-spangled banner"], ["coffee mug", "spoon", "plate"]]

# # Process image and text inputs
# inputs = processor(text=text_queries, images=images, return_tensors="pt").to(device)

# # Print input names and shapes
# for key, val in inputs.items():
#     print(f"{key}: {val.shape}")

# input_ids: torch.Size([8, 16])
# attention_mask: torch.Size([8, 16])
# pixel_values: torch.Size([2, 3, 768, 768])

# Note: Notice the size of the input_ids and attention_mask is [batch_size * num_max_text_queries,
# max_length]. Max_length is set to 16 for all OWL-ViT models.

# # Get predictions
# with torch.no_grad():
#   outputs = model(**inputs)

# for k, val in outputs.items():
#     if k not in {"text_model_output", "vision_model_output"}:
#         print(f"{k}: shape of {val.shape}")
        
# print("\nText model outputs")
# for k, val in outputs.text_model_output.items():
#     print(f"{k}: shape of {val.shape}")

# print("\nVision model outputs")
# for k, val in outputs.vision_model_output.items():
#     print(f"{k}: shape of {val.shape}") 

# logits: shape of torch.Size([2, 576, 4])
# pred_boxes: shape of torch.Size([2, 576, 4])
# text_embeds: shape of torch.Size([2, 4, 512])
# image_embeds: shape of torch.Size([2, 24, 24, 768])
# class_embeds: shape of torch.Size([2, 576, 512])

# Text model outputs
# last_hidden_state: shape of torch.Size([8, 16, 512])
# pooler_output: shape of torch.Size([8, 512])

# Vision model outputs
# last_hidden_state: shape of torch.Size([2, 577, 768])
# pooler_output: shape of torch.Size([2, 768])

# # Let's plot the predictions for the second image
# image_idx = 1
# image_size = model.config.vision_config.image_size
# image = mixin.resize(images[image_idx], image_size)
# input_image = np.asarray(image).astype(np.float32) / 255.0

# # Threshold to eliminate low probability predictions
# score_threshold = 0.1

# # Get prediction logits
# logits = torch.max(outputs["logits"][image_idx], dim=-1)
# scores = torch.sigmoid(logits.values).cpu().detach().numpy()

# # Get prediction labels and boundary boxes
# labels = logits.indices.cpu().detach().numpy()
# boxes = outputs["pred_boxes"][image_idx].cpu().detach().numpy()

# plot_predictions(input_image, text_queries[image_idx], scores, boxes, labels)

# Bonus: one-shot / image-guided object detection

# what we have been doing so far is called zero shot detection with text inputs(also may be called text-guided detection), 
# which means without training the underlying model for a specific task, we simply use it for said task!
# here using text prompts. 
# We can also use the OwlViTForObjectDetection for 1 shot detection. 
# using its image_guided_detection() method, we can query an input image with a query/example image 
# and detect similar objects.
# Everything is the nearly same, this time instead of a text prompt, we simply pass in a query images 
# to the processor to get the query_pixel_values which is basically the preprocessed image ready to be 
# fed to the model. 
# 
# Note though, unlike text input, OwlViTProcessor expects one query image per target(input) image 
# we'd like to query for similar objects. 
# We will also see that the output and post-processing of one-shot object detection is 
# very similar to the zero-shot / text-guided detection.

# Let's try this out by querying an image with cats with another random cat image. 
# For this part of the demo, we will perform image-guided object detection, 
# post-process the results and display the predicted boundary boxes on the original input image using OpenCV.

# Input image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = get_image(url)
target_sizes = torch.Tensor([image.size[::-1]])
print(f'{target_sizes=}')
# Query image - the image we use to search for its lookalike in the input image
query_url = "http://images.cocodataset.org/val2017/000000058111.jpg"
query_image = get_image(query_url)

image = np.asanyarray(image)
query_image = np.asanyarray(query_image)

inputs = processor(images=image, query_images=query_image, return_tensors='pt').to(device)
print(f'inputs:')
for key,value in inputs.items():
    print(f'{key=} {value.shape=}')

# query_pixel_values: torch.Size([1, 3, 768, 768])
# pixel_values: torch.Size([1, 3, 768, 768])

with torch.no_grad():
    outputs = model.image_guided_detection(**inputs)

print(f'outputs:')
for k,v in outputs.items():
    print(f'{k=}')

print("\nVision model outputs")
for k, val in outputs.vision_model_output.items():
    print(f"{k}: shape of {val.shape}")

# logits: shape of torch.Size([1, 576, 1])
# image_embeds: shape of torch.Size([1, 24, 24, 768])
# query_image_embeds: shape of torch.Size([1, 24, 24, 768])
# target_pred_boxes: shape of torch.Size([1, 576, 4])
# query_pred_boxes: shape of torch.Size([1, 576, 4])
# class_embeds: shape of torch.Size([1, 576, 512])

# Vision model outputs
# last_hidden_state: shape of torch.Size([1, 577, 768])
# pooler_output: shape of torch.Size([1, 768])

img2 = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
outputs.logits = outputs.logits.cpu()
outputs.target_pred_boxes = outputs.target_pred_boxes.cpu()
# note we are passing target_sizes, so the bbox coordinates are adjusted accordingly
results = processor.post_process_image_guided_detection(outputs=outputs, threshold=0.6, nms_threshold=0.3, target_sizes=target_sizes)
boxes = results[0]["boxes"] 
scores = results[0]["scores"]

# Draw predicted bounding boxes
for box, score in zip(boxes, scores):
    box = [int(i) for i in box.tolist()]

    img2 = cv2.rectangle(img2, box[:2], box[2:], (255,0,0), 5)
    if box[3] + 25 > 768:
        y = box[3] - 10
    else:
        y = box[3] + 25 

plt.imshow(img2[:,:,::-1])
plt.show()

# we can also select a part of an image to look for other similar things like this
# img_url='https://static.vecteezy.com/system/resources/previews/037/998/433/large_2x/ai-generated-two-great-tit-birds-parus-major-drinking-water-from-a-fountain-photo.jpg'
img_url='./board_src.jpg'

img = get_image(img_url)
img = np.asanyarray(img)
# select a roi 
y,x,h,w = cv2.selectROI('select an object to be searched', img[...,::-1], False, False)
print(f'{x=}{y=}{w=}{h=}')
roi_img = img[x:x+w,y:y+h]
# cv2.imshow('roi',roi_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# now lets create the processor 
inputs = processor(images=img, query_images=roi_img,return_tensors='pt').to(device)

with torch.no_grad():
    outputs = model.image_guided_detection(**inputs)

for k,v in outputs.items():
    print(f'{k} {v.shape if hasattr(v,"shape") else "-"}')

outputs.logits = outputs.logits.cpu().detach()
outputs.target_pred_boxes = outputs.target_pred_boxes.cpu().detach()
# now lets call posprocess!
# since we have a batch of 1, we send our shape this way to match input
target_sizes = torch.Tensor([[*img.shape[:2]]])
print(f'{target_sizes=}')
results = processor.post_process_image_guided_detection(outputs, threshold=0.1, nms_threshold=0.6, target_sizes=target_sizes)
probs = results[0]["scores"]
bbox = results[0]["boxes"]

for prob,bbox in zip(probs, bbox):
    bbox = [int(i) for i in bbox.tolist()]
    print(f'{bbox=}')
    img = cv2.rectangle(img=img.copy(), pt1=bbox[:2],pt2=bbox[2:],color=(0,255,0),thickness=2)
    
cv2.imshow('image-bbox',img[...,::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()

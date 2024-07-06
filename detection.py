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

# The OWLv2 model (short for Open-World Localization) was proposed in Scaling Open-Vocabulary Object Detection paper in june 2023
# by Matthias Minderer, Alexey Gritsenko, Neil Houlsby. OWLv2, like OWL-ViT, is a zero-shot text-conditioned 
# object detection model that can be used to query an image with one or multiple text queries. 
# The v2 model uses a CLIP backbone with a ViT-L/14 Transformer architecture as an image encoder 
# and uses a masked self-attention Transformer as a text encoder. These encoders are trained to 
# maximize the similarity of (image, text) pairs via a contrastive loss. T
# he CLIP backbone is trained from scratch and fine-tuned together with the box and class prediction heads
# with an object detection objective.

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
# as for the weights, there are several variants with various patch sizes, and training schemes 
# (self-trained only, self-trained + fine-tuned, and an ensemble). The ensemble checkpoints perform best.
# the authors also released larger checkpoints, which have even better performance:
# https://huggingface.co/google/owlvit-base-patch16
# https://huggingface.co/google/owlvit-large-patch14
# v2 variants
# https://huggingface.co/google/owlv2-large-patch14
# https://huggingface.co/google/owlv2-large-patch14-ensemble
# https://huggingface.co/google/owlv2-large-patch14-finetuned
# https://huggingface.co/google/owlv2-base-patch16
# https://huggingface.co/google/owlv2-base-patch16-finetuned
# https://huggingface.co/google/owlv2-base-patch16-ensemble

#TODO: v2 needs adjustmet for bbox 
# see https://github.com/NielsRogge/Transformers-Tutorials/blob/master/OWLv2/Zero_and_one_shot_object_detection_with_OWLv2.ipynb
v1 = 0
model_name = "google/owlvit-base-patch32" if v1 else "google/owlv2-base-patch16"
# model_name = "google/owlvit-large-patch14" if v1 else "google/owlv2-large-patch14"

# to get this to work, we need a processor to take care of our input 
# and the actual model for detection (to do the forward pass)
processor = transformers.OwlViTProcessor.from_pretrained(model_name) if v1 else transformers.Owlv2Processor.from_pretrained(model_name)
# OwlViTForObjectDetection model outputs the prediction logits, boundary boxes and class embeddings,
# along with the image and text embeddings outputted by the OwlViTModel, which is the CLIP backbone.
model = transformers.OwlViTForObjectDetection.from_pretrained(model_name) if v1 else transformers.Owlv2ForObjectDetection.from_pretrained(model_name)
model = model.to(device)
# remember to model in evaluation mode
model.eval()
# lets grab an image
img_url = 'https://images.thalia.media/-/BF2000-2000/c745c2eb05804daabf8d3886d1ce9791/jujutsu-kaisen-the-official-anime-guide-season-1-taschenbuch-gege-akutami-englisch.jpeg'
img_url = 'https://upload.wikimedia.org/wikipedia/commons/8/88/Commander_Eileen_Collins_-_GPN-2000-001177.jpg'
# img_url = '/media/hossein/CodingStuffs/CodingStuff/Projects/OpenCVProjects/img/dice2.jpg'
# img_url = '/media/hossein/CodingStuffs/CodingStuff/Projects/OpenCVProjects/img/coffee.png'
img_url = 'https://mymodernmet.com/wp/wp-content/uploads/2019/09/100k-ai-faces-1.jpg'

# what we want to seach for in our image? we simply specify that as a text prompt.
# we can have several text prompts per image. we feed them as a list. 
# obviously if we have several images, we create text-queries for each image by creating separate 
# nested list. like text_queries = [["human face", "rocket", "nasa badge", "star-spangled banner"], ["coffee mug", "spoon", "plate"]]
# and we would have a list of images instead of a single image obviously!
# note that simply using 'face' will not give us anything! we have to specifically write human face!
text_queries = ["human face"]
# the image of interest in which we will be searching!
print(f'downloading image...')
image = get_image(img_url)
# to resize image we can use huggingface transformers
# ImageFeatureExtractionMixin to resize and preprocess the image:
# 
# from transformers.image_utils import ImageFeatureExtractionMixin
# image_size = model.config.vision_config.image_size
# mixin = ImageFeatureExtractionMixin()
# image = mixin.resize(image, image_size)
# 
# or simply use opencv to resize the input image properly
img_size = model.config.vision_config.image_size
image = cv2.resize(np.asanyarray(image), dsize=(img_size,img_size))
print(f'{img_size=}')
# lets grab the prepared inputs which is a dictionary of input_ids, attention_mask and pixel_values (image batch)
inputs = processor(text=text_queries, images=image, return_tensors='pt').to(device)
# lets see what fields we have in our inputs here
for key, val in inputs.items():
    print(f"{key}: {val.shape}")

# good now lets do an actual forward pass and grab our predictions 
# note that the found objects here correspond to the our text prompts/queries
with torch.no_grad():
    outputs = model(**inputs)

# lets see what our output contains: 
for k, val in outputs.items():
    if k not in {"text_model_output", "vision_model_output"}:
        print(f"{k}: shape of {val.shape}")

print("\nText model outputs")
for k, val in outputs.text_model_output.items():
    print(f"{k}: shape of {val.shape}")

print("\nVision model outputs")
for k, val in outputs.vision_model_output.items():
    print(f"{k}: shape of {val.shape}") 

# we resize our image so the bbox properly fits it
# to resize image we can use huggingface transformers ImageFeatureExtractionMixin
# to resize and preprocess the image 
# from transformers.image_utils import ImageFeatureExtractionMixin
# mixin = ImageFeatureExtractionMixin()
# image_size = model.config.vision_config.image_size
# image = mixin.resize(image, image_size)
# or simply use opencv to resize the input image properly
# the catch however is, for owlv2, the authors used the preprocessed
# image, that is the image thats padded and resized, rather than the original one 
# to visualize the bounding boxes. 
# Therefore we need to take the preprocess image created by the processor and "unnormalize" it
# This gives us the preprocessed image, minus the normalization.(i.e. padded&resized image)
def denormalize_owlv2(inputs, return_as_PIL=False):
    from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

    # we can access the preprocess image by either:
    # inputs["pixel_values"] or directly accessing 
    # it as an attribute like: inputs.pixel_values
    pixel_values = inputs.pixel_values.squeeze().cpu().numpy()
    image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    image = (image * 255).astype(np.uint8)
    print(f'{image.shape=}')
    # move the channels from first dim to the last dim, note the copy() thats needed here
    # otherwise due to the transpose operation here, the array is not congiguous, we need a contigueous array
    # for cv2 to properly work. copy fixes that for us!
    image = np.moveaxis(image, 0, -1).copy()
    print(f'{image.shape=}')
    if return_as_PIL:
        image = Image.fromarray(image)
    return image

if v1:
    img_size = model.config.vision_config.image_size
    image = cv2.resize(np.asanyarray(image), dsize=(img_size,img_size))
    # image = np.asanyarray(image).copy()

else:
    image = denormalize_owlv2(inputs)

# Threshold to eliminate low probability predictions
# the accuracy of detection may not be high, in fact, in many cases, its pretty low
# and its not sufficient for some tasks. it works better for some usecases than others
# and its directly corollated with how good the underlying model is trained on similar concepts
# (this includes both parts of the model, the language and vision part need to be sufficiently trained
# otherwise, a term may not exist, or registered for a concept, so cant get the result you want or the vision
# model has not seen sth and therefore the outcome is undefiened)
probability_threshold = 0.05

# Get prediction logits
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
    cx,cy = int(cx*image.shape[0]),int(cy*image.shape[1])
    w,h = int(w*image.shape[0]),int(h*image.shape[1])
    x,y = cx-w//2,cy-h//2
    text = text_queries[label]
    print(f'{text=:<12} {prob=:.4f} {label=:^3} bbox={tuple(bbox)}')
    cv2.rectangle(image, (x,y), (x+w,y+h),color=(0,0,255),thickness=2)
    cv2.putText(image,text,org=(x,y+10),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(0,255,0),thickness=2)

cv2.imshow('image-bbox',image[...,::-1])
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

# post-processing model predictions
# As we have just seen, the OWL-ViT outputs are normalized box coordinates in [cx, cy, w, h] format 
# which is based on the assumption that all input image sizes are fixed. 
# aside from what we did to convert them back to normal coordinates, we can use the OwlViTProcessor's 
# convenient post_process() method as well to convert the model outputs to a COCO API format 
# and retrieve rescaled coordinates (with respect to the original image sizes) in [x0, y0, x1, y1] format
# like this:
# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
# note we have several images here:
# target_sizes = torch.Tensor([img.size[::-1] for img in images]).to(device)
# Convert outputs (bounding boxes and class logits) to COCO API
# results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

# Note: Notice the size of the input_ids and attention_mask is [batch_size * num_max_text_queries, max_length]. 
# Max_length is set to 16 for all OWL-ViT models.

# one-shot/image-guided object detection
# what we have been doing so far is called zero shot detection with text inputs(also may be called text-guided detection), 
# which means without training the underlying model for a specific task, we simply use it for said task!
# here using text prompts. 
# We can also use the OwlViTForObjectDetection for 1 shot detection. 
# using its image_guided_detection() method, we can query an input image with a query/example image 
# and detect similar objects.
# Everything is nearly same, this time around instead of a text prompt, we simply pass in a query image
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

# Query image - the image we use to search for its lookalike in the input image
query_url = "http://images.cocodataset.org/val2017/000000058111.jpg"
query_image = get_image(query_url)

image = np.asanyarray(image)
query_image = np.asanyarray(query_image)

# processor, applies the required preprocessings like resizing/etc as well
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

outputs.logits = outputs.logits.cpu()
outputs.target_pred_boxes = outputs.target_pred_boxes.cpu()

# before going for drawing predicted bounding boxes, lets use the proper image
image = image if v1 else denormalize_owlv2(inputs)
target_sizes = torch.Tensor([image.size[::-1]]) if v1 else torch.Tensor([(denormalize_owlv2(inputs).shape[:2])])
# target_sizes = torch.Tensor([image.size[::-1]]) if v1 else torch.Tensor([(image.shape[:2])])
print(f'{target_sizes=}')

# note we are passing target_sizes, so the bbox coordinates are adjusted accordingly
# sidenote2: 
# nms_threshold here needs a bit of explanation.(basically we usually dont expose it like this
# here nms threshold specifies how much overlapped boxes are allowed, i.e. if two or more boxes
# have this much overlap, nms wont be applied!otherwise it will be applied and will suppress non maximally boxes, retaining only the most probable one)
# heres the full explanation : 
# the nms_threshold parameter controls the IoU (Intersection over Union) threshold for
# non-maximum suppression of overlapping boxes during object detection. 
# When multiple bounding boxes overlap significantly, non-maximum suppression helps 
# retain only the most confident and relevant predictions. 
# 
# The nms_threshold determines how much overlap is acceptable before suppressing redundant boxes. 
# A lower value results in more aggressive suppression, while a higher value allows more overlapping
# boxes to be retained. 
# so in our example here, it’s set to 0.1, meaning that boxes with an IoU greater than 0.1 will be suppressed.
# Additionally, the target_sizes parameter as we have covered before, allows us to rescale predicted bounding boxes
# to our input image which was fed to the model.
inputs_list = processor.post_process_image_guided_detection(outputs=outputs, threshold=0.7, nms_threshold=0.1, target_sizes=target_sizes)
boxes = inputs_list[0]["boxes"] 
scores = inputs_list[0]["scores"]

for box, score in zip(boxes, scores):
    box = [int(i) for i in box.tolist()]
    image = cv2.rectangle(image, box[:2], box[2:], (255,0,0), 5)

plt.imshow(image)
plt.show()

# we can also select a part of an image to look for other similar things like this
# sidenote: for somereason it seems our version here doesnt work at all
# it could be becasue the roi_img is too small and the network cant identify any meaningful object
# by looking at that small image and thus it results in nonsensical detection. 
# if we used a larger image of something wed like, we may probably get better result-nope I didnt get any improvments
# seems the model is off! or has issues!
# TODO: fix this
# 
# img_url='https://static.vecteezy.com/system/resources/previews/037/998/433/large_2x/ai-generated-two-great-tit-birds-parus-major-drinking-water-from-a-fountain-photo.jpg'
img_url='./board_src.jpg'
# resitor image
# target_url = 'https://res.cloudinary.com/rsc/image/upload/bo_1.5px_solid_white,b_auto,c_pad,dpr_2,f_auto,h_399,q_auto,w_710/c_pad,h_399,w_710/R0131772-01?pgw=1'
# target_url = 'https://media.rs-online.com/image/upload/w_620,h_413,c_crop,c_pad,b_white,f_auto,q_auto/dpr_auto/v1529599145/Y1742636-01.jpg'
# target_url = 'https://media.rs-online.com/image/upload/w_620,h_413,c_crop,c_pad,b_white,f_auto,q_auto/dpr_auto/v1482295262/F2141951-01.jpg'
# ic
# target_url = 'https://www.semiconductorforu.com/wp-content/uploads/2017/06/IC.jpg'
image = get_image(img_url)
# select a roi 
y,x,h,w = cv2.selectROI('select an object to be searched', np.asanyarray(image)[...,::-1], False, False)
print(f'{x=}{y=}{w=}{h=}')
roi_img = np.asanyarray(image)[x:x+w,y:y+h]
roi_img = Image.fromarray(roi_img, mode="RGB")
# roi_img.save("roi.jpg")
# roi_img = get_image(target_url)


# cv2.imshow('roi',roi_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# now lets create the processor 
inputs = processor(images=image, query_images=roi_img,return_tensors='pt').to(device)

with torch.no_grad():
    outputs = model.image_guided_detection(**inputs)

for k,v in outputs.items():
    print(f'{k} {v.shape if hasattr(v,"shape") else "-"}')

outputs.logits = outputs.logits.cpu().detach()
outputs.target_pred_boxes = outputs.target_pred_boxes.cpu().detach()
# 
image = image if v1 else denormalize_owlv2(inputs)
# now lets call posprocess!
# since we have a batch of 1, we send our shape this way to match input
target_sizes = torch.Tensor([[*image.shape[:2]]])
print(f'{target_sizes=}')
result_list = processor.post_process_image_guided_detection(outputs, threshold=0.9, nms_threshold=0.01, target_sizes=target_sizes)
probs = result_list[0]["scores"]
bbox = result_list[0]["boxes"]

# image = np.asarray(image).copy()
for prob,bbox in zip(probs, bbox):
    bbox = [int(i) for i in bbox.tolist()]
    print(f'{bbox=}')
    image = cv2.rectangle(img=image, pt1=bbox[:2],pt2=bbox[2:],color=(0,255,0),thickness=2)
    
cv2.imshow('image-bbox',image[...,::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()


# there are other works like this lik : 
# https://huggingface.co/IDEA-Research/grounding-dino-base
# https://huggingface.co/IDEA-Research/grounding-dino-tiny


# TODO: explain how the underlying thing works
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/OWLv2/Zero_and_one_shot_object_detection_with_OWLv2.ipynb
#
# now lets see how the image guided detection works under the hood(we use the v2 model): 
# the basic idea is very simple, using a text prompt or text query as some people like to call it, 
# we look for and grab all the objects in the image that satisfy said prompt. 
# for example, if we used a prompt like, cat, we get all the image patches the network identifies as cat
# 
# 
# 
# We do this by getting the objectness logits, and simply get the box with the highest objectness value
# (i.e. the box that has the highest probability of containing an object).
# Below, we show the top 3 predictions on the source image so that 
# the user can select one to use as a query (we select the cat here).
# Note that we cannot directly embed a whole image (as we need an embedding of one particular patch token).
# 

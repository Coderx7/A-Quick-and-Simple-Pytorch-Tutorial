#%% in the name of God the most compassionate the most merciful 
# geremey howard part 9,9a and 9b and 10 watch them first to get the initial idea. 
# then feel free to watch any of the following:
# 
# https://huggingface.co/docs/diffusers/en/tutorials/basic_training
# 
# Stable Diffusion - What, Why, How? 
# https://www.youtube.com/watch?v=ltLNYA3lWAQ
#
#  CS 198-126: Lecture 12 - Diffusion Models 
# https://www.youtube.com/watch?v=687zEGODmHA

# https://www.youtube.com/watch?v=iNeauvp3JU0
# 
#  Tutorial on Denoising Diffusion-based Generative Modeling: Foundations and Applications 
# a very good introduction
# https://www.youtube.com/watch?v=cS6JQpEY9cs
# 
# Diffusion Models | Paper Explanation | Math Explained 
# https://www.youtube.com/watch?v=HoKDTa5jHvg
# 
# 
# Denoising Diffusion Probabilistic Models | DDPM Explained 
# https://www.youtube.com/watch?v=H45lF4sUgiE
# 
#  Diffusion Models | PyTorch Implementation 
# https://www.youtube.com/watch?v=TBCRlnwJtZU
# 
#  HuggingFace Diffusion Model Class, Unit 1 (casual notebook walkthough) 
# https://www.youtube.com/watch?v=09o5cv6u76c
# 
#  Diffusion Models Explained : From DDPM to Stable Diffusion 
# https://www.youtube.com/watch?v=hVk7Py1c24Q
# 
#  🤗 Hugging Face just released *Diffusers* - for models like DALL-E 2 and Imagen! 
# https://www.youtube.com/watch?v=UzkdOg7wWmI
# 
#  Lecture 9 - Imagen 
# https://www.youtube.com/watch?v=4GMYlR0OCDI
#
#  Diffusion and Score-Based Generative Models 
#https://www.youtube.com/watch?v=wMmqCMwuM2Q

# Practical Deep Learning 2022 Part 2
# https://www.youtube.com/playlist?list=PLfYUBJiXbdtRUvTUYpLdfHHp9a58nWVXP
#
# Lesson 9B - the math of diffusion 
# https://www.youtube.com/watch?v=mYpjmM7O-30
#
# Jeffrey Fessler - An Introduction to Score Based Generative Models 
#https://www.youtube.com/watch?v=gtsmZx_quaI
# 
# Whats Inductive Bias?
# Inductive bias, also known as learning bias, is the set of assumptions that a 
# learning algorithm uses to predict outputs for inputs it has not encountered¹.
# It's what makes an algorithm learn one pattern instead of another¹. 
# For example, in machine learning, one might assume that the target 
# function should be a constant 1 or constant 0⁵. These assumptions, or biases,
# are necessary for the learning algorithm to generalize from training data to 
# unseen data².

# Recently, there's been a lot of discussion about inductive bias due to its role
# in the performance and generalization of machine learning models⁴⁵. 
# For instance, a recent study explored the limits of the hypothesis "less 
# inductive bias is better", popularized due to transformers eclipsing 
# convolutional models⁴. The study used multi-layer perceptrons (MLPs), 
# which lack any vision-specific inductive bias, as a test bed⁴. The results 
# showed that the performance of MLPs drastically improves with scale, 
# highlighting that lack of inductive bias can indeed be compensated⁴.
# Another reason for the increased interest in inductive bias is the growing 
# gap between theory and practice in machine learning⁴. Theoretical works often 
# focus on simple models like MLPs due to their mathematical simplicity⁴.
# However, these models may not reflect the empirical advances exhibited by 
# practical models, leading to discussions about the role of inductive bias in 
# bridging this gap⁴.
# In summary, inductive bias is a crucial aspect of machine learning algorithms
# that allows them to make predictions on unseen data. The recent discussions 
# around it are largely due to its impact on the performance of machine learning
# models and the role it plays in the theoretical understanding of these models⁴⁵.
#
# Source: Conversation with Bing, 1/28/2024
# (1) Inductive bias - Wikipedia. https://en.wikipedia.org/wiki/Inductive_bias.
# (2) What is inductive bias? – Towards AI. https://towardsai.net/p/artificial-intelligence/what-is-inductive-bias.
# (3) What Is Inductive Bias in Machine Learning? - Baeldung. https://www.baeldung.com/cs/ml-inductive-bias.
# (4) Scaling MLPs: A Tale of Inductive Bias - arXiv.org. https://arxiv.org/pdf/2306.13575.pdf.
# (5) 如何理解Inductive bias？ - 知乎. https://www.zhihu.com/question/264264203.
# (6) Learning Inductive Biases with Simple Neural Networks - arXiv.org. https://arxiv.org/pdf/1802.02745v1.
# (7) arXiv:2110.10090v2 [cs.LG] 24 Jun 2022. https://arxiv.org/pdf/2110.10090.pdf.
# (8) Inductive Biases for Deep Learning of Higher-Level Cognition. https://arxiv.org/abs/2011.15091.
# (9) undefined. https://github.com/gregorbachmann/scaling_mlps.
# 
#
# lets import our modules that we may need 
import os
import sys
import time
import random
import numpy as np
# import urllib
import requests
from pathlib import Path
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import PIL 
import PIL.Image as Image 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tfms

import transformers as trans
import evaluate 
import diffusers as dfs
#
# in the first step we are going to see how we can use huggingface diffusers library/module to 
# create/finetune/train diffusion models and then go on to implement  
# 
# 
from ipywidgets import interact
# sidenote: 
# ipywidgets is a Python library that provides interactive widgets for the Jupyter notebook. 
# The interact function from ipywidgets is a particularly useful tool that automatically creates
# user interface (UI) controls for exploring code and data interactively 
# 
# we can use interact as a decorator over our function and simply have a gui to play with its arguments
# sth like this for example 
# @interact (a=1,b=0.5,c=0.8)
# def plot_sth(a,b,c):
#    pass
# or use them like this 
# def f(m, b):
#     plt.figure(2)
#     x = np.linspace(-10, 10, num=1000)
#     plt.plot(x, m * x + b)
#     plt.ylim(-5, 5)
#     plt.show()
# interact(f, m=(-2.0, 2.0), b=(-3, 3, 0.5))
# 
# First we see how we can use diffusion modles hosted on huggingface and how they work
# and learn to do stuff such as lora, textinversion, dreamboth, etc imagetoimage,etc
# and then we we got a good grasp on the hugging face transformers/diffusers libraries
# we go on and try to train one model ourseleves from scratch
#
# this guide is inspired by official huggingface diffuser blog post and 
# geremy howards 2022 course -fastai #9 and 10
# 
# side note these links are intersting see them when you finished this once
# refs : 
# https://huggingface.co/blog/stable_diffusion
# https://huggingface.co/blog/annotated-diffusion
# https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
# http://yang-song.net/blog/2021/score/
#
# https://www.youtube.com/watch?v=0_BBRNYInx8&list=PLfYUBJiXbdtRUvTUYpLdfHHp9a58nWVXP&index=19&t=6s
# https://github.com/fastai/diffusion-nbs 
# refs : https://github.com/fastai/diffusion-nbs/pull/15 
# https://github.com/fastai/diffusion-nbs/issues
#
# We start off with importing StableDiffusionPipeline) which allows us to easily load a stablediffusion
# model and use it 
# but before it what is Stable Diffusion? 
# Simply put, Stable Diffusion is a text-to-image latent diffusion model created by the researchers and
# engineers from CompVis, Stability AI and LAION. It is trained on 512x512 images from a subset of the 
# LAION-5B database. LAION-5B is the largest, freely accessible multi-modal dataset that currently exists
# as Im writing this Feb 2024.
# 
from diffusers import StableDiffusionPipeline, DiffusionPipeline,  AutoPipelineForText2Image
#
# before we start lets first print the module's version we are using for reference
#
modules = [np, torch, torchvision, dfs, trans, matplotlib, PIL]
print(f'{"Python":<12}: {sys.version}')
for module in modules:
    print(f'{module.__name__:<12}: {module.__version__}')
# which for me at the time of writting this is as follows: 
# Python      : 3.11.4 (main, Jul  5 2023, 14:15:25) [GCC 11.2.0]
# numpy       : 1.24.3
# torch       : 2.2.0+cu118
# torchvision : 0.17.0+cu118
# diffusers   : 0.26.2
# transformers: 4.37.2
# matplotlib  : 3.7.1
# PIL         : 9.4.0
#
# refs good to read: https://blog.segmind.com/the-a-z-of-stable-diffusion-essential-concepts-and-terms-demystified/
# https://huggingface.co/docs/diffusers/main/en/using-diffusers/write_own_pipeline
#
# LoRA vs Dreambooth vs Textual Inversion vs Hypernetworks 
# https://www.youtube.com/watch?v=dVjMiJsuR5o
#
#
# now lets get back to our main objective. 
# what is stable diffusion again? 
# Stable Diffusion is a text-to-image latent diffusion model capable of generating photo-realistic
# images given any text input.
#
# we are going to use CompVis checkpoint files v1-4 and runway v1.5 (the 1.5 being massively popular!)
# although there are literally thousands of models.
# both hosted on huggingface hub, there are also other website such as civitai.com that host a plethora of great models.
#  
# For now we stick to hf hub and use the official stable diffusion model v1 
# (this is kind of old, as right now the XLv1.0 is the latest version (ignoring the finetuned variants on 
# websites such as civitai.com )which came out in july 2023 and offers images with resolution up to 1024x1024
# The v1 was released back in Dec 2021, 
# The v2 was released almost a year later in Nov 2022, followed by the v2.1 in Dec of the same year)
# Heres a list of trending models on hF hub: https://huggingface.co/models?pipeline_tag=text-to-image&sort=trending
# you can choose any models you like and start playing experimenting with them. 
#
# from https://stability.ai/news/stable-diffusion-v2-release:  
# The dynamic team of Robin Rombach (Stability AI) and Patrick Esser (Runway ML) from the CompVis Group
# at LMU Munich, headed by Prof. Dr. Björn Ommer, led the original Stable Diffusion V1 release. 
# They built on their prior work in the lab with Latent Diffusion Models and got critical support from 
# LAION and Eleuther AI. 
# The Stable Diffusion 2.0 release includes robust text-to-image models trained using a brand new text
# encoder (OpenCLIP), developed by LAION with support from Stability AI, which greatly improves the 
# quality of the generated images compared to earlier V1 releases. 
# The text-to-image models in this release can generate images with default resolutions of 512x512 pixels
# and 768x768 pixels. 
# These models are trained on an aesthetic subset of the LAION-5B dataset created by the DeepFloyd team 
# at Stability AI, which is then further filtered to remove adult content using LAION’s NSFW filter.
# (in practice the v1 model performed better especially for finetuned models and the v1.5 was especially 
# popular in 2022/2023)
# The latest stable diffusion model as of July 2023 is Stable Diffusion XL 1.0 (SDXL)1. 
# This next-generation open weights AI image synthesis model can generate novel images from text 
# descriptions and produces more detail and higher-resolution imagery than previous versions of 
# Stable Diffusion.
# Before SDXL, there were several versions of Stable Diffusion models. 
# For instance, Stable Diffusion 2.1 was released with two variants: one with a resolution of 
# 768x768 pixels and another with a resolution of 512x512 pixels. 
# Both models were based on the same number of parameters and architecture as 2.0 and fine-tuned 
# on 2.0.
# There was also a depth-guided stable diffusion model called depth2img, which extends the previous 
# image-to-image feature from V1 with brand-new possibilities for creative applications. 
# Depth2img infers the depth of an input image (using an existing model) and then generates new 
# images using both the text and depth information. 
# 
# The Stable-Diffusion-v1-4 checkpoint was initialized with the weights of the Stable-Diffusion-v1-2 
# checkpoint and subsequently fine-tuned on 225k steps at resolution 512x512 on "laion-aesthetics v2 5+"
# and 10% dropping of the text-conditioning to improve classifier-free guidance sampling.
# 
# sidenote - introductory notes: 
# I copied the following descriptions from blog.segmind.com
# These should provide a background for what we are going to be dealing with in this session.
#
# Stable Diffusion Versions
# Stable Diffusion (SD) models have evolved through various versions, each offering improvements
# and new features over its predecessors.
# 
# Stable Diffusion 1.5
# Released in the middle of 2022, the 1.5 model feature a resolution of 512x512 with 860 million parameters.
# It relies on OpenAI’s CLIP ViT-L/14 for interpreting prompts and is trained on the LAION 5B dataset. 
# SD 1.5 is known for being beginner-friendly and excels in creating portraits. However, 
# it tends to struggle with longer prompts and is limited by their lower resolution.
# 
# Stable Diffusion 2.1
# The SD 2.1 model was  introduced towards the end of 2022. It offer's an improved resolution of 768x768 and
# with 860 million parameters. The SD 2.1 use's LAION’s OpenCLIP-ViT/H for prompt interpretation and require 
# more detailed negative prompts. 
# It was trained on the LAION 5B dataset, supplemented with the LAION-NSFW classifier. 
# Its's strengths include handling shorter prompts more effectively and producing images with richer colors. 
# However, SD 2.1 genrates images of medium-level resolution.
# 
# Stable Diffusion XL (SDXL 1.0 )
# Launched in July 2023, the SDXL 1.0 represents a significant leap with a resolution of 1024x1024 and a 
# massive 3.5 billion parameters. This model utilizes both OpenCLIP-ViT/G and CLIP-ViT/L for a more nuanced
# inference of prompts. The training data specifics are not mentioned. The SDXL 1.0 is known for its ability
# to work with shorter prompts and deliver high-resolution images. However, it is resource-intensive and 
# requires a GPU, making it less accessible for users with limited hardware capabilities.(my update: in 2024 its much better as there have been lots of optimizations!)
# 
# Stable Diffusion Model Types
# Stable Diffusion models are broadly two types: txt2img & Img2Img. 
# These model types utilize the underlying principles of diffusion models 
# to generate or modify images, but they serve different purposes. 
# Txt2Img is more about creating entirely new images based on text descriptions, 
# while Img2Img is focused on transforming existing images based on additional 
# inputs or style guidelines.
# 
# Txt2Img (Text-to-Image)
# Txt2Img Stable Diffusion models generates images from textual descriptions. The user provides 
# a text prompt, and the model interprets this prompt to create a corresponding image.
# 
# Img2Img (Image-to-Image)
# The Img2Img Stable Diffusion models, on the other hand, starts with an existing image and modifies 
# or transforms it based on additional input. This could involve style transfer, where the artistic 
# style of one image is applied to another, or it could involve modifying certain aspects of the image
# according to specified parameters or prompts
# 
# Stable Diffusion API
# The Stable Diffusion API's allows developers to easily integrate the image generation capabilities of 
# the Stable Diffusion models into their own software applications. 
# It enables the automation of creating images based on text prompts, offering customizable options 
# like image resolution and style. The API's simplifies accessing Stable Diffusion Models for image 
# generation and is designed to handle multiple requests, making it scalable for various applications.
# 
# Stable Diffusion Model Formats
# Stable Diffusion models come in different formats, each serving a unique purpose in the image generation
# process. The two primary formats are Checkpoints and LoRAs. (me: add diffusers, coreml, onnx, pickletensor,safetensors or hypernetworks, controlnet, athestic gradients, etc)
# 
# Checkpoints
# These are the larger models responsible for the core image generation. 
# They have been trained on extensive datasets, enabling them to generate a wide variety of images independently.
# Think of them as the main engine in image creation, capable of understanding and interpreting a range of inputs
# to produce diverse visual outputs.
# 
# LoRA's
# These are smaller, more specialized models that work in conjunction with the checkpoints. 
# They are designed to modify or enhance the outputs of the checkpoints. Their primary role is 
# to introduce specific styles, characters, or concepts that the main checkpoint model might not
# know or might not render as effectively on its own. By integrating with a checkpoint, they allow
# for more nuanced and specialized image generation.
# 
# Distilled Stable Diffusion Models
# Distilled versions of the Stable Diffusion (SD) model, represent efforts to create more efficient,
# often smaller versions of the original SD model. These distilled models aim to retain as much of 
# the original model's capabilities as possible while being faster or more resource-efficient. 
# The creation of these distilled models is a response to the growing need for more accessible 
# and efficient AI models. They allow broader usage across various platforms and devices, especially
# where computational resources are a limiting factor.
# 
# SSD-1B
# Segmind Stable Diffusion-1B, a diffusion-based text-to-image model, is part of a Segmind's 
# distillation series, setting a new benchmark in image generation speed, especially for high-resolution
# images of 1024x1024 pixels. Compared to its predecessor, the SDXL 1.0 model, SSD-1B boasts significant 
# improvements: it's 50% smaller in size and 60% faster. This increase in speed and reduction in size is
# achieved with only a minimal compromise on image quality, maintaining a high standard close to that of 
# the SDXL 1.0.
# 
# Segmind Vega
# The Segmind Vega Model is a distilled version of the Stable Diffusion XL (SDXL), tailored for enhanced efficiency 
# and speed. This model is a remarkable 70% reduction in size compared to SDXL, making it much more compact. 
# Additionally, it boasts an impressive 100% increase in speed, effectively doubling the image generation speed.
# Despite these substantial improvements in size and speed, the Segmind Vega Model retains a high level of quality
# in text-to-image generation, demonstrating its capability to efficiently produce detailed and high-quality images.
# 
# Real-time Stable Diffusion Image Generation
# Stable Diffusion real-time models are designed to generate AI images at an exceptionally rapid pace, with the 
# capability to produce images almost as quickly as one can type in real-time. These models are optimized for 
# efficiency without compromising the quality of the generated images.
# 
# Segmind VegaRT
# The Segmind VegaRT (Real-time) excels at efficient and fast AI image generation. This model is a distilled version
# of the LCM-LoRA adapter specifically designed for the Vega model. Its key innovation lies in the significant reduction
# of the number of inference steps needed to produce a high-quality image. Typically, AI image generation models require
# multiple steps to gradually build up the image detail and quality. However, with the Segmind VegaRT, this process is 
# streamlined to require only between 2 to 8 steps. This reduction in steps allows the model to generate images at an 
# exceptionally fast pace, essentially keeping up with the speed of inputing a text prompt. This advancement marks a 
# substantial leap in real-time image generation capabilities, offering both speed and quality.

# Stable Diffusion Fine-tuning
# Fine-tuning involves further training the base Stable Diffusion model using a dataset tailored to a particular subject 
# or style of interest. This approach refines the model, enabling it to retain its extensive general capabilities while 
# also becoming adept in the specified field.
# 
# Checkpoint Training
# Checkpoint training expands a base Stable Diffusion model's capabilities by incorporating a new dataset focused on a specific 
# theme or style. This method enhances the model's proficiency in areas like anime or realism, equipping it to produce content 
# with a distinct thematic emphasis. It's particularly effective for instilling a desired bias in the model.  
# 
# Dreambooth
# Dreambooth training significantly improves a model's proficiency in generating images of a specific subject. It achieves this 
# by utilizing a small set images, associating them with a unique token. This allows the model to render the subject in various 
# styles and settings. An example of Dreambooth's application is integrating personal photos into the model, thus enabling it to
# create images featuring these specific subjects. Models trained via Dreambooth use a distinct keyword to navigate the image 
# generation process.
#
# LoRA
# LoRA stands for Low-Rank Adaptation. LoRA models are streamlined adaptations of Stable Diffusion that implement minor but 
# effective changes to the standard models. These models are considerably smaller, often many times less in size than full 
# checkpoint models, making them suitable for handling numerous models. LoRA training balances efficiency and effectiveness,
# offering relatively smaller file sizes without significantly sacrificing training quality. Requiring a small set of images,
# LoRA training is a more efficient fine-tuning approach, modifying only a portion of the weights.
#
# Dreambooth LoRA
# Dreambooth LoRA, which combines Dreambooth fine-tuning on the base Stable Diffusion model with the subsequent extraction of
# the LoRA component. Initially, the base SD model is fine-tuned using a select set of images through Dreambooth, enhancing 
# its ability to generate highly personalized and style-specific content. This is followed by extracting LoRA from the 
# Dreambooth-trained checkpoint. The integration of Dreambooth and LoRA enables capability to efficiently produce 
# high-quality, customized images, blending personalization with advanced rendering architecture.
# 
# Fine-tuning with Dreambooth LoRA
# Textual Inversion
# Textual Inversion is a technique used in diffusion models like Stable Diffusion to teach the AI new associations between 
# text and images. In this process, a pre-trained model is further trained by introducing a very small set of images along
# with their corresponding textual descriptions. This enables the model to establish new embeddings that link the provided
# text with the images. Essentially, it's a method of expanding the model's vocabulary and understanding, allowing it to 
# generate images that are aligned with newly introduced concepts or terms. The training mimics the model's original 
# reverse process of diffusion, where it learned to reconstruct images from noise, but in this case, it's learning to
# associate new text with specific images.
# 
# Stable Diffusion File Formats( me: add more fileformats, onnx, diffusers, coreml, etc)
# In stable diffusion, two primary file types are commonly encountered: .ckpt files and .safe tensor files. 
# These files serve as containers for the models and provide the necessary data for the diffusion process.
# 
# CKPT
# CKPT stands for "checkpoint". A checkpoint file typically contains all the information about a trained model — this includes
# the model's weights, its architecture, and the state of the optimizer. These files are crucial for resuming training, 
# transferring learning, or deploying the model for inference. CKPT files are directly used by the model for training or
# generating images. They are integral in various stages of model development and deployment. While CKPT files are 
# versatile, there's a theoretical risk that they could contain malicious code, particularly if they are obtained 
# from untrusted sources. This is because they can store custom operations or layers that might execute unwanted code.
# 
# Safetensors
# Safetensors is a file format developed to address some of the potential security concerns associated with CKPT files.
# As the name suggests, it aims to ensure safety in tensor operations. CKPT files can be converted into Safetensors. 
# This process involves extracting the essential information (like weights) from the CKPT files and storing them in 
# a safer, more restricted format.The primary advantage of Safetensors is the increased security. By design, they are
# less likely to harbor malicious code, making them a safer option when using models from less verified sources. 
# There are claims that models loaded from Safetensors may perform faster than those loaded from CKPT files. 
# This could be due to optimizations in how the data is stored and accessed, though the actual performance gain
# can vary based on the specific implementation and use case.
# 
# Stable Diffusion Utility Models
# Image generation with Stable Diffusion complemented with utility models like ESRGAN, Codeformer and Segment Anything Model (SAM),
# to enhance image generation ability of Stable Diffusion.
# 
# Upscaling
# Upscaling refers to the process of increasing the resolution of generated images to make them clearer and more detailed. 
# This is particularly important because generative models like Stable Diffusion often produce images at lower resolutions
# due to computational constraints. Two popular upscaling techniques used in conjunction with Stable Diffusion are ESRGAN 
# and Codeformer.
# 
# (R): Image Enhacement with ESRGAN; (L) Face Recovery with Codeformer
# Segmentation
# The "Segment Anything" model, as its name suggests, is designed for versatile and robust image segmentation tasks. 
# Image segmentation is a crucial process in computer vision where an image is divided into different segments to 
# simplify or change the representation of an image into something more meaningful and easier to analyze.
# 
# Segmentation
# ControlNets
# ControlNets  augment and control Stable Diffusion models and are used in conjunction with any of the Stable Diffusion models,
# enhancing their functionality.
# The fundamental usage of Stable Diffusion models is in the text-to-image format. Text prompts are used as the primary conditioning
# factor. These prompts guide the image generation process, ensuring that the produced images align with the text's description.
# ControlNet introduces an additional layer of conditioning to this process, extending beyond just text prompts. This extra layer of
# conditioning in ControlNet can manifest in various forms. It allows for more nuanced and precise control over the image generation,
# giving users the ability to influence and direct the outcome in more specific ways than with text alone. This multi-conditioning 
# approach significantly broadens the scope and capabilities of Stable Diffusion models in creating tailored and detailed images.
# 
# Canny
# Canny preprocessor extract the outlines of an image, effectively capturing its essential composition and structure. 
# This feature is particularly useful for preserving the fundamental composition of the original image. By focusing on outlines, 
# key shapes and forms, ensuring that the core visual elements are maintained
#
# Depth
# Depth preprocessor estimates depth information from a given reference image. This conditioning is particularly adept at analyzing 
# the visual cues within a two-dimensional image to deduce how far objects are from the viewer, effectively creating a depth map
#
# Open Pose
# Open Pose conditioning detects human key points such as the positions of the head, shoulders, hands, and other body parts. 
# It is particularly valuable for replicating human poses in images while omitting other specifics like outfits, hairstyles, 
# and backgrounds, making it ideal for applications where the focus is on pose accuracy and not on other details
# 
# Scribble
# Scribble conditioning of Stable Diffusion,  replicates the look and feel of a scribble drawing, achieving an artistic effect 
# that mimics hand-drawn lines and strokes.
# 
# Softedge
# Soft Edge conditioning extract and highlight edges from images, creating sketch-like line drawings. It efficiently captures 
# intricate details while filtering out noise, and then colorizes these extracted lines and contours to produce the final image.
# 
# MulticontrolNet
# Multi-ControlNet, you use multiple conditioning's on Stable Diffusion at the same time. This means you can combine more than one 
# of the above ControlNet conditionings to guide the creation of a single image. Essentially, it allows for more complex and layered
# instructions to be used in image generation, leveraging the strengths of multiple models simultaneously.
# 
# IP Adpater
# IP Adapter enables pre-trained text-to-image models to understand and respond to image prompts in addition to text prompts. 
# This adds a new dimension to the model's capabilities, allowing it to process and integrate visual information directly as part of 
# the image generation process.
# 
# Image Prompt
# Image prompt (IP Image) is an image used as input to guide or influence the output of the model.
# 
# IP Adapter + Canny
# Canny edge preprocessor, extracts outlines from images, aiding in maintaining the original composition. Combined with IP Adapter, 
# merges elements from both prompts, with text providing further refinement. The outcome is complex, context-rich images that smoothly
# integrate visual aspects guided by the text prompt.
# 
# IP Adapter XL Canny
# IP Adapter + Depth
# Depth Preprocessor, extracts depth cues from images to grasp and recreate the scene's spatial dimensions. Combined with the IP adapter,
# it produces images that are detailed and have a rich sense of depth. This system merges elements from the original image and text prompts,
# with the text guiding the refinement process.
# 
# IP Adapter XL Depth
# IP Adapter + Openpose
# Open Pose Preprocessor, excels in detecting and analyzing human poses and gestures in images, crucial for accurately representing human 
# figures and movements from the original scene. Working in conjunction with the IP Adapter, it enables the creation of visually striking
# images that are contextually rich, proving particularly effective in scenarios involving human subjects.
# 
# IP Adapter XL Openpose
# Inpainting
# Inpainting allows to alter specific parts of an image. It works by using a mask to identify which sections of the image need changes. 
# In these masks, the areas targeted for inpainting are marked with white pixels, while the parts to be preserved are in black. 
# The model then processes these white pixel areas, filling them in accordance with the given text prompt.
# Inpainting involves four key components. The first is the Input Image, which is the original image subject
# to alteration or restoration. The quality and resolution of this image are crucial, as they significantly
# influence the final outcome. This image could vary from a photograph to a digital painting, or even a scan
# of a physical document. Next is the Mask Image, which is vital in the inpainting process. It acts as a guide,
# using a binary mask where white pixels indicate areas to be altered (the inpainting areas), and black pixels 
# mark the regions to remain unchanged. Following this is the Prompt, comprising textual descriptions or 
# instructions that direct the model on what to generate in the masked areas. For instance, a prompt like 
# "a lush green forest" will lead the model to fill the masked area with imagery matching this description.
# The final component is the Output, which is the altered image where the masked areas are filled in or changed 
# according to the prompt, ensuring a seamless blend with the unmasked parts of the original image.
# 
# Inpainting Steps
# Outpainting
# Outpainting is a technique that uses AI to generate new pixels that seamlessly extend an image's existing bounds. 
# This means that you can add new details to an image, extend the background, or create a panoramic view without any visible seams or artifacts.
# Outpainting Model expanding the edges of the image
# 
# Inference
# Inference refers to the process of generating an image output based on a given input prompt. 
# This input can be either textual or an image. In case of a text prompt, the model interprets the text and generates an image
# that aligns with the description provided in the prompt. For an image input, the model might perform tasks such as style transfer,
# image enhancement, or other modifications based on the specific instructions included with the image. It is essentially about the 
# model applying its learned patterns and knowledge to create a new, unique output based on the input it receives.
# 
# Prompt
# A prompt is a set of textual instructions or descriptions provided to the model to guide the creation of an image. 
# Prompts are crucial for steering the output of the model, as they define what the generated image should depict or represent.
# Some advanced models, such as those based on IP Adapter, allow for a combination of text and image prompts, enabling more complex 
# and nuanced image generation.
# 
# Negative Prompt
# This refers to any specific attributes or characteristics that you do not want to appear in the generated image. 
# If you don't want the model to generate images with a certain feature, you can specify that in the negative prompt.
# 
# Steps
# This is the number of inference steps the model will take to generate the output. 
# Higher values can lead to more refined images, but it may take longer and consume more computational resources.
# 
# Guidance Scale
# This parameter is responsible for controlling the strength of the guidance from the input prompt. 
# Higher values would result in a stronger emphasis on the input text prompt, possibly leading to a more accurate representation of
# the prompt in the output image.
#
# Seed
# This is a random seed number that is used to initialize the random number generator for the model. 
# This seed number ensures that the model will generate the same output if it is run with the same inputs and parameters in
# the future. It's a useful tool for reproducibility.
# 
# Styles
# With Stable Diffusion XL, you have a rich palette of diverse styles to choose from, allowing you to dictate the visual language of
# your output. Whether you're aiming for the sharp realism of photography, the playful and exaggerated features of a cartoon, the defined
# and minimalist strokes of line art, or the geometric simplicity of low poly, the choice is yours. Each style brings its own flavor, 
# transforming the same prompt into vastly different visual experiences. It's like choosing between oil paints, watercolors, charcoal,
# or pastels for a piece of art. Your chosen style can dramatically influence the mood, tone, and impact of the final image.
# {
#   "prompt": "cinematic film still, 4k, realistic, ((cinematic photo:1.3)) of panda wearing a blue spacesuit, sitting in a bar, Fujifilm XT3, long shot, ((low light:1.4)), ((looking straight at the camera:1.3)), upper body shot, somber, shallow depth of field, vignette, highly detailed, high budget Hollywood movie, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
#   "negative_prompt": "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft",
#   "style": "base",
#   "samples": 1,
#   "scheduler": "UniPC",
#   "num_inference_steps": 25,
#   "guidance_scale": 8,
#   "strength": 0.2,
#   "high_noise_fraction": 0.8,
#   "seed": 468685,
#   "img_width": 1024,
#   "img_height": 1024,
#   "refiner": true,
#   "base64": false
# }
# SDXL Inference
# Stabel Diffusion Workflows
# Developing a remarkable piece of art using Stable Diffusion involves a series of intentional and strategic steps, 
# AKA workflow or pipeline. Each  is critical, contributing significantly to the enhancement of the final artwork's 
# aesthetic beauty and charm. This systematic approach ensures that every element, from the initial concept to the 
# final rendering, cohesively works together to create a visually stunning and compelling masterpiece.
# 
# This article serves as a comprehensive review of Stable Diffusion, encompassing an array of concepts for easy access
# to information. The content will be regularly updated to ensure it reflects the most current information available.
##
#
# side note 1:
# DiffusionPipeline() is the base class which StableDiffusionPipline inherits from, since we want to
# work with text-to-image, we can start using StableDiffusionPipeline first, but since we can load it with
# DiffusionPipeline as well and newer models such as XDLv1.0 dont work properly with StableDiffusionPipeline
# we use DiffusionPipeline from now on.
#
# lets use StableDiffusionPipline/DiffusionPipeline and load the weights for stable-diffusion-v1-4 from CompVis repo
# we use the fp16 variant so it takes much less space. 
# since we are using fp16, we directly specify the output type by setting torch_dtype, we can also
# use 'auto' to let the class infer the type from the downloaded weights, the default is torch.float32
# sidenote: 
# Make sure to have a look at the from_pretrained() docs/help, it has a wealth of information
# and keywords to use!
# by the way our model size is around 1.7Gb in fp16 form! so it should be downloaded fairly quickly!
#
# sidenote: Difference between v1.4 and v1.5
# The CompVis Stable-Diffusion-v1-4 checkpoint was initialized with the weights of the Stable-Diffusion-v1-2
# checkpoint and subsequently fine-tuned on 225k steps at resolution 512x512 on "laion-aesthetics 
# v2 5+" and 10% dropping of the text-conditioning to improve classifier-free guidance sampling.
# 
# The Runway Stable-Diffusion-v1-5 checkpoint was initialized with the weights of the Stable-Diffusion-v1-2
# checkpoint and subsequently fine-tuned on 595k steps at resolution 512x512 on "laion-aesthetics 
# v2 5+" and 10% dropping of the text-conditioning to improve classifier-free guidance sampling.
#
# repo_model_name = "CompVis/stable-diffusion-v1-4"
repo_model_name = "runwayml/stable-diffusion-v1-5"
# this is the latest base model its fp16 is 6.5Gb and offers native 1024x1024
# repo_model_name = "stabilityai/stable-diffusion-xl-base-1.0" 
# the trubo version, takes as much but faster in image generation : https://huggingface.co/stabilityai/sdxl-turbo
# repo_model_name = "stabilityai/sdxl-turbo"
#
# by default these models will be downloaded and stored in the ~/.cache/huggingface/hub directory under your user dir
# so make sure your home directory has enough space, otherwise we can use cache_dir parameter to set the desired path 
# to download and store the models into.
print(f'using {repo_model_name}...')
# text2image = StableDiffusionPipeline.from_pretrained(repo_model_name,
#                                                 variant='fp16',
#                                                 # cache_dir="./models"  
#                                                 torch_dtype= torch.float16,
#                                                 # incase our download is interuppted due to bad connection
#                                                 # lets resume from where we left off last time
#                                                 resume_download=True).to('cuda')
# # 
# for sdxl-base-1.0 DiffusionPipeline works!
text2image = DiffusionPipeline.from_pretrained(repo_model_name,
                                                variant='fp16',
                                                torch_dtype= torch.float16,
                                                # cache_dir="./models"                                                  
                                                # disable the internal nsfw checker
                                                safety_checker=None,
                                                requires_safety_checker=False,
                                                # incase our download is interuppted due to bad connection
                                                # lets resume from where we left off last time
                                                resume_download=True).to('cuda')
#
# by the way we could also use AutoPipelineForText2Image as well, everything stays the same!
# text2image = AutoPipelineForText2Image.from_pretrained(repo_model_name,
#                                                 variant='fp16',
#                                                 torch_dtype= torch.float16,
#                                                 # cache_dir="./models"         
#                                                 # incase our download is interuppted due to bad connection
#                                                 # lets resume from where we left off last time
#                                                 resume_download=True).to('cuda')

# if we dont have enough vram! we can use Enable sliced attention computation. 
# when this option is enabled, the attention module splits the input tensor
# in slices to compute attention in several steps. For more than one attention head, the computation
# is performed sequentially over each head. This is useful to save some memory in exchange for
# a small speed decrease.
#
# Side Note - Slow Down WARNING!
# Don't enable attention slicing if you're already using `scaled_dot_product_attention`` (SDPA) 
# from PyTorch 2.0 or xFormers. These attention computations are already very memory efficient 
# so you won't need to enable this function. 
# If you enable attention slicing with SDPA or xFormers, it can lead to serious slow downs!
# 
# the default value for enable_attention_slicing is "auto", which by default halves the input 
# to the attention heads, so attention will be computed in two steps. 
# the other options are "max", or a custom slice_size. 
# The "max", option means, maximum amount of memory will be saved by running only one slice at a time.
# If a number is provided, it uses as many slices as `ttention_head_dim // slice_size`. 
# In this case, "attention_head_dim" must be a multiple of "slice_size".
# 
text2image.enable_attention_slicing() 
# or
# text2image.enable_attention_slicing("max")
#
# now we simply pass our textual prompt, describing the image we want and get the result
# the output is a dictionary with two keys, "images" in pil format, and "nsfw_content_detected"
# which is a boolean value denoting whether ther result contains anything not safe for work and if
# it does, it returns a black image instead! (its detection can not be trusted though! its not accurate)
prompt = "a beautiful day in a lush forest"
result = text2image(prompt=prompt, height=512,width=512)
# doesnt work on the sdxl_turbo model
if 'xl' not in repo_model_name:
    print(f'{result.nsfw_content_detected=}')
plt.imshow(result.images[0])
plt.show()
#%%
# now if we want we can send a list of prompts and get a list of images
prompt = ["a beautiful day in a lush forest","a dog sleeping at the beach"]
result = text2image(prompt)
# doesnt work on the sdxl_turbo model
if 'xl' not in repo_model_name:
    print(f'{result.nsfw_content_detected=}')
def display_images(prompt, result,split=50):
    for msg, img in zip(prompt,result.images):
        plt.imshow(img)
        msg = '\n'.join([msg[i:i+split] for i in range(0, len(msg), split)]) + '\n'
        plt.title(msg)
        plt.show()
        
display_images(prompt, result)
#%%
# As we just saw, this pipeline allows us to easily use a diffusion model, but it also hides a lot
# of details. so in this section lets dive deeper.
# lets see what our pipeline object has to offer. 
print(f'{text2image.__dict__.keys()}')
# which gives us : 
# dict_keys(['_internal_dict',
# 'vae', 'text_encoder', 'tokenizer', 'unet', 'scheduler', 'safety_checker', 'feature_extractor',
# 'image_encoder', 'vae_scale_factor', 'image_processor', 
# '_guidance_scale', '_guidance_rescale', '_clip_skip', '_cross_attention_kwargs', '_interrupt', '_num_timesteps', '_progress_bar_config'])
# 
# The "vae" (AutoencoderKL) is  the Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
# The "text_encoder" (transformers.CLIPTextModel) is the frozen text-encoder (clip-vit-large-patch14).
# The "tokenizer" (transformers.CLIPTokenizer) is a CLIPTokenizer to tokenize the input text.
# The "unet" (UNet2DConditionModel) is a UNet2DConditionModel to denoise the encoded image latents.
# The scheduler (SchedulerMixin) is a scheduler(or sampler) to be used in combination with "unet" to denoise 
# the encoded image latents. It can be one of: DDIMScheduler, LMSDiscreteScheduler, or PNDMScheduler
# to see our current scheduler we simply use the scheduler property
# The image_encoder is for ipadapter workloads, (explained in the next part)
print(f'{text2image.scheduler=}')
# which prints 
# text2image.scheduler=PNDMScheduler {
#   "_class_name": "PNDMScheduler",
#   "_diffusers_version": "0.26.2",
#   "beta_end": 0.012,
#   "beta_schedule": "scaled_linear",
#   "beta_start": 0.00085,
#   "clip_sample": false,
#   "num_train_timesteps": 1000,
#   "prediction_type": "epsilon",
#   "set_alpha_to_one": false,
#   "skip_prk_steps": true,
#   "steps_offset": 1,
#   "timestep_spacing": "leading",
#   "trained_betas": null
# }
# if at somepoint we decided we want to change our scheduler to something else
# we can see a list of compatible schedulers with our current one. 
print(f'compatible schedulers: {text2image.scheduler.compatibles}')
# prints:
# compatible schedulers: [
# <class 'diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler'>,
# <class 'diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler'>,
# <class 'diffusers.utils.dummy_torch_and_torchsde_objects.DPMSolverSDEScheduler'>,
# <class 'diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler'>,
# <class 'diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler'>,
# <class 'diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler'>,
# <class 'diffusers.schedulers.scheduling_ddim.DDIMScheduler'>,
# <class 'diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler'>,
# <class 'diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler'>,
# <class 'diffusers.schedulers.scheduling_pndm.PNDMScheduler'>,
# <class 'diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler'>,
# <class 'diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler'>,
# <class 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler'>,
# <class 'diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler'>]
# now if we want to go on, we simply use the config property in combination with the from_config() on the new
# scheduler to initialize it. 
# this returns a dictionary of the configuration of the scheduler
config = text2image.scheduler.config 
print(f'scheduler config: {config}')
# which prints 
# scheduler config: FrozenDict([
    # ('num_train_timesteps', 1000),
    # ('beta_start', 0.00085),
    # ('beta_end', 0.012),
    # ('beta_schedule', 'scaled_linear'),
    # ('trained_betas', None),
    # ('skip_prk_steps', True),
    # ('set_alpha_to_one', False),
    # ('prediction_type', 'epsilon'),
    # ('timestep_spacing', 'leading'),
    # ('steps_offset', 1),
    # ('_use_default_values',
    # ['timestep_spacing', 'prediction_type']),
    # ('_class_name', 'PNDMScheduler'),
    # ('_diffusers_version', '0.26.2'),
    # ('clip_sample', False)])
text2image.scheduler = dfs.DDIMScheduler.from_config(config)
# 
# Here is a brief explanation of each scheduler you mentioned:
# 1. **DPMSolverSDEScheduler**: This scheduler implements the stochastic sampler from the Elucidating the Design Space of Diffusion-Based Generative Models paper[^30^].
# 2. **EulerDiscreteScheduler**: This is a fast scheduler which can often generate good outputs in 20-30 steps⁸.
# 3. **LMSDiscreteScheduler**: LMSDiscreteScheduler is a linear multistep scheduler for discrete beta schedules. The scheduler is ported from and created by Katherine Crowson⁹.
# 4. **DDIMScheduler**: Creates a new DDIM scheduler given the number of steps to be used for inference as well as the number of steps that was used during training¹⁶.
# 5. **DDPMScheduler**: DDPMScheduler extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with non-Markovian guidance¹³.
# 6. **HeunDiscreteScheduler**: I couldn't find specific information about this scheduler.
# 7. **DPMSolverMultistepScheduler**: You can use a combination of offset=1 and set_alpha_to_one=False to make the last step use step 0 for the previous alpha product like in Stable Diffusion[^30^].
# 8. **DEISMultistepScheduler**: DEISMultistepScheduler is a fast high order solver for diffusion ordinary differential equations (ODEs). This implementation modifies the polynomial fitting formula in log-rho space instead of the original linear tspace in the DEIS paper²⁴.
# 9. **PNDMScheduler**: By default, the stable diffusion pipeline uses the PNDM scheduler¹⁵.
# 10. **EulerAncestralDiscreteScheduler**: EulerAncestralDiscreteScheduler is based on the timestep, a scheduler may be discrete in which case the timestep is an int or continuous in which case the timestep is a float²⁷.
# 11. **UniPCMultistepScheduler**: I couldn't find specific information about this scheduler.
# 12. **KDPM2DiscreteScheduler**: I couldn't find specific information about this scheduler.
# 13. **DPMSolverSinglestepScheduler**: I couldn't find specific information about this scheduler.
# 14. **KDPM2AncestralDiscreteScheduler**: I couldn't find specific information about this scheduler.
#
#  
# The "safety_checker" (StableDiffusionSafetyChecker) is a Classification module that estimates whether 
# generated images could be considered offensive or harmful.
# The "feature_extractor" (transformers.CLIPImageProcessor) is a CLIPImageProcessor to extract features from 
# generated images, its used as inputs to the "safety_checker".
# The "requires_safety_checker" is a boolean value specifying whether to run safty check on the output or not the default is Ture. 
# we ignore the ones with _, as they are implementation details and we will get to them later on inshaallah.
# 
# this command in jupyeter notebook allows us to see the implementation details of our pipeline
# which if we have a look at, we'll find a lot of useful comments concerning how certain sections work!
# note that this will print the whole source code for our pipeline! if youre in vscode
??text2image
# and we can see a few intersting arguments we can utilize to have more control on the result
# note that obviously the keywords are different for the sdxl version of the models.
# below we are looking at some of the keywrods for the sdv1.5 has, note that these are not all the keywords!
# just some of the most used ones
# prompt: Union[str, List[str]] = None,
# height: Optional[int] = None,
# width: Optional[int] = None,
# num_inference_steps: int = 50,
# timesteps: List[int] = None,
# guidance_scale: float = 7.5,
# negative_prompt: Union[str, List[str], NoneType] = None,
# num_images_per_prompt: Optional[int] = 1,
# eta: float = 0.0,
# generator: Union[torch._C.Generator, List[torch._C.Generator], NoneType] = None,
# latents: Optional[torch.FloatTensor] = None,
# prompt_embeds: Optional[torch.FloatTensor] = None,
# negative_prompt_embeds: Optional[torch.FloatTensor] = None,
# ip_adapter_image: Union[PIL.Image.Image, numpy.ndarray, torch.FloatTensor, List[PIL.Image.Image], List[numpy.ndarray], List[torch.FloatTensor], NoneType] = None,
# output_type: Optional[str] = 'pil',
# return_dict: bool = True,
# cross_attention_kwargs: Optional[Dict[str, Any]] = None,
# guidance_rescale: float = 0.0,
# clip_skip: Optional[int] = None,
# callback_on_step_end: Optional[Callable[[int, int, Dict], NoneType]] = None,
# callback_on_step_end_tensor_inputs: List[str] = ['latents'],
# **kwargs,
# )
# 
# sidenote: these 
# 
# %%
# so lets do some testing with our new parameters
# smaller images requires less vram! but the results may not turnout good! the higher the better!(but up to 512!
# as the 512 is the native resolution 1.4/1.5 support(we get higher resolution using superresolution later on))
# the inclusion of terms related to high quality, or a specific artstyle or photographer, etc will 
# result in the image to take those said terms. consider the following prompts, try adding and 
# removing the following concepts : 
# Renaissance-style 
# Surrealist painting
# Impressionist
# Abstract painting 
# Pop Art painting
# Baroque-style 
# Cubist painting 
# Art Nouveau
# Romantic painting
# Gothic painting
# Black and white street photography
# High-fashion photography
# Macro photography
# Aerial photography 
# Urban portrait
# Underwater photography
# Vintage-style
# Long-exposure night photography
# Documentary-style photography 
# Fantasy illustration
# Digital painting
# Concept art 
# Pixel art 
# photorealistc 8k
# warm portrait
# kodak portrait
# Closeup portrait
# and many more you should now have an idea how stuff work
# you can find more about such useful prompts on websites such as 
# https://mspoweruser.com/best-stable-diffusion-prompts/ and
# https://medium.com/phygital/top-40-useful-prompts-for-stable-diffusion-xl-008c03dd0557 
# and many more if you google for it
prompt = ["a beautiful photorealistic horse in a lush forest",
          "a Vintage-style photo of a family playing at the beach"]
# resolution also plays an important role in good outcome, 
# usually the smaller the size, the worse the result
height = 512
width = 512
# num_inference_steps controls the number of denoising steps during the image generation process.
# it is directly correlated with the scheduler we use, the default for v1.4 and v1.5 is 50
# fewer steps may result in botched/ugly/weird outcome! while more steps can result better
# although too many steps can also destroy the image. usually 25~50 is used, 30 being the most common
# but as I said its model and scheduler related.
# the sdxlv1.0 requires fewer iteration to create good images! sdxlv1.0-turbo requires 1-4 only!
# (its a distilled version of the base sdxlv1.0 finetuned using Adversarial Diffusion Distillation (ADD))
num_inference_steps = 50
# The `timesteps` parameter is a tensor that contains the timesteps at which the model denoises 
# an image. Each element in this tensor corresponds to a timestep.
# this tensor is created when we set the number of timesteps for the denoising process. 
# For example, if we set the number of timesteps to 50, the scheduler creates a tensor 
# with 50 evenly spaced elements.
# during the denoising process, we iterate over this tensor to denoise an image. 
# at each timestep, the model predicts the noise residual and the scheduler uses it to
# predict a less noisy image.
# the more noise is on the image, the more the network spends on less relevant visual features,
# so typically more examples are sampled at earlier time steps than later time steps.
# It's important to set the timesteps  carefully because:
# The number of timesteps can affect the quality of the generated image and the time it takes 
# to generate the image.
# Using too few timesteps may result in a "butched" outcome, where the image is not denoised
# properly.
# Using too many timesteps can make the image generation process slower¹.
# The recommended number of timesteps can depend on your specific use case and the complexity 
# of the text prompt. It's always best to experiment with different numbers of timesteps to 
# see what works best for your use case.
# Source: Conversation with Bing, 2/12/2024
# (1) Understanding pipelines, models and schedulers - Hugging Face. https://huggingface.co/docs/diffusers/main/en/using-diffusers/write_own_pipeline.
# (2) How and why stable diffusion works for text to image generation. https://www.paepper.com/blog/posts/how-and-why-stable-diffusion-works-for-text-to-image-generation/.
# (3) A Comprehensive Beginner's Guide to Stable Diffusion: Key Terms and .... https://blog.segmind.com/the-a-z-of-stable-diffusion-essential-concepts-and-terms-demystified/.
# (4) T-Stitch: Accelerating Sampling in Pre-trained Diffusion Models with .... https://t-stitch.github.io/.
# side note:
# The num_inference_steps and timesteps both control the number of denoising steps during the
# image generation process. If we only specify num_inference_steps, the model will automatically
# generate a tensor of timesteps evenly spaced between 0 and 1. This may make timesteps argument
# excessive and useless.
# However, the timesteps parameter gives us more control over the specific points in time 
# at which the denoising process is applied. By providing a custom timesteps tensor, we can 
# control the distribution of the timesteps, which can influence the denoising process and 
# the final generated image.
# For example, we might want to use more timesteps early in the process when the image is 
# noisier, and fewer timesteps later when the image is closer to its final state. 
# This could potentially improve the quality of the generated image or speed up the generation
# process.
# So while num_inference_steps and timesteps can both be used to control the number of denoising
# steps, they offer different levels of control. 
# num_inference_steps is simpler to use and may be sufficient for most use cases, 
# while timesteps offers more flexibility for advanced use cases or for users who want to
# experiment with different timestep distributions.
# Sidenote 2: 
# we may ask ourselves why would I want to use timesteps when its already taken care of when 
# I set num_inference_steps. it seems it should not have been exposed, as its an implementation
# detail! setting it from outside provides no benifit to when we set num_inference_steps, and 
# that creates a timesteps tensor automatically internally.
#
# well its true that num_inference_steps and timesteps both control the number of denoising steps 
# during the image generation process. If we only specify num_inference_steps, the model 
# will automatically generate a tensor of timesteps evenly spaced between 0 and 1.(this is important remember this)
# However, the timesteps parameter gives us more control over the specific points in time at
# which the denoising process is applied. By providing a custom timesteps tensor, we can 
# control the distribution of the timesteps, which can influence the denoising process and 
# the final generated image.
# For example, we might want to use more timesteps early in the process when the image is 
# noisier, and fewer timesteps later when the image is closer to its final state. 
# This could potentially improve the quality of the generated image or speed up the generation
# process.
# So while num_inference_steps and timesteps can both be used to control the number of denoising
# steps, they offer different levels of control. num_inference_steps is simpler to use and may
# be sufficient for most use cases, while timesteps offers more flexibility for advanced use 
# cases or for users who want to experiment with different timestep distributions.
#
# let's delve into it a bit more.
# The timesteps parameter in the Stable Diffusion model allows us to specify the exact points in 
# time at which the denoising process is applied. This gives us control over the distribution of
# the timesteps so far so good.
# we also know that in the diffusion process, the image starts as noise and gradually becomes 
# less noisy over time. Early in the process, when the image is very noisy, small changes can
# have a big impact on the final image. Later in the process, when the image is less noisy, 
# changes tend to be more subtle.
# Now, by providing a custom timesteps tensor, we can control where in this process we want to
# focus the denoising steps. 
# that is for example, we might want to use more timesteps early in the process when the image 
# is noisier, and fewer timesteps later when the image is closer to its final state. 
# This could potentially improve the quality of the generated image or speed up the generation 
# process.
# Here's an example of how we might create a custom timesteps tensor that focuses more on the
# early stages of the process:
# import torch
# # Create a tensor with more timesteps early in the process
# timesteps = torch.cat([torch.linspace(0, 0.5, steps=30), torch.linspace(0.5, 1, steps=10)])
# 
# In this code, torch.linspace(0, 0.5, steps=30) creates a tensor with 30 timesteps evenly spaced
# between 0 and 0.5, and torch.linspace(0.5, 1, steps=10) creates a tensor with 10 timesteps 
# evenly spaced between 0.5 and 1. The torch.cat function then concatenates these two tensors 
# to create a single timesteps tensor.
# This timesteps tensor has a total of 40 elements, with 30 of them focused on the first half
# of the process and 10 of them focused on the second half. This means the model will apply 
# more denoising steps early in the process when the image is noisier.
# now if you recall our previous headsup, you remember that by default the model generates a
# timesteps tensor with uniform step sizes(timesteps evenly spaced between 0 and 1), whereas here, we specify the density of numbers
# how close/far away/etc the values are for each point, in different [denoising] steps so to speak!
# this should have hopefully clarified how the timesteps parameter can be used to control the
# distribution of the denoising steps! 
# side note: 
# note that not all schedulers/samplers support custom timesteps, that is if we set 
# timesteps other than None, while using those schedulers like PNDMScheduler for example, it will error out!
# timesteps = torch.cat([torch.linspace(0, 0.5, steps=30), torch.linspace(0.5, 1, steps=10)])
# text2image.scheduler = dfs.schedulers.scheduling_ddim.DDIMScheduler.from_config(text2image.scheduler.config)
# text2image.scheduler.set_timesteps(timesteps)
#! sidenote: I myself couldnt get this to work! asked a question and still no answer
timesteps = None
# This parameter corresponds to the eta (η) parameter in the DDIM paper. 
# It only applies to DDIMScheduler and will be ignored for others. The eta parameter controls
# the noise schedule in the diffusion process.
eta = 0.0
# this is used to create deterministic results (note the device=cpu part)
# generator = torch.Generator(device="cpu").manual_seed(128)
# torch.manual_seed(), sets the seed for the global generator, which might be used elsewhere in our code
# or in the libraries we're using, however, the GPU is mostly nondeterminstic unless we specifically disable
# nondeterminstic kernels. For this reason, we rather create the generator on the cpu, the tensor on the cpu
# and then move it to the GPU. this way, we create a determinstic outcome, that is not possible if we simply 
# use torch.manual_seed().  
# torch.Generator(device="cpu").manual_seed(), ensures that the same sequence of random numbers is used in 
# the diffusion pipeline every time we run our code, leading to the same image being generated.
# from : https://huggingface.co/docs/diffusers/using-diffusers/reproducibility#control-randomness
# Every time the pipeline is run, torch.randn uses a different random seed to create Gaussian noise 
# which is denoised stepwise. This leads to a different result each time it is run, which is great 
# for diffusion pipelines since it generates a different random image each time.
# But if you need to reliably generate the same image, that’ll depend on whether you’re running the
# pipeline on a CPU or GPU.
# To generate reproducible results on a CPU, you’ll need to use a PyTorch Generator and set a seed:
# sidenote: 
# It might be a bit unintuitive at first to pass Generator objects to the pipeline instead of just 
# integer values representing the seed, but this is the recommended design when dealing with probabilistic
# models in PyTorch, as Generators are random states that can be passed to multiple pipelines in a sequence.
# Writing a reproducible pipeline on a GPU is a bit trickier, and full reproducibility across different 
# hardware is not guaranteed because matrix multiplication - which diffusion pipelines require a lot of -
# is less deterministic on a GPU than a CPU. 
# So eveing using torch.Generator(device="cuda").manual_seed(128), the result is not the same even though
# we're using an identical seed because the GPU uses a different random number generator than the CPU.
# To circumvent this problem, Diffusers has a randn_tensor() function for creating random noise on the CPU,
# and then moving the tensor to a GPU if necessary. The randn_tensor function is used everywhere inside the
# pipeline, allowing the user to always pass a CPU Generator even if the pipeline is run on a GPU.
# sidenote2: 
# If reproducibility is important, we recommend always passing a CPU generator. The performance loss is 
# often neglectable, and you’ll generate much more similar values than if the pipeline had been run on a GPU.
# Finally, for more complex pipelines such as UnCLIPPipeline, these are often extremely susceptible to 
# precision error propagation. Don’t expect similar results across different GPU hardware or PyTorch versions.
# In this case, you’ll need to run exactly the same hardware and PyTorch version for full reproducibility.
# Deterministic algorithms
# You can also configure PyTorch to use deterministic algorithms to create a reproducible pipeline. 
# However, you should be aware that deterministic algorithms may be slower than nondeterministic ones
# and you may observe a decrease in performance. But if reproducibility is important to you, then this
# is the way to go!
# Nondeterministic behavior occurs when operations are launched in more than one CUDA stream. 
# To avoid this, set the environment variable CUBLAS_WORKSPACE_CONFIG to :16:8 to only use one buffer size
# during runtime.
# PyTorch typically benchmarks multiple algorithms to select the fastest one, but if you want reproducibility,
# you should disable this feature because the benchmark may select different algorithms each time. 
# Lastly, pass True to torch.use_deterministic_algorithms to enable deterministic algorithms.
# sample code: 
# import os, torch
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
# # RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)`
# # or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it
# # uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an 
# # environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or 
# # CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
# import diffusers as dfs
# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = dfs.DiffusionPipeline.from_pretrained(model_id, variant='fp16', torch_dtype= torch.float16).to('cuda')
# pipe.scheduler = dfs.DDIMScheduler.from_config(pipe.scheduler.config)
# g = torch.Generator(device="cuda")
# prompt = "A bear is playing a guitar on Times Square"
# g.manual_seed(0)
# result1 = pipe(prompt=prompt, num_inference_steps=50, generator=g, output_type="latent").images
# g.manual_seed(0)
# result2 = pipe(prompt=prompt, num_inference_steps=50, generator=g, output_type="latent").images
# print("L_inf dist =", abs(result1 - result2).max()) # prints: "L_inf dist = tensor(0., device='cuda:0')"

generator = None
# This parameter is typically used to provide precomputed embeddings for the text prompt. 
# Instead of passing a text string as the prompt, we can pass a tensor of embeddings. 
# This can be useful if we want to use a custom text encoder or if we want to reuse the 
# same prompt embeddings multiple times
# # Precompute the prompt embeddings
# prompt_embeds = text2image.encode_prompt("A beautiful sunset over the mountains")
# # Use the precomputed embeddings to generate an image
# result = text2image(prompt_embeds=prompt_embeds)
prompt_embeds =None
# Similar to prompt_embeds, this parameter is used to provide precomputed embeddings for 
# the negative text prompt. The negative prompt is used to guide the model away from 
# generating certain types of images, more on this in a moment!
# # Precompute the negative prompt embeddings
# negative_prompt_embeds = text2image.encode_prompt("-glasses")
# # Use the precomputed embeddings to generate an image
# result = text2image(prompt="A human", negative_prompt_embeds=negative_prompt_embeds)
negative_prompt_embeds=None
# ipAdapter is short for Image Prompt Adapter, it is a method of enhancing Stable Diffusion models 
# that was developed by Tencent AI Lab and released in August 2023 (IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models: https://arxiv.org/abs/2308.06721)
# their github repo also hosts several demos along with some best practices : https://github.com/tencent-ailab/IP-Adapter
# If we were to describe it in a single sentence it would be "single image fine-tuning", 
# instead of training models with 20-1000 images, all we have to do is input a single image.
# basically this parameter is typically used when we want to incorporate an image alongside our text prompt,
# shaping our final resulting image’s composition, style, color palette, or even faces. 
# I'd like to emphasis that this is really a prompt! we use this image as our base canvas and then using
# our textual prompt, try to finetune it/change it the way we like. 
# This is done by employing an Image Prompt Adapter (IP-Adapter model) so we need an ip-adapter model and
# our base model also needs to support it.
# ref1 https://stable-diffusion-art.com/ip-adapter/
# 
# IP-Adapter models function kind of like ControlNets. 
# There are many ip-adapter models, each have specialized purposes. 
# where do we get our ip-adapter models? huggingface hub is one!
# since we are using sdv1.5 we choose a general model from the official repo here : https://huggingface.co/h94/IP-Adapter
# I said general, becasue there are different models for different tasks, 
# some are specific to faces (require cropped faces to work, etc)
# remember to use ipadapter models, we both need an image encoder model, a weight file. in the given
# link we just saw, you can see each ip-adapter lists its imageencoder as well, so if we are downloading 
# them manually, we need to download them both, and send the image-encoder as image-encoder argument in our
# pipeline constructor when we are intantiating one. otherwise, we can use one of the pipeline utility
# methods to load them both automatically. (this is what we do)
# for the reference this is the model weights url(is around 150mb) and its imageencoder model is OpenCLIP-ViT-H-14 (which is around 2.3Gb)
# ip_adapter_plus_sd15_url = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin?download=true"
# this is much better than the ip-adapter_sd15.bin as ours use patch image embeddings from OpenCLIP-ViT-H-14 
# as condition, closer to the reference image than ip-adapter_sd15. 
# sidenote: The original IP-adapter used the CLIP image encoder to extract features from the reference image.but later on
# used OpenCLIP-ViT-H-14. 
# The novelty of the IP-adapter is training separate cross-attention layers for the image. 
# This makes the IP-adapter more effective in steering the image generation processing. 
# ip-adapter model scheme: https://github.com/tencent-ailab/IP-Adapter/raw/main/assets/figs/fig1.png
# the models we use here, use OpenCLIP-ViT-H-14 with 632.08M parameter however.
# 
# IP-Adapter works with most of our pipelines, including Stable Diffusion, Stable Diffusion XL (SDXL),
# ControlNet, T2I-Adapter, AnimateDiff. And we can use any custom models finetuned from the same base
# models. It also works with LCM-Lora out of box.
# 
# sidenote: 
# we can use the set_ip_adapter_scale() method to adjust the text prompt and image prompt condition ratio.
# If we're only using the image prompt, we should set the scale to 1.0. 
# we can lower the scale to get more generation diversity, but it'll be less aligned with the prompt.
# scale=0.5 can achieve good results in most cases when we use both text and image prompts.
# 
# more explanation: 
# The `set_ip_adapter_scale()` function is used to set the scale of the image prompt adapter in the
# diffusion model. This scale factor can influence the impact of the image prompt on the generated image.
# A higher scale value gives more weight to the image prompt, while a lower scale value reduces its impact.
# As for the importance of the text prompt versus the image in IP-Adapter, both play significant roles 
# but in different ways:
# - **Text Prompt**: The text prompt provides high-level guidance for the image generation process. 
#   It's used to specify the main subject or theme of the generated image.
# - **Image Prompt (IP-Adapter)**: The image prompt, on the other hand, influences the style, color 
#   palette, composition, and even specific features of the generated image. It provides more detailed
#   guidance on the visual aspects of the image.
# In the IP-Adapter mode, the image prompt and the text prompt can work together to achieve multimodal 
# image generation. The image prompt can also work well with the text prompt to accomplish multimodal 
# image generation. This means that the final generated image is influenced by both the text and the 
# image prompts, combining the high-level guidance from the text with the detailed visual guidance from
# the image.
# So, both the text prompt and the image prompt are important in the IP-Adapter mode, and their relative importance can depend on the specific requirements of your image generation task⁴⁵.

# to use it we simply do 
text2image.set_ip_adapter_scale(0.5)
# this downloads all the necessary files and puts them into models directory in the default cache directory which for me is:
# ~/.cache/huggingface/hub/models--h94--IP-Adapter/snapshots/92a2d51861c754afacf8b3aaf90845254b49f219
# the last folder is randomly generated so you get the idea where to find them. like before, we can set "cache_dir" argument
# and specify where we want to store the downloaded models! 
text2image.load_ip_adapter("h94/IP-Adapter", 
                           #here I decided to use the default!
                           cache_dir=None, 
                           # means to go and grab the subfolder named "models" in the main repository
                           # this is the subfolder by the way: https://huggingface.co/h94/IP-Adapter/tree/main/models
                           subfolder="models", 
                           # there are two variants, .bin files and .safetensors files. both will do it
                           # however, safetensors are recommended becasue they are safer! 
                           weight_name="ip-adapter-plus_sd15.bin",
                           resume_download=True)
# IP-Adapter relies on an image encoder to generate the image features, 
# if our IP-Adapter weights folder contains a "image_encoder" subfolder,
# the image encoder will be automatically loaded and registered to the pipeline.
# Otherwise we can so load a CLIPVisionModelWithProjection model and pass it to 
# a Stable Diffusion pipeline when we create it.
#
# from diffusers import AutoPipelineForText2Image, or DiffuserPipeline, or StableDiffusionPipeline
# from transformers import CLIPVisionModelWithProjection
# image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter",subfolder="models/image_encoder",
#                                                               torch_dtype=torch.float16,).to("cuda")
# pipeline = AutoPipelineForText2Image.from_pretrained(repo_model_name, image_encoder=image_encoder, torch_dtype=torch.float16).to("cuda")
# 
# since we are using multiple prompts here, we need to have an ip-adapter model defined for every image. 
# we can load multiple IP-Adapter models and use multiple reference images at the same time. 
# so to do this we simply specify the weights for each ip-adapter model in the weights section 
text2image.load_ip_adapter("h94/IP-Adapter",
                           #here I decided to use the default!
                           cache_dir=None, 
                           # means to go and grab the subfolder named "models" in the main repository
                           # this is the subfolder by the way: https://huggingface.co/h94/IP-Adapter/tree/main/models
                           # if its nested, we use a list!
                           subfolder="models",
                           # since some files like the imagenecoder are big, lets enable this
                           resume_download=True,
                           # since we want to use the same adapter model for our two images 
                           # to show both variants are the same, I use both extensions here
                           weight_name=["ip-adapter-plus_sd15.bin",
                                        "ip-adapter-plus_sd15.safetensors"])#ip-adapter-plus-face_sd15.bin

# In this example we use IP-Adapter-Plus face model to create a consistent character and also 
# use IP-Adapter-Plus model along with 10 images to create a coherent style in the image we generate.
# 
# sidenote: peft(backend) module needs to be installed by the way(pip install --upgrade peft)
# sidenote2: 
# PEFT, or Pretrained Efficient Fine-Tuning, is a library integrated with the Transformers, Diffusers,
# and Accelerate libraries to provide a faster and easier way to load, train, and use large models for 
# inference.
# In the context of IP adapters in diffusion models, PEFT is used to manage and load adapters for inference.
# Adapters are small modules inserted into pre-existing models, allowing us to fine-tune the model on 
# a specific task without having to retrain the entire model.
# For example, we can use PEFT to easily fuse/unfuse multiple adapters directly into the model weights
# (both UNet and text encoder) using the fuse_lora() method, which can lead to a speed-up in inference 
# and lower VRAM usage.
# To perform the adapter injection, we can use the inject_adapter_in_model method that takes 3 arguments:
# the PEFT config, the model itself, and an optional adapter name. 
# we can also attach multiple adapters in the model if we call inject_adapter_in_model multiple times with
# different adapter names.
# img = tfms.ToTensor()(tfms.Resize((512,512))(Image.open("./pretty_mage1.jpeg")))
# print(f'{img.shape=}')
# Sidenote: when using ipadapter our image needs to make sense to our prompt, 
# becasue our prompt is going to work on this given image, that is, our prompt
# changes this given image so to speak
# lets define new prompts for our image prompts
# again, note that we have two prompts here, the text prompt and the image "PROMPT"
# we are using the images as references so the final image takes the shape/form/style/etc
# of the reference image. the text prompt is used to add the changes we want in that seeting!
# also as indicated before, we can have more than 1 reference image, and we decide how much 
# of each reference image prompts we want in our final image by using set_ip_adapter_scale.
# using several image from certain concept/artstyle/etc can allow us to create consistent 
# style for example in our final image. 
# we can also use several images for each model. this is to get the most consistent look of the
# ip-images. for this we simply use a list of images instead of a single image.
# try these and then uncomment or only set one scale (0.5) and see the result
prompt = ["a Robot riding a horse from hell",
          "a boat in heavy thunderstorm ocean with black sky raining"]
ip_img1 = dfs.utils.load_image("/media/hossein/SSD1/code_dl/pretty_image4.jpeg")
ip_img2 = dfs.utils.load_image("./pretty_image2.jpeg")
# imgs taken from "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy"
fldr = "/media/hossein/SSD1/code_dl/"
ip_imgs3 = [dfs.utils.load_image(f"{fldr}/img{i}.png") for i in range(10)]
ip_adapter_image=[ip_img1, ip_img2]
# ip_adapter_image=[ip_imgs3, ip_img2]
# ip_adapter_image=[ip_imgs3, [ip_img1, ip_img2]]
# lets change the scales for these images
# the higher the values, the more impact the image prompts will have. 
# and it will disregard the text prompt, try [0.5,0.5] or higher and see it for yourself
# also these numbers dont need to add to one, as each belongs to one ipadapter model and image
# try [0.4,0.3] and see the result. 
# if you dont set_ip_adapter_scale here, both ref images will affect all promps eaually
# comment these lines and see the effect 
# try the image prompts with these one by one and see how they affect each one
text2image.set_ip_adapter_scale([0.1,0.1])
# text2image.set_ip_adapter_scale([0.2,0.2])
# text2image.set_ip_adapter_scale([0.4,0.3])
# text2image.set_ip_adapter_scale([0.4])
# while you may think this is good for style, its not really that impressive. well see another
# usage next where we can use a single image (like ourseleves, and then use it as image prompt
# and then change it however we like, pose a certain way, wear a specific cloth, etc
# or we can create caricatures, or avatars of ourseleves using style images(a list of images with
# the same style and art) and our own image, and get what we want
# a good example is given here: https://huggingface.co/docs/diffusers/en/using-diffusers/loading_adapters?tasks=image-to-image#ip-adapter

# lets do this and see it in action 
# uncomment this section to see how it looks
## we use ip_imgs3 for style and use a face image I used leon s.kennedy
# ip_img_face =  dfs.utils.load_image(f"{fldr}/leon_face_re2.jpg")
## lets create a separate ip_adapter here 
# text2image.load_ip_adapter("h94/IP-Adapter",
#                            subfolder="models",
#                            resume_download=True,
#                            # we use two models, the first to capture the overall style 
#                            # and the second one is used specifically for the face alteration
#                            weight_name=["ip-adapter-plus_sd15.bin",
#                                         "ip-adapter-plus-face_sd15.bin"])
# prompt = ["kodak portrait 4k"]
# ip_adapter_image = [ip_imgs3, ip_img_face]
## use a lot of style_images, and a bit of face image
# text2image.set_ip_adapter_scale([0.6,0.45])
#! another example with face only 
# give another example with face only
# lets first load our ip-adapter model, since we are after face, we use a face model 
# text2image.load_ip_adapter("h94/IP-Adapter", subfolder="models", resume_download=True,
#                            weight_name=["ip-adapter-plus-face_sd15.bin"])
# # lets load our image prompt 
# ip_image = dfs.utils.load_image(f"{fldr}/leon_face_re2.jpg")
# # since our negative prompt is a list, we send our prompt in a list as well
# prompt = ["A photo of a man holding a Glock 19, upper body, behind is a brick wall"]
# # now lets set our ip-adapter
# ip_adapter_image = ip_image
# text2image.set_ip_adapter_scale(0.7)
# # DDIMScheduler seems to give better results for face models
# # text2image.scheduler = dfs.DDIMScheduler.from_config(text2image.scheduler.config)
# width = 512
# height = 704
# print(torch.random.initial_seed())
# # generator = torch.Generator(device="cpu").manual_seed(128)
#
# we can also use ip-adapter with LCM Lora (which we will get to shortly) released in nov 2023
# to achieve “instant fine-tune” with custom images. Note that we need to load
# IP-Adapter weights before loading the LCM-Lora weights.
# the whole point of lcm-lora is to get an image in the least amount of steps (reach realtime image
# generation using difusion model. thats it. the rest of the example below is just us trying sth
# for fun! that is we can use our existing pipeline which uses sd1.5 and ip-adapter just fine but
# for the sake of it, we are going to use the herge-style here) 
#
# for a change lets use a stylized model this time, this is a finetuned model using dreambooth method
# we want to apply this style to our ip_adapter_image. this model size is around the same size of sd1.5
# note that to enable this style we have to use "herge_style" in our prompt otherwise the
# style wont take effect.
# 
# uncomment the following section to run the snippet! 
#
# model_id =  "sd-dreambooth-library/herge-style"
# # https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
# lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
# # before loading a new ip_adapter, lets unload our previous one up there
# # text2image.unload_ip_adapter()
# text2image = DiffusionPipeline.from_pretrained(model_id,
#                                resume_download=True, 
#                                torch_dtype= torch.float16,
#                                # cache_dir="./models"                                                  
#                                # disable the internal nsfw checker
#                                safety_checker=None,
#                                requires_safety_checker=False,).to("cuda")

# text2image.enable_attention_slicing("max")

# # now lets load our ip-adapter model
# text2image.load_ip_adapter("h94/IP-Adapter", 
#                            resume_download=True,
#                            subfolder="models", 
#                            weight_name="ip-adapter-plus_sd15.bin")
# # now lets load our lcm-lora weights
# # only when we load the lcm lora weights, the lcm scheduler works as intended, without the wieght
# # we dont get anywhere and the final result is just absymal! non-existent!
# text2image.load_lora_weights(lcm_lora_id)
# # lcmscheduler 
# text2image.scheduler = dfs.LCMScheduler.from_config(text2image.scheduler.config)
# # if we have low gpu vram and want to offload the models to cpu we can use
# # text2image.enable_model_cpu_offload()
# # the other option is to use enable_sequential_cpu_offload() which sequentially moves
# # models to the gpu, one at a time at forward() pass, and then moves them back to the cpu
# # this saves more vram but is slower compared to the previous method.
# # text2image.enable_sequential_cpu_offload()
# prompt = ["1girl herge_style"]
# # image taken from https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png
# ip_adapter_image = dfs.utils.load_image(f"{fldr}/knight_woman.png")
# # set the proper scale (try using 0.2 and 0.5 as well and see the result)
# text2image.set_ip_adapter_scale(0.47)
# # now we can have as few as 4 short steps to get the final image instead of 50!
# # this doesnt mean we acnt use larger numbers, it means we can get a result with
# # as few as 4-8 steps!. try this with 4,8,20 and 40 steps and see how it turns out
# num_inference_steps = 8
# # also note that specifying image height and width also directly affects our image generation
# # note that they must be divisible by 8!
# height=704
# width=512
# # ref: https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm_lora
# # note that we set guidance_scale=1.0, which disables classifer-free-guidance. 
# # This is because the LCM-LoRA is trained with guidance, so the batch size does 
# # not have to be doubled in this case. 
# # This leads to a faster inference time, with the drawback that negative prompts 
# # don’t have any effect on the denoising process.
# # we can also use guidance with LCM-LoRA, but due to the nature of training the 
# # model is very sensitve to the guidance_scale values, high values can lead to 
# # artifacts in the generated images. In huggingface team's experiments, they 
# # found that the best values are in the range of [1.0, 2.0].
# # remember to uncomment the other guidance_scale down below!
# guidance_scale = 1
# # but what is LCMLora? 
# LCM-LoRA stands for Latent Consistency Model - LoRA. It's a groundbreaking approach in the realm of 
# image synthesis³. LCM-LoRA is a product of LoRA distillation applied to Stable-Diffusion models, 
# including SD-V1.5, SSD-1B, and SDXL³. 
# This process allows for a significant reduction in memory consumption and enhances the quality of image
# generation³.
# LCM-LoRA was proposed in "LCM-LoRA: A universal Stable-Diffusion Acceleration Module" by Simian Luo, 
# Yiqin Tan, Suraj Patil, Daniel Gu, et al¹. It is a distilled consistency adapter for stable-diffusion-xl-base-1.0 
# that allows to reduce the number of inference steps to only between 2 - 8 steps¹. 
# LCM-LoRA can be used with any custom checkpoint model to speed up the image generation to as few as 
# four steps⁴. It can be used for various tasks such as text-to-image, image-to-image, and inpainting¹². 
# For example, to use LCM-LoRA for text-to-image task, you can load it with its base model 
# stabilityai/stable-diffusion-xl-base-1.0, change the scheduler to LCMScheduler, and reduce the number 
# of inference steps to just 2 to 8 steps¹. (checkout https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning which does this in 4 steps as well)
# Similarly, for image-to-image tasks, you can use it with the dreamshaper-7 model and the LCM-LoRA for 
# stable-diffusion-v1-5².
# Please note that the specific usage might require additional libraries such as the Hugging Face Diffusers 
# library¹². For detailed usage examples, it's recommended to check out the official LCM-LoRA docs¹².
# Source: Conversation with Bing, 2/16/2024
# (1) LCM LoRa: A Universal Stable-Diffusion Acceleration Module. https://lcmlorasd.com/.
# (2) latent-consistency/lcm-lora-sdxl · Hugging Face. https://huggingface.co/latent-consistency/lcm-lora-sdxl.
# (3) LCM-LoRA: High-speed Stable Diffusion - Stable Diffusion Art. https://stable-diffusion-art.com/lcm-lora/.
# (4) latent-consistency/lcm-lora-sdv1-5 · Hugging Face. https://huggingface.co/latent-consistency/lcm-lora-sdv1-5.
# (5) undefined. https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png.
# (6) undefined. https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png.
# (7) undefined. https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png.
# 
# ref https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm_lora
#
#
#
# guidance_scale is very important as it specifies how much our model should pay attention to the prompt
# The guidance_scale parameter controls how much the image generation process follows the text prompt closely.
# It can also be thought of as the "prompt strength".
# Basically the higher the guidance_scale, the more the generated images adhere to the given prompt.
# this means the model has less freedom to be random/creative!.
# conversely, a lower guidance_scale gives the model more freedom to generate diverse images,
# but these may not adhere as closely to the text prompt.
# It’s important to note that the guidance_scale should be set carefully:
# Setting it too high (e.g., 20) can result in images that only/strictly follow the prompt 
# but with worse image quality.
# Setting it too low (e.g., 1) basically ignores the text prompt and creates random images.
# usually the values from 7 to 12 are used.
# Using a scale up to 15 still produces results with little to no artifacts though it depends on the model
# The recommended guidance_scale value is typically between 7 and 9.
guidance_scale = 8.5
# more on this later, for now, it allows us to remove unwanted concepts in our image
# like extra limbs, ugliness, low quality, a specific artstyle, etc
# here is a negative prompt with the concepts I dont want in my result,
# try the result with and without this and see how much it affects the output (it affects alot)
# note that this is a rudemintary prompt, we can use much richer negative prompts obvioulsy but
# this should suffice for our demonstration purposes (you get the idea!)
negative_prompt = [
    "worst quality, low quality, ugly, low quality, extra limbs, bad anatomy, poorly rendered face, deformed,"
    " bad proportions, blurry cloned face, cropped, disfigured, duplicate, out of frame,"
    " extra arms, extra fingers, extra legs, fused fingers, gross proportions, long neck,"
    " lowres, malformed limbs, missing arms, missing legs, morbid, mutated hands, mutation, mutilated,"
    " poorly drawn face, poorly drawn hands, too many fingers, watermark, worst quality,"]*len(prompt) 
# we can generate multiple images for a single prompt and chose all or the best one if we wish
num_images_per_prompt = 1
# sidenote:
# by the way if we wanted to change the scheduler post pipeline creation we could do 
# text2image.scheduler = dfs.DDIMScheduler.from_config(text2image.scheduler.config)
# 
# To disable the safety check, we can set safety_checker=None, the
# the requires_safety_checker attribute is a boolean that indicates
# whether a safety checker is required for the pipeline. 
# If requires_safety_checker is set to True, the pipeline will raise a warning 
# if safety_checker is set to None
text2image.safety_checker=None
# text2image.requires_safety_checker = False
result = text2image(prompt=prompt, 
                    height=height, 
                    width=width, 
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps, 
                    guidance_scale=guidance_scale, 
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    output_type = 'pil',
                    generator=generator,
                    # use the image prompt adapter to add a style to our final image
                    ip_adapter_image=ip_adapter_image,
                    # If return_dict is `True`, StableDiffusionPipelineOutput is returned,
                    # otherwise a `tuple` is returned where the first element is a list with
                    # the generated images and the second element is a list of `bool`s 
                    # indicating whether the corresponding generated image contains
                    # "not-safe-for-work" (nsfw) content. the default is True
                    return_dict = True,
                    )

display_images(prompt, result)

# if you noticed, the faces are sometimes really ugly! to fix that, there are several methods 
# from finetuning, using proper positive and negative prompts, and separate models to fix faces
# we will get to them later on inshaalah.
# 
# TODO: talk about Stable Cascade that came out afew days ago(4 days ago) https://github.com/Stability-AI/StableCascade)! its power lies in its ability to follow
# the given prompt much better than the ordinary diffusion models we have seen so far! like dalle3!
#  
#%%
# ref https://huggingface.co/docs/diffusers/en/tutorials/basic_training
# 
# OK before we dive deeper, lets get to know diffusion models better and see how they work under
# the hood. here we are tyring to get a deeper understanding of how everytihng works and implement
# what these pipelines are hiding/abstracting away. 
# 
# a difffusion model consistes of several parts or components. what are these components you may aks?
# a text encoder (happens to be clip for sd1.5) becasue obviously we are dealing with text prompts
# 
# an autoencoder (vae) to take us from noise to the final image
# 
# a noise generator like a unet model to generate the noise given the input image. This noise is
# then added to the image to create a ‘noisy’ version of the image. The purpose of this component
# is to introduce variability into the model, which can help prevent overfitting and improve the
# models ability to generalize to new data. 
# 
# and finally a scheduler/sampler that creates the final image by repeatedly denoising the latent
# noise. This component controls the noise level at each step of the denoising process. 
# It determines how much noise to add or remove at each step. 
# The scheduler/sampler plays a crucial role in controlling the trade-off between the amount of 
# noise and the quality of the reconstructed image.
#
# lets expand on this a bit more:
# a diffusion model works by adding noise to the data and then learning to remove this noise 
# so it can recover the original data. This process is done iteratively, with the model making the 
# data less and less noisy in each step. The final result is a model that can generate new data that is similar to the training data.
# but why does denoising has to be done iteratively? 
# there are a few reasons why this is the case, 
# it so happens that jumping from pure noise to the perfect image is very hard, so instead its done
# in several steps, gradually removing noise to reach to the final result, its akin to an artist might
# start with a rough sketch and then gradually add details until the final image is complete. 
# Each iteration allows the model to refine its reconstruction, adding more and more detail with each
# step.
# by doing this the whole process becomes more stable than trying to denoise the image in one step.
# as each iteration allows the model to make small adjustments, which can help prevent it from making
# large errors that could result from trying to denoise the image all at once.
# In addition to that, training a model to denoise an image in one step is challenging, as it 
# requires the model to learn a complex mapping from the noisy image to the original image. 
# By breaking this task down into smaller steps, each of which involves denoising the image 
# a little bit, the learning problem becomes easier and the training process can be more efficient.
# note that this is solely due to the specifics of architecture design and not an inherent attribute
# for example the SDXL requires way fewer steps to generate a good image. other methods such as LCM-lora
# even lower this to 2-8 steps, basically removing the need for multiple iterations. but these require
# specific design choices we will get to later on. 
# for now it suffices to say that SDXL uses a technique called SDEdit and LCM-Lora is a distillation
# approach. we'll get to them later inshaallah
# Now we said our diffusion model consits of several part, like vae,text-encoder,scheduler and a noise generator
# this is not accuate though, a diffusion model by nature at its simplest form is composed of 
# a noise generator(unet) really and a scheduler for denoising. 
# the rest are really used for either conditioning the generation process (text-encoder) or aiding in
# lowering the process overhead. 
# what really happens under the hood at the simplest form is we want to generate images that resemble
# our training dataset. what do we do? we feed it noise, and expect the model to generate an image
# any image, this is called unconditional generation, the model just produces an outcome randomly.
# this is fine if our goal is to generate only one type of image, like only dogs, or cats, etc, but
# its very lacking becasue it would be very useful if we could have control on different aspects of
# the image generation, like the color, shape, form environment etc related to the images. this is 
# where the conditioning factor comes into play, by using the text-encoder, we effectively control
# and steer the generation process to toward where we want, creating images according to our dynamic needs!
# now before we get to the autoencoder part, lets see how our model is trained exactly. 
# we want to generate images, unconditional generation for now, what do we need to do? 
# our model needs something to start, we can use noise! pure noise, the model takes the noise matrix
# which has the same dimension as the imagesize we are interested in, and changes it to a prefect image!
# how do we make the model do that? the training is relatively straightfoward, we take an image from our
# dataset, add some noise to it and feed it to the model, we ask the model to give us back the added noise!
# why? for once, if we were to give the image intact, and ask the model to generate it, it would be an autoencoder
# it would only, duplicate our dataset, we want brand new images, so the raw image doesnt cut it for us. 
# so what do we do? we add noise to the image, and and use this instead, now the model either has to create 
# the image, which is not useful to us, or give us the noise. 
# now when we repeat this process many times, with varying amount of noise from no noise at all, to a lot
# of noise to the point the image is not visible anymore, the model learns about two things. where the actual
# image manifold lies and where the noises lie. why is this important, it is important becasue now we
# can feed the model a noise or noisy image, and ask it to give us the noise, the model tries its best to 
# detect what it identifies as noise, we subtract the input image from this noise, and get a new image, 
# now its still not there, theres still a lot of noise! so we keep doing this process a few more times, 
# and hopefully the model removes all the noise and gradually goes towards the manifold of images
# and generates the final noise-free image.
#! now we skiped a alot of details here, like the type of model we use, how we sample noise, the choice of
# noise and how it affects the output, choice of activation functions and their effect on result, etc
# well get to these details when we start implementing our diffusion model ourseleves, for now, lets
# briefly talk about some of the more important aspects here and leave the rest for our future encounter 
# inshaallah. 
# the type of model we are going to use is something called a U-Net model. U-Nets are typically used for 
# segmentation tasks while our usecase is a diffusion model. whats the difference you may ask? 
# The Standard U-Nets are designed for segmentation, where the input is a clean image and the output is a
# mask or segmentation of different image regions while diffusion models involve a different learning task: 
# they predict the noise added or removed at each time step to progressively refine an image. 
# This means, we need to incorporate some changes into our implementation of UNet that is specific to our 
# usecase. so what are these modifications we are talking about here. 
# The choice of activation functions used in the model, the depth for encoder/decoder parts, and the skip
# connections are afew that needs careful attention and needs to change. 
# for example, depending on our choice of noise range, Leaky ReLU, Sigmoid or SiLU can be used. 
# LeakyReLU is appropriate for handling negative noise, Sigmoid for strictly positive noise,
# and SELU offers improved gradient flow. 
# !(If our noise is strictly positive, a Sigmoid activation in the final layer ensures the output
# remains within the valid range (0-1). For negative noise, Leaky ReLU allows both positive and 
# negative values. SiLU can be beneficial in both encoders and decoders for improved learning and
# gradient flow.)
# Also while Skip Connections are essential for both Standard U-Nets and diffusion models, their 
# importance for preserving spatial information is crucial for accurate noise removal/image generation.
# in addition to what we discussed, Time step embedding and residual/gated connections are also part of 
# enhancements to our U-Net that we can add depending on our usecase. 
# 
# Also concerning Time Step Embedding, Diffusion models involve adding and removing noise at different
# time steps. Incorporating a time step embedding into the U-Net provides the model with knowledge about
# the current stage of the diffusion process. This can be implemented by concatenating a time step 
# representation or its features with the input or intermediate activations.
#
# And concerning Skip Connections, beyond their usual role in U-Nets, skip connections are crucial in
# diffusion models to preserve high-resolution features from the encoder. These features help in 
# accurately reconstructing the clean image during the denoising process. jus remember to make sure 
# that the skip connections have compatible dimensions for concatenation with the decoder outputs. 
#
# concerning Output and Loss Function, In denoising diffusion, the U-Net output should represent the 
# clean image. we need to adjust the final layer's number of channels to match the image channels 
# (e.g., 1 for grayscale, 3 for RGB).
# For loss function, we need to use a loss function appropriate for our diffusion goal. For denoising, 
# mean squared error (MSE) or L1 loss are common choices. we might need to adapt the loss function 
# depending on our specific task (e.g., image generation, inpainting).
#
# so to recap:
# for diffusion models, some modifications are essential to optimize performance and correctly address 
# the diffusion process:
# 1. Input Noise Characteristics:
# - Input Dimensions: Standard U-Nets often take clean images as input. In diffusion models, the input
#   is a noisy version of the target image. we need to adjust the input channels in the U-Net to 
#   accommodate the number of noise channels or dimensions based on our diffusion process.
# - Activation Functions: we must consider the noise range. If our noise is strictly positive, 
#   a Sigmoid activation in the final layer ensures the output remains within the valid range (0-1).
#   For negative noise, Leaky ReLU allows both positive and negative values. SELU can be beneficial 
#   in both encoders and decoders for improved learning and gradient flow.
# 2. Time Step Information:
# - Time Step Embedding: Diffusion models involve adding and removing noise at different time steps. 
#   Incorporating a time step embedding into the U-Net provides the model with knowledge about the 
#   current stage of the diffusion process. This can be implemented by concatenating a time step 
#   representation or its features with the input or intermediate activations.
# - Conditional U-Net: we need to explore using a conditional U-Net architecture, where the time step
#   information is directly fed into the U-Net layers, allowing the model to dynamically adapt its 
#   predictions based on the current stage.
# 3. Skip Connections:
# - Beyond their usual role in U-Nets, skip connections are crucial in diffusion models to preserve 
#   high-resolution features from the encoder. These features help in accurately reconstructing the clean image during the denoising process. Ensure that the skip connections have compatible dimensions for concatenation with the decoder outputs.
# 4. Output and Loss Function:
# - Output Layer: In denoising diffusion, the U-Net output should represent the clean image. 
#   we need to adjust the final layer's number of channels to match the image channels (e.g., 1 for grayscale, 3 for RGB).
# - Loss Function: Use a loss function appropriate for your diffusion goal. For denoising, mean squared error 
#   (MSE) or L1 loss are common choices. You might need to adapt the loss function depending on your specific task
#   (e.g., image generation, inpainting).
# Why are these modifications needed?
# Standard U-Nets are designed for segmentation, where the input is a clean image and the output is a mask or
#   segmentation of different image regions. Diffusion models involve a different learning task: predicting the 
#   noise added or removed at each time step to progressively refine an image. The modifications address these 
#   differences:
# - Input channels: Adapt to the noisy input representation.
# - Activation functions: Ensure valid predictions based on the noise range.
# - Time step information: Guide the model with contextual knowledge of the diffusion process.
# - Output and loss: Align with the specific goal of denoising or other diffusion tasks.
# 
#
# sidenote:
# in a typical diffusion model, noise is usually added to each channel of the input image, 
# so it might not seem necessary to explicitly change the number of channels in the U-Net. 
# However, there are a few nuances to consider:
# **1. Noise Representation:**
# - our diffusion model might not simply add random noise to each channel. 
#   Some methods add different types of noise (e.g., Gaussian, Laplacian) or add noise in a more 
#   structured way (e.g., channel-wise scaling, frequency-domain noise). In such cases, the input
#   to your U-Net might have additional channels representing these different noise components. 
#   You'll need to adjust the U-Net's input channels accordingly.
# **2. Diffusion Model Architecture:**
# - The diffusion model architecture itself might involve processing the added noise separately 
#   from the original image data. For example, some models pass the noisy image through a separate 
#   network specifically designed for denoising the added noise. In such cases, the U-Net might only
#   need to handle the original image channels, not the noise channels directly.
# **3. Conditional U-Nets:**
# - As mentioned earlier, we could explore using a conditional U-Net, where the time step information
#   (including noise characteristics) is directly fed into the U-Net layers. This might require changes 
#   to the U-Net structure to accommodate the additional input channels representing the noise information.
# **Therefore, although directly adding noise to each channel is common, the specific implementation of our
#   diffusion model and its noise representation might influence the number of channels needed in the U-Net.**
# **Key Takeaway:**
# Always analyze our specific diffusion model architecture and the way noise is incorporated to determine the 
# appropriate number of channels for your U-Net input. Don't simply assume it should match the original image 
# channels in every case.
#
#
# sidenote 2 : 
# In diffusion models, the choice between strictly positive noise and negative noise has important 
# implications for the training process and the resulting generated images. 
# Here's a breakdown of the key differences and considerations:
# **Strictly Positive Noise:**
# * **Definition:** All values added to the image at each diffusion step are non-negative, typically 
#     Gaussian noise with a zero mean and positive variance.
# * **Training and Inference:**
#     * Easier to implement computationally, as calculations don't need to handle negative values.
#     * Might require special activation functions in the U-Net (e.g., Sigmoid) to ensure predictions 
#       remain within the valid range.
# * **Generated Images:**
#     * Tend to be smoother and less sharp due to the non-negative noise distribution.
#     * Often lead to higher Inception Scores (IS) due to more natural-looking textures.
#     * Might lack detail and clarity compared to models using negative noise.
# * **When to Choose:**
#     * Prioritize smoothness and high IS scores.
#     * Prefer simpler training setup without handling negative values.
#     * Dealing with image domains where negative values don't make sense (e.g., grayscale images).
# **Negative Noise:**
# * **Definition:** Noise at each diffusion step can be both positive and negative, usually Gaussian noise 
#     with a zero mean and a non-zero variance.
# * **Training and Inference:**
#     * Requires more careful handling of negative values during calculations.
#     * U-Net activation functions need to accommodate both positive and negative predictions (e.g., Leaky ReLU).
# * **Generated Images:**
#     * Can be sharper and more detailed due to the wider range of noise values.
#     * Might yield lower IS scores as they can appear more "noisy" than strictly positive noise models.
#     * Often capture finer details and high-frequency information.
# * **When to Choose:**
#     * Aim for sharp and detailed images with potential trade-off in smoothness.
#     * Willing to invest in a slightly more complex training setup.
#     * Working with image domains where negative values are meaningful (e.g., natural images with shadows).
# **Additional Considerations:**
# * The best choice for you depends on your specific goals and requirements.
# * Experiment with both options and evaluate their performance on your dataset using relevant metrics.
# * Some diffusion models employ mixtures of positive and negative noise for more flexibility and potentially 
#   better results.
# * The effectiveness of each approach can also be influenced by the U-Net architecture and hyperparameters.
# I hope this comprehensive explanation helps you make an informed decision about choosing strictly positive
#   or negative noise for your diffusion model!
# 
# By incorporating these modifications, you can leverage the U-Net architecture effectively within your MNIST 
# diffusion model for improved performance. Remember to adapt and experiment based on your specific model details and requirements.
# this type of model also exibits an intersting characteristc. 
# !it preserves the spatial information of our image much better than normal feed forward models, 
# !unet starts off big and shrink in several stages, then it starts to upsample the featuremaps until 
# !we reach the initial size, in doing so, it allows the model to extract features at different resolutions
# !due to its architecture while allowing the featuremaps in later stages to recieve information form earlier
# !counterparts. 
# this is particularly useful in our diffusion models, where high-frequency information is dominated by 
# noise exponentially faster. unet models also utilize skip connections, which help in adding detail
# to images. These connections allow the model to bypass layers, which helps in preserving the spatial
# !resolution throughout the network and allows for more precise localization. 
# well see more information concerning this
# now since this is images that we are dealing with, the higher resolution, the images, the higher the
# processing cost gets, therefore the idea of making enhancements to lower said overhead comes into play
# this is where the autoencoder part comes into play. instead of using the full resolution image/noise
# to train our model, and later use, we simply use an autoencoder to compress the images and instead 
# work with a compressed representation of the original data, resulting in much faster computation and 
# less computation overhead. 
# to make this more intuitive lets implement one from scratch ourselves 
#%%
# we are going to use mnist dataset and create a diffusion model to generate digits for us
# lets import what we need
import sys,os,math,random
from pathlib import Path
import numpy as np
from tqdm import tqdm
# from tqdm.notebook import tqdm

import torch
# for pylance so we get autocomplete for submodules!
import torch.utils
import torch.utils.data

import torch.nn as nn
import torch.nn.functional as F
import torchvision
# for pylance intellisense to work properly!
import torchvision.utils
import torchvision.datasets.utils

import torchvision.transforms as tfms
import torchvision.datasets as dataset
import matplotlib.pyplot as plt

fldr="/media/hossein/SSD1/code_dl/"

print(f'{torch.__version__}')
print(f'{sys.version}')

# we need to have a dtaset 
# we need to have a unet model
# we need to have a scheduler to add noise and reverse it
# lets instantiate our dataset 
# before we create our dataset, we need to create our transformations 
# because we have to send one during dataset creating. 
# for transformation everything is straight forward, we resize our image to a given size
# for easier manipulation, then we can normalize it, first to tensor (which makes in 0-1
# and then we normalize it. if we are using other datasets, like cars, cifar, etc
# its usually normalized in -1 to 1 range instead, and the unet also need to take this
# into account) for now lets keep this simple. we later on use a more sophesticated example
# and use a different set of transformations.
transform=tfms.Compose([tfms.Resize(32), tfms.ToTensor(),
                        tfms.Normalize((0.5,), (0.5,)),
                        #tfms.Normalize((0.1307,), (0.3081,)) # --> this caused the noise! 
                        # like the network wouldnt learn a thing! even at epoch 2100 the sampling
                        # was pure noise!
                        # tfms.Lambda(lambda x: x*2-1)
                        ])

dt_train = dataset.MNIST(f"{fldr}/data",train=True,download=True, transform=transform)
dt_val = dataset.MNIST(f"{fldr}/data",train=False,download=True, transform=transform)
# since we dont need a validation set, we can use all the images,
# to use both train and validation set images we can concatenate the two datasets
dt_mnist = torch.utils.data.ConcatDataset([dt_train, dt_val])
batch_size = 256
num_workers = 8
dataloader = torch.utils.data.DataLoader(dt_mnist, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True, drop_last=True)
# now lets create our unet architecture 
# we start off with 28x28 size, downsample it until we reach
# a small featuremap,then start to upsample it to reach 28x28
#
# our unet can be viewed an encoder/decoder model
# the encoder part being the first section where 
# the input is shrunk and the decoder part is where
# its upsampled to match the initial input size in
# the begining. you might also see terms such as middle
# which refers to the bottom part of the model right
# after the encoder and before the decoder. 
# before we create the model as a monolithic class, lets
# create separate modules and make our life easier

# to create layers for encoder
# first of all we need positional encoding to retain position info and differentiate 
# between different timesteps, so lets first create a sinusoidal positional encoding
# to encode our timesteps

import numpy as np
def sin_pos_enc_v2(pos, embd_d):
    pos_vec = torch.zeros(embd_d)
    div_term = torch.exp(-torch.arange(0, embd_d, 2) * (torch.log(torch.tensor([10_000])) / embd_d))
    pos_vec[0::2] = torch.sin(pos * div_term)
    pos_vec[1::2] = torch.cos(pos * div_term)
    return pos_vec

def pos_enc(pos, embd_size):
    div_term = np.exp(-np.arange(0, embd_size, 2 )) * (np.log(10_000.0)/embd_size)
    rep = np.zeros(embd_size)
    rep[0::2] = np.sin(pos * div_term)
    rep[1::2] = np.cos(pos * div_term)
    return rep

sin_pos_enc_v2(1, 32)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embd_size=32,device='cpu') -> None:
        super().__init__()
        self.embd_size = embd_size
        self.device = device
    
    @torch.no_grad()
    def forward(self, positions):
        pos_vec = torch.zeros(size=(positions.size(0),self.embd_size),device=self.device)
        div_term = torch.exp(-torch.arange(0, self.embd_size, 2, device=self.device) * (torch.log(torch.tensor([10_000], device=self.device)) / self.embd_size))
        positions = positions[:,None]
        pos_vec[:,0::2] = torch.sin(positions * div_term)
        pos_vec[:,1::2] = torch.cos(positions * div_term)
        return pos_vec

device = 'cpu'    
pos_enc_model = SinusoidalPositionalEncoding(embd_size=32, device=device)
print(pos_enc_model(torch.tensor([1],device=device)))
t = torch.randint(0,2,size=(2,),device=device)
print(f'{t=}')
print(f'{pos_enc_model(t)=}')

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, timestep_embd_size=32, use_bn=True, act=nn.SiLU(),device='cpu'):
        super().__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding, bias=False,device=device)
        self.bn = nn.BatchNorm2d(num_features=out_channels,device=device)
        self.act = act
        
        # we need our sinusoidal positional encoding to be incorporated in the model
        # so we create a small mlp to do this so that we get a higher representation out of our
        # sinusoidal positional embeddings
        self.time_embd_size = timestep_embd_size
        self.timemlp = torch.nn.Sequential(SinusoidalPositionalEncoding(self.time_embd_size,device=device),
                                           nn.Linear(self.time_embd_size,out_channels,device=device),
                                           nn.ReLU())
        
        # if in_channels is not equal to out_channels, or if stride is not 1, 
        # then the dimensions of x and the output of the convolutional layer 
        # will not match, and we'll get an error when we try to add them together.
        # to create a skip/resudial connection. so in such cases, we use a 1x1 conv
        # instead so the outputs match
        if in_channels != out_channels or stride != 1:
            # we can use a conv1x1 or simply use a pooling operation to 
            # downsample the input to match it with the output of our block
            # but since the in_channels and out_channels are different, we 
            # need to use conv layer instead of a simple pooling
            self.match_dimensions = nn.Conv2d(in_channels,
                                              out_channels, 
                                              kernel_size=1, 
                                              stride=stride, 
                                              bias=False,device=device)
        else:
            self.match_dimensions = None

    def forward(self, x, t)->torch.Tensor:
        identity = x
        # get the time_embeddings
        timestep_embeddings = self.timemlp(t)
        if self.match_dimensions is not None:
            identity = self.match_dimensions(x)
        out = self.conv(x)
        
        # make them compatible. the ... (ellipsis) is a shortcut that means 
        # "all preceding dimensions" and None adds a new dimension. 
        # so t[..., None, None] will add two new dimensions at the end of the tensor.
        # we could use other methods(unsqueeze, expand, etc), but this is consise and elegent! 
        out += timestep_embeddings[...,None,None] 
        if self.use_bn:
            out = self.bn(out)
        out = self.act(out)
        return out + identity

class DeconvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, timestep_embd_size=32, act=nn.SiLU(),device='cpu'):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,device=device)
        self.bn = nn.BatchNorm2d(num_features=out_channels,device=device)
        self.act = act
        # we need our sinusoidal positional encoding to be incorporated in the model
        # so we create a small mlp to do this so that we get a higher representation out of our
        # sinusoidal positional embeddings
        self.time_embd_size = timestep_embd_size
        self.timemlp = torch.nn.Sequential(SinusoidalPositionalEncoding(self.time_embd_size,device=device),
                                           nn.Linear(self.time_embd_size,out_channels,device=device),
                                           nn.SiLU())
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels=out_channels,
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            device=device),
                                  nn.SiLU())
        if in_channels != out_channels or stride != 1:
            # like our convbnact, here we need to use a deconv which adjust the spatial
            # resolution and also fixes the discrepency in channels. 
            # we can use a bn after this, but for now lets not do it!
            self.match_dimensions = nn.ConvTranspose2d(in_channels,
                                                       out_channels, 
                                                       kernel_size=2, 
                                                       stride=2, 
                                                       bias=False,device=device)
            self.bn2 = nn.BatchNorm2d(out_channels,device=device)
        else:
            self.match_dimensions = None
    
    def forward(self, x, t)->torch.Tensor:
        identity = x 
        # get the time_embeddings
        # with torch.device(t.device):
        timestep_embeddings = self.timemlp(t)
        
        if self.match_dimensions is not None:
            identity = self.match_dimensions(x)
            # identity = self.bn2(identity)
        output = self.conv(self.act(self.bn(self.deconv(x)))) 
        
        # print(f'output.shape={tuple(output.shape)} {timestep_embeddings.shape=}')
        output += timestep_embeddings[...,None,None]
        
        return output + identity

device='cuda'
c = ConvBnAct(1, 64, kernel_size=3, stride=1, padding=1, timestep_embd_size=32,device=device)
d = DeconvBnAct(64, 1, kernel_size=2, stride=1, padding=1, timestep_embd_size=32,act=nn.SiLU(),device=device)
x,_ = next(iter(dataloader))
t = torch.randint(0,200, size=(batch_size,)).long()
x2 = torch.randn(size=(batch_size,64,2,2))
x,t,x2 = tuple(t.to(device) for t in (x,t,x2))
output = c.forward(x,t)
output2 = d.forward(x2,t)
print(f'{output.shape=}')
print(f'{output2.shape=}')

class Unet(nn.Module):
    
    def __init__(self, in_channel=1, initial_fmap=64, time_embd_size=32,device='cuda'):
        super().__init__()
        self.device = device
        # we have two sections in the first section/part/encoder 
        # we shrink the input, its a series of conv/bn/relu layers
        # then we reverse this, i.e. we start upsampling it till we
        # reach the initial image output
        # we define the first layer of unet normally and then use for
        # loop to create the rest of the encoder module. 
        self.conv_in = nn.Conv2d(in_channels=in_channel,out_channels=initial_fmap,
                                 kernel_size=3,stride=1,padding=1, device=device)
        self.relu = nn.SiLU()
        self.bn_in = nn.BatchNorm2d(num_features=initial_fmap,device=device)
        # we need our sinusoidal positional encoding to be incorporated in the model
        # so we create a small mlp so that we get a higher representation out of our
        # sinusoidal positional embeddings
        # self.timemlp = torch.nn.Sequential(SinusoidalPositionalEncoding(time_embd_size),
        #                                    nn.Linear(time_embd_size,time_embd_size),
        #                                    nn.ReLU())
        # we also need a transformation involving timesteps in every layer of our encoder
        # and decoders so lets create a block for them as well. we use this block
        # to combine the image features and the timestep positional embedding features
        # the output of this block gets fed to our normal layers in the encoder/decoders
        # so they work with both features merged, not just image features
        #! write the block? or add the timestep to the convbnact module? 
        self.encoder = nn.ModuleList()
        # we are dealing with mnist dataset, images are 28x28 but to make things easier
        # we resize it to 32x32 before we feed it to our network
        fmaps = initial_fmap
        for i in range(4):
            self.encoder.append(ConvBnAct(fmaps,fmaps*2,stride=2,padding=1,
                                          timestep_embd_size=time_embd_size,device=device,act=nn.SiLU()))
            fmaps*=2
        # print(self.encoder)
        # now lets build our decoder/upsampler part
        # we will use upsample2d+ a convlayer
        self.decoder = nn.ModuleList()
        for i in range(4):
            self.decoder.append(DeconvBnAct(fmaps, fmaps//2, kernel_size=2, stride=2,padding=0,
                                            timestep_embd_size=time_embd_size,device=device,act=nn.SiLU()))
            fmaps//=2

        # print(self.decoder)
        # now for the final output layer 
        self.conv2 = nn.Conv2d(fmaps,in_channel, kernel_size=1,stride=1,padding=0,device=device)
        # incase we used positive noise
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x,t):
        # first conv layer
        out = self.relu(self.bn_in(self.conv_in(x)))
        
        skip_connections = []
        for l in self.encoder:
            out = l(out,t)
            skip_connections.append(out)
            # print(f'{out.shape=}')
        
        for l in self.decoder:
            out = l(out+skip_connections.pop(),t)
            # print(f'{out.shape=}')
        
        out = self.conv2(out)
        # changing actf rom sigmoid to tanh, decreased the loss from 0.98 to 0.48!!
        out = F.tanh(out)
        return out

device = 'cpu'
model = Unet(device=device)
# model.to(device)
x = torch.randn(size=(1,1,32,32))
t = torch.randint(0,200, size=(batch_size,)).long()
imgs, _ = next(iter(dataloader))
imgs,x,t = tuple(t.to(device) for t in (imgs,x,t))
print(f'{model(imgs,t).shape=}')

def show_image(imgs_tensor, title=''):
    plt.imshow(torchvision.utils.make_grid(imgs_tensor).permute(1,2,0))
    plt.title(title)
    plt.show()

show_image(imgs)
# now https://www.youtube.com/watch?v=a4Yfz2FxXiY&t=298s
# https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=16612s

# refs: https://github.com/Jmkernes/Diffusion/blob/main/diffusion/ddpm/tutorial.md
# https://daviddmc.github.io/blog/2020/DDPM/

# the forward process is fairly easy, 
# all that needs to be done is to add noise
# to the input image, feed it to our unet
# model, and then our models job is to predict 
# the noise from the actual image. 
# this part is done by a noise scheduler
# in the paper the equation describing this
# process is given as : 
# q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})
# This equation represents a conditional probability distribution 
# over a sequence \(x_{1:T}\) given an initial state \(x_0\). 
# The distribution is factorized into a product of conditional probabilities,
# each dependent on the previous state in the sequence. 
# This is a common form for Markov chains and hidden Markov models.
# this means our diffusion model is also a type of markov chain, in fact 
# a diffusion probabilistic model is a parameterized Markov chain
# trained using variational inference to produce samples matching
# the data after finite time.(more https://daviddmc.github.io/blog/2020/DDPM/)
# 
# ref from https://www.youtube.com/watch?v=a4Yfz2FxXiY
# the equation here describes the forward process(the equation represents the noise process),
# where the model generates a sequence of states starting from a data point
# (x_0 which is our initial input image) and applying a series of noise transformations.
# Here, (q(x_t|x_{t-1})) is a Gaussian distribution with mean (\sqrt{1-\beta_t}x_{t-1})
# and covariance (\beta_t I), where (\beta_t) is either a learned parameter or a constant(hyperparameter).
# basically the amount of noise we add to an image depends on the previous image
# and the way the noise is sampled is dervided from the following formula: 
# q(x_t|x_{t-1}):=\mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
# its a conditional guassian distribution with a mean that depends on the previous image
# and a specific variance. in this equation N(x_t; sqrt(1-β_t)x_t-1, β_t.I)
# x_t is the outupt, sqrt(1-β_t).x_t-1) is the mean (of the distribution), and beta_t.I is the variance (fixed in our case)
# the sequence of betas are called variance schedule they describe how much noise we want
# to add in each of the time steps,x_t-1 is the previous less noisy image,which means
# the mean of our distribution is exactly the previous image multiplied by this term
# that depends on the variance schedule beta. the variance of this normal distribution
# is fixed to beta multiplied by the Identity.
# to get a better intuition about beta, lets imagin this
# lets pick a pixel out of an image of a car for example, this pixel has 3 channels for
# rgb, and the range of values is between 0-255 or normalized between -1 and 1 for example,
# lets assume our picked pixel has these values (1,-1,-1) (suppose its normalized between -1 and 1)
# this means, its a red pixel, (the red channel is 1, the maximum value, and the other two channels
# are -1, the minimum value, signifying they are no green or blue colors! so its a red pixel!)
# based on our equation just now, the distribution of our next image is now described by 
# this mean and variance (i.e. N(sqrt(1-βₜ)xₜ₋₁), βₜ.I) ) that means the value of
# our red pixel multiplied by square root of 1 minus beta_t (sqrt(1-βₜ)) is exactly 
# the mean of our distribution, (i.e. μ = sqrt(1-βₜ)xₜ₋₁ and σ = βₜ.I. 
# depending on our noise level βₜ, suppose it could be something like ~0.99 for example, 
# the variance is fixed to βₜ, so if we choose a large number, it means, not only the pixel
# distribution is wider now, but also more shifted, which results in a more corrupted image. 
# also, when sampling from this distribution, we will consequently endup with more noise,
# eventually β controls how fast we converge towards a mean of 0 which corrosponds to standard
# guassian distribution. (try different beta and see how it controls this)
#(see image ./Diffusion_model_mean_var_example_1.png and ./Diffusion_model_mean_large_beta_var.png)
# the important thing here is to add the right amount of noise, such that we arrive at an isotropic 
# distribution with a mean of 0 and fixed variance in all direction otherwise the sampling later 
# will not work. this simply means we dont wnat to add too few noise or too much noise!
# in order to have a too noisy image too early! there are different schedulings for that
# in our case, we would add the noise linearly, but there are many more variants, and ways to do this
# use quadratics, cosine, sigmoidal.
# it turns out, in practice, the noise is not added sequentially, becasue the sum of guassians is
# still a gaussian distribution, therefore we can directly calculate the noisy version of an image
# for a specific timestep t and thats without iterating over its predecessors.
# so basically based on the initial image x_0 we can calcualte its noisy version for any abitrart
# timestep t. 
# to do this we need to pre-calculate the closed form of the mean and variance based on the cumulative
# variance schedules.
# that means 
# q(xₜ|x0) = N(xₜ, sqrt(α⁻)x0, (1-α⁻)I)
# lets see an example of what this means, lets imagine we have 200 steps in our diffusion process,
# the variance schedule beta(β) tells us how much noise we want to add in each time steps.
# we linearly increase it until we reach a maximum value of 0.02. if we wouldnt increase it at all,
# it would take for ever to endup with pure noise (full noise image). therefore the authors
# defined a new term alpha(α) which is simply (1-β), we can think of it as, how much information
# we get to keep about an image when transitioning to another/next image.
# β₁ = 0.0001, β₂ = 0.0002, β₃ = 0.0003, β₄ = 0.0004, .... β₂₀₀ = 0.02
# α₁ = 0.9999, α₂ = 0.9998, α₃ = 0.9997, α₄ = 0.9996, .... α₂₀₀ = 0.98
# 
# cumulative products of alpha results in alpha overline:
#  ̅α₁=0.9999,   ̅α₂ =0.9997,  ̅α₃ =0.9994,  ̅α₄ =0.9990, ....  ̅α₂₀₀ =0.1322 
# by calculating the cumulative product of these alpha terms, we get a new term called ̅α 
# (alpha overline).
# with this we can specify a new distribution that allows us to sample for a specific
# timestep.(more details look at url: )
# consequently for training, we can now simply sample a timestep t and pass the nosified version of it to the model 
# this makes the training much more smoother and easier compared to sequentially iterating over
# the same image .
# sidenote: 
# I wrote ̅α like \overbar and then \alpha , by pressing space after each of them their unicode symbol
# is written. (Fast Unicode Math Characters plugin is installed by theway)
# 
# now lets implement our noise scheduler
# we say scheduler, becasue we specify at different steps/schedules so to speak to add noise
# and not in once go. so the first step is to create our betas 
# we use a simple linear interpolation to create our betas. 
def create_betas(start=0.0001, end=0.02, timesteps=200, device='cpu'):
    return torch.linspace(start=start, end=end, steps=timesteps, device=device)
# next lets calculate the closed form of mean and variance based on the cumulatie varianec schedules
# basically calculate alpha overline( ̅α )
# first lets calculate alpha, for that we need betas, since alphas =1-betas
betas = create_betas(timesteps=400)
alphas = 1.0-betas
# now lets calculate the alpha overline ( ̅α )
alphas_cumprod = torch.cumprod(alphas, dim=0)
# now √̅α
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# now √1-̅α 
sqrt_one_minus_alphas_cumprod = torch.sqrt(1-alphas_cumprod)
# becasue we need them both Lₛᵢₘₚₗₑ(θ):=Eₜ,ₓ₀,ϵ [‖ ϵ-ϵ₀(√̅αₜX₀ + √1-̅αₜϵ,t) ‖²₂], the better text form is available https://daviddmc.github.io/blog/2020/DDPM/ in the training and sampling section
# so we need to multiply alphas_cumprod for timestep t by x_0 and sqrt_1_minus_alphas_prod
# by our noise at timestep t. 
# so we need to grab alphas_cumprod for a specific timestep t, lets do that 
def get_alphas_cumprod_t(alphas_cumprod_values:torch.Tensor, timestep:torch.Tensor, x0_shape:torch.Size):
    batchsize = timestep.shape[0]
    # note the tensor.gather() function expects the second argument to be a tensor 
    # that specifies the indices of elements to gather. 
    # The dimension of the output tensor is the same as the dimension of the index tensor (which is timestep here).
    alphas_cumprod_t = alphas_cumprod_values.gather(-1,timestep.cpu())
    # before we return, we make sure the shape is compatible with the input image(s) x0.
    # so that when later on we mutiply them together everything works out. 
    # e.g. for a single image x0 it'd be torch.Size([1, 1, 1]) instead of torch.Size([1])
    # and for a batch of images (x0 is a batch of images), the final shape would be torch.Size([1, 1, 1, 1])
    # len(x0_shape) simply tells us how many dimensions our x0 has, does it have 3 ( a single image) or 4(a batch of images)
    # we decrease 1, since we added the batch at the begining, so (1,) is then only repeated one time less
    # and later we unpack this new tuple (which is (1,1) or (1,1,1) depending on whether x0 has a batch dim or not)
    # and this makes up our final shape! we also could do alphas_cumprod_t[..., None,None,None] etc
    shape=(batchsize, *( (1,) * (len(x0_shape)-1) ))
    return alphas_cumprod_t.reshape(shape)
#
# now we can calculate the forward diffusion process 
def forward_diffusion(x0:torch.Tensor, t:torch.Tensor, sqrt_alphas_cumprod:torch.Tensor, sqrt_one_minus_alphas_cumprod:torch.Tensor, device:str|torch.device='cpu'):
    # first lets create our noise 
    noise = torch.randn_like(x0)
    # now lets get alphas_cumprod for timestep t, since we are using its sqrt version we 
    # use the sqrt versions instead 
    sqrt_alphas_cumprod_t = get_alphas_cumprod_t(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_alphas_cumprod_t(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    # now lets calculate the output which is mean + variance 
    # we also return the noise!
    # sidenote: since torch 2.0.0 we can use torch.device as a context manager!
    with torch.device(device):
        output = (sqrt_alphas_cumprod_t * x0) + (sqrt_one_minus_alphas_cumprod_t * noise), noise
    return output
#
# now lets test our forward diffision model for now
# lets grab a few images
imgs,_ = next(iter(dataloader))
show_image(imgs,'test')
# now lets determine how many steps we want until we face full noise!
# try 300 or 100 and see what happens
# i tried 400, and it seemed nothing worked! though loss was very low, so it
# might have been the sampling? need to test this more!around 200 epochs, we should
# have initial digits formed, but previously with 400 it was pure noise even at 1900 epoch!
# ok more timesteps, results in lower loss, the thing is I needed to add more steps to sapling
# likespecify  20 for num_imgs otherwise it seems we only get noise
timesteps_t = 400
# number of steps we want to visualize the transition of noisification
num_imgs_for_visualization = 10
# now lets determine the step size 
step_size = timesteps_t//num_imgs_for_visualization
device = 'cpu'
plt.figure(figsize=(128,64))
# lets create a dictionary to store images transitions for later visualization
imdic = {i:[] for i in range(imgs.size(0))}
# now lets apply noise to our image, or in other words run our diffusion's forward() pass
for idx in range(0, timesteps_t, step_size):
    # lets convert our t into a tensor 
    t = torch.tensor([idx]).long()
    # now lets do a forward
    imgs_output, noise = forward_diffusion(imgs, t, sqrt_alphas_cumprod,
                                      sqrt_one_minus_alphas_cumprod, device)
    # plt.subplot(nrows, ncols, index)
    plt.subplot(1, num_imgs_for_visualization+1, (idx//step_size)+1)
    plt.axis("off")
    # show the first image only
    # plt.imshow(imgs_output.permute(0,2,3,1).numpy()[0])
    # if we want we can display all the batch but we
    # instead save them to view them better individually later
    # plt.imshow(torchvision.utils.make_grid(imgs_output).permute(1,2,0).numpy())
    # or we can simply treat the batch as one big image (stack them)
    ims = imgs_output.permute(0,2,3,1)
    img_rows = []
    # how many images do we want in each row, lets create equal rows/cols
    # i assume the batchsize are 2^sth!
    ncol = int(np.sqrt(ims.size(0)))
    # print(f'{ncol=}')
    for i in range(0, ims.size(0), ncol):
        # grab ncol images at a time from our batch
        img_row = ims[i:i+ncol]
        # concatenate them along the column, so we get a row of images
        img_row = torch.cat(img_row.chunk(ncol, dim=0), dim=2).reshape(32, -1)
        # print(f'{i=} {img_row.shape=}')
        # store them to later stack them and get a full image
        img_rows.append(img_row)
    # stack the images along the height and get our final image
    img_grid = torch.cat(img_rows,dim=0)
    plt.imshow(img_grid)
    plt.title(f'{idx}')
    # to save the images for visualizing the whole batch later
    for i, img in enumerate(imgs_output):
        imdic[i].append(img.permute(1,2,0).numpy())
plt.show()
#
# to display all the images in our batch individually
# for k, imgs in imdic.items():
#     plt.figure(figsize=(16,12))
#     for i in range(len(imgs)):    
#         plt.subplot(1, len(imgs)+1, i+1)
#         plt.axis("off")
#         plt.imshow(imgs[i])
# plt.show()
#
# now lets carry on - Parameterized backward pass (reverse pass)
# now we get to the unet part of our model. unet as we said previously is mainly used
# for semantice segmentation tasks, so the output matches the input size, this makes it
# very good for images, and especially for our work becasue we feed our noisy image and
# get the noise back from the unet(predict the noise).
# since our beta is fixed, (variance is fixed), we will only generate 1 value per pixel,
# and that means the model learns the mean of the gaussian distribution of images, this
# is also called denoising score matching.
# note that there were experiments with predicting the image mean instead of the noise mean
# and both approaches seem to work. 
# one important thing to note is that, we need to tell the model which timestep we are in
# becasue the model always uses the same shared weights for each input, no matter if its 
# timestep 45 or 8! we'll explain about this in more detail in a moment
# the reverse process can be formulated mathimatically like this :
# p₀(X_T) = Ν(xₜ;0;I) (p_theta(X_T) = N(x_t;0;I) )
# which means, we start in X_t with gaussian noise with zero mean and variance of 1 (unit variance)
# then in a sequence the transition from one latent to the next is predicted.
# this makes the model to learn the probablity density of an earlier timestep, given the current
# timestep.
# p_theta(x_0:T) = p(x_T) πᵀₜ₌₁ p_theta(x_t-1|x_t)
# as said previously, during training we just randomly sample timesteps and dont go through
# sequence at sampling time.
# however we need to iterate from pure noise from xt to x0 which is the final image.
# The density p is predicted by the gaussian noise distribution in the image. 
# in order to get the actual image of timestep xt-1 we have to subtract this predicted noise
#! from the image xt during sampling. x_t-1 = x_t-noise is a rough form of it
#! see the formula in paper
#
# now how do we use timestep in our model and involve it in the process? 
# we know that our neural network has shared parameters across time (i.e. different execution
# with different values, all are dealing with the same shared weights) which means it cant 
# distinuish between different timesteps. so it means, it needs to somehow filter out images
# with very different noise intensities, to this end, the authors used the sinusoidal positional embedding
# to circumvent/fix/address this issue. its a neat idea to encode discrete positional information
# like sequence steps. we talked about positional embedding in detail in our
# transformer architecture section please head over there and read about it.
# so thats why we added the positional emebedding to our unet
# 
#
# now for loss, the loss for diffusion models are optimized with the variational lower bound
# like how its done in vaes.however as the authors briged the connection to dneoising score matching
# they propose an alternative formulation that is equivalent to using the variational inference.
# to make it shore the final loss function is defined by this(below):
# Lₛᵢₘₚₗₑ(θ):=Eₜ,ₓ₀,ϵ [‖ ϵ-ϵ₀(Xₜ,t) ‖²₂] (ϵ is the added noise and ϵ₀(Xₜ,t) is the predicted noise here)
# simply means we calulate l2 distance of the predicted noise and actual noise in the imaghe.
# this loss function is quite easy but there are quite some derivations and considerations to arrive at
# this simple term which I couldnt include in this handson video, it is however highly recommended
# to look into some of the literature to get a better understanding.
# therefore the loss function for that is pretty straighforward! its l1/l2 loss or mse! mean sequre error
# between the forward diffusion images and the noise generated (timestep is required)
def calculate_loss(model, x_0, t, device):
    imgs_noisy, noise = forward_diffusion(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
    noise_predicted = model(imgs_noisy, t)
    #! test with mse, and others as well
    return torch.nn.functional.l1_loss(noise, noise_predicted)

# now before going to the training, we need to implement the sampling part
# so here it is 
@torch.no_grad()
def sample_timesteps(unet_model,x,t,device='cpu'):
    unet_model.to(device)
    # in order to do the sampling we need 3 more terms which are as follows 
    # here we are padding the alphas_cumprod at the begining with the value of -1.0.
    # This padded tensor is used in the calculation of the posterior variance.
    alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1,0),value=-1.0).to(device)
    # we are calculating the square root of the reciprocal of the `alpha` values. 
    # recall that these `alpha` values represent the variance reduction per time step in 
    # our diffusion model.
    # The `sqrt_recip_alphas` tensor is used in the calculation of the mean of the 
    # posterior distribution.
    sqrt_recip_alphas = torch.sqrt(1.0/alphas).to(device)
    # we are calculating the variance of the posterior distribution at each time step. 
    # # The `alphas_cumprod` and `alphas_cumprod_prev` tensors represent the 
    # cumulative product of the `alpha` values up to the current and previous time step,
    # respectively.
    posterior_variance = betas.to(device) * (1.0 - alphas_cumprod_prev.to(device)) / (1.0 - alphas_cumprod.to(device)) 
    # These calculations are part of the reverse process of the diffusion model, 
    # where the model gradually denoises an image starting from pure noise⁷. 
    # The `alphas_cumprod_prev`, `sqrt_recip_alphas`, and `posterior_variance` are 
    # used in the calculation of the Gaussian distribution from which the denoised 
    # image is sampled at each time step⁷.
    # 
    # Source: Conversation with Bing, 2/26/2024
    # (1) Denoising Diffusion Probabilistic Model - Keras. https://keras.io/examples/generative/ddpm/.
    # (2) Bayesian inversion of a diffusion model with application to biology. https://link.springer.com/article/10.1007/s00285-021-01621-2.
    # (3) A Diffusion Model from Scratch | by Amir Behbahanian - Medium. https://medium.com/@amir.behbahanian/a-diffusion-model-from-scratch-cf1131988e78.
    # (4) Chapter 5. Bayesian Statistics - Brown University. https://www.dam.brown.edu/people/huiwang/classes/am166/Ch5.pdf.
    # (5) How diffusion models work: the math from scratch | AI Summer. https://theaisummer.com/diffusion-models/.
    # (6) 1 On the Design Fundamentals of Diffusion Models: A Survey - arXiv.org. https://arxiv.org/pdf/2306.04542.pdf.
    # (7) [2107.00630] Variational Diffusion Models - arXiv.org. https://arxiv.org/abs/2107.00630.
    # (8) latent-diffusion/ldm/models/diffusion/ddim.py · multimodalart .... https://huggingface.co/spaces/multimodalart/latentdiffusion/blob/8c6eab45567a29aee245e0ecfd6b87c0c41fefbf/latent-diffusion/ldm/models/diffusion/ddim.py.
    # (9) [Bug]: alphas_cumprod are downcasted to half precision during model .... https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/14071.
    # (10) The Annotated Diffusion Model - Google Colab. https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb.
    # (11) Three Stable Diffusion Training Losses: x0, epsilon, and v ... - Medium. https://medium.com/@zljdanceholic/three-stable-diffusion-training-losses-x0-epsilon-and-v-prediction-126de920eb73.
        
    # get the betas for current timestep
    betas_t = get_alphas_cumprod_t(betas, t, x.shape).to(device)
    # get the sqrt_one_minus_alphas_cumprod for current timestep as well
    sqrt_one_minus_alphas_cumprod_t = get_alphas_cumprod_t(sqrt_one_minus_alphas_cumprod, t, x.shape).to(device)
    # also get the sqrt_recip_alphas for current timestep
    sqrt_recip_alphas_t = get_alphas_cumprod_t(sqrt_recip_alphas.cpu(), t, x.shape).to(device)
    
    # now call model for current image - noise_prediction 
    x=x.to(device)
    predicted_noise = unet_model(x,t)
    model_mean =  sqrt_recip_alphas_t * (x - betas_t*predicted_noise/sqrt_one_minus_alphas_cumprod_t)
    # now get the posterior variance for the current timestep as well
    posterior_variance_t = get_alphas_cumprod_t(posterior_variance.cpu(),t,x.shape).to(device)
    
    if t==0:
        return model_mean.to(device)
    else:
        noise = torch.randn_like(x).to(device)
        return model_mean + torch.sqrt(posterior_variance_t)*noise
    
output = sample_timesteps(model, imgs, torch.tensor([100]))
# model = Unet()
# x = torch.randn(size=(64,1,32,32))
# output = model(imgs)
print(f'{output.shape=}')
show_image(output)

# now for visualization
def create_image_from_batch(imgs_output:torch.Tensor)->torch.Tensor:
    ims = imgs_output.permute(0,2,3,1).detach().cpu()
    img_rows = []
    # how many images do we want in each row
    ncol = int(np.sqrt(ims.size(0)))
    for i in range(0, ims.size(0), ncol):
        # grab ncol images at a time from our batch
        img_row = ims[i:i+ncol]
        # concatenate them along the column, so we get a row of images
        img_row = torch.cat(img_row.chunk(ncol, dim=0), dim=2).reshape(32, -1)
        # print(f'{i=} {img_row.shape=}')
        # store them to later stack them and get a full image
        img_rows.append(img_row)
    # stack the images along the height and get our final image
    img_grid = torch.cat(img_rows,dim=0)
    return img_grid

@torch.no_grad()
def sample_plot_image(model,in_channel=1,num_images = 20,msg='',device='cpu'):
    # Sample noise
    img_size = 32
    img = torch.randn((batch_size, in_channel, img_size, img_size), device=device)
    plt.figure(figsize=(64,32))
    plt.axis('off')
    stepsize = int(timesteps_t/num_images)

    for i in range(0,timesteps_t)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timesteps(model,img, t, device)
        # Edit: This is to maintain the natural range of the distribution
        # its important, or otherwise we get a very blury almost all noise image
        img = torch.clamp(img,-1.0, 1.0)#-1.0, 1.0
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            plt.imshow(create_image_from_batch(img))
            plt.title(msg)
    plt.show()     

#now lets train!
device='cuda' if torch.cuda.is_available() else 'cpu'
model = Unet(in_channel=1, initial_fmap=64, time_embd_size=32,device=device)
# now for training we need an optimizer to get going
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000,gamma=0.1)
load_checkpoint = Path(f"{fldr}/diffusion_mnist.pt").exists()
epoch_start=0
if load_checkpoint:
    checkpoint = torch.load(f"{fldr}/diffusion_mnist.pt")
    epoch_start = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

print(f'running on {device}...')
print(f'batch count: {len(dataloader):,}')
print(f'checkpoint loaded!')

epochs = 3000
interval = 10
model = model.to(device)
# leave=True, makes tqdm stays in place, and position=0 and 1 makes each loop
# to occure in a different lines

for epoch in tqdm(range(epoch_start, epochs), leave=True, position=0):
    model.train()
    losses = []
    for i, (imgs,_) in tqdm(enumerate(dataloader), leave=True, position=1):
        # with torch.device(device):
        # lets create a few timesteps 
        t = torch.randint(0,timesteps_t, size=(batch_size,),dtype=torch.long, device=device)
        # now lets do a forward diffusion
        noisy_imgs, actual_noise = forward_diffusion(imgs, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,device)
        noisy_imgs, actual_noise = tuple(t.to(device) for t in (noisy_imgs, actual_noise))
        # now lets get the predicted noise from noisy images from previous step
        predicted_noises = model(noisy_imgs,t)
        # now lets calculate the loss 
        loss = F.l1_loss(actual_noise, predicted_noises)
        # now lets do a backward pass 
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # if i%interval==0:
            # print(f'Epoch: {epoch}/{epochs} | Iter: {i}/{len(dataloader)} | loss: {loss.item():.6f}')
            # sample_plot_image(model,device=device)
    if epoch%interval==0:  
        with torch.no_grad():
            model.eval()
            print(f'Epoch: {epoch}/{epochs} | loss: {np.mean(losses):.6f}')
            sample_plot_image(model,device=device,msg=f"Epoch: {epoch} | Loss: {np.mean(losses):.6f}")
            torch.save({"epoch":epoch,
                        "state_dict":model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "scheduler":scheduler.state_dict()},f"{fldr}/diffusion_mnist.pt")
print(f'finished')
#%%
@torch.no_grad()
def sample_plot_image(model,in_channel=1,num_images = 20,msg='',device='cpu'):
    # Sample noise
    img_size = 32
    img = torch.randn((batch_size, in_channel, img_size, img_size), device=device)
    plt.figure(figsize=(128,64))
    plt.axis('off')
    stepsize = int(timesteps_t/num_images)

    for i in range(0,timesteps_t)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timesteps(model,img, t, device)
        # Edit: This is to maintain the natural range of the distribution
        # its important, or otherwise we get a very blury almost all noise image
        img = torch.clamp(img,-1.0, 1.0)#-1.0, 1.0
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            plt.imshow(create_image_from_batch(img))
            plt.title(msg)
    plt.show()
sample_plot_image(model,num_images=10, device=device)

#%%
# now lets consolidate what we have learned so far, and write a diffusion model 
# in a much better way. previously everything was all over the place! lets implement
# it properly now.
# before we start implementing lets review what we need to implement/have
# 1. a diffusion foward process, this requires several values/betas/alphas etc
# 2. a reverse diffusion process/denoising part: this requires a unet model for denoising
# 3. a loss function, which is usually an l1 loss. 
# 4. a sampling function/method so we can generate new images from noise using our model
# 5. a dataset of images and the required dataloader. 
# 6. a training loop to train our model

from typing import Tuple
# lets import what we need
import time 
from datetime import datetime
import sys,os,math,random,copy
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
# from tqdm.notebook import tqdm


import torch
# for pylance so we get autocomplete for submodules!
import torch.utils
import torch.utils.data

import torch.nn as nn
import torch.nn.functional as F
import torchvision
# for pylance intellisense to work properly!
import torchvision.utils
import torchvision.datasets.utils

import torchvision.transforms as tfms
import matplotlib.pyplot as plt

fldr="/media/hossein/SSD1/code_dl/"

print(f'{torch.__version__}')
print(f'{sys.version}')

# since we need to encode our timesteps to 
# use them in our model we need to use a method 
# such as sinusoidal positional encoding.
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embd_size=32, device='cpu') -> None:
        super().__init__()
        self.embd_size = embd_size
        self.device = device
    
    @torch.no_grad()
    def forward(self, positions):
        with torch.device(self.device):
            pos_vec = torch.zeros(size=(positions.size(0),self.embd_size))
            div_term = torch.exp(-torch.arange(0, self.embd_size, 2) * (torch.log(torch.tensor([10_000], device=self.device)) / self.embd_size))
            positions = positions[:,None]
            pos_vec[:,0::2] = torch.sin(positions * div_term)
            pos_vec[:,1::2] = torch.cos(positions * div_term)
            return pos_vec

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# we need a block to do conv on the images and timesteps
# we can do this using functions, but a class/module form
# is more prefered
class ResBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 use_bn=True, 
                 act=nn.SiLU(),
                 time_embd_size=32, 
                 is_encoder=True,
                 dropout=None,
                 device='cpu',) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embd_size = time_embd_size
        self.device = device
        self.drpout = dropout
        
        if is_encoder:
            # if we are making encoder blocks, then we will be using conv2ds like normal and we shrink
            # the inputsize at each step, thats why we are using strides of 2. obviously in a realworld
            # scenario we dont downsample this rapidly! we use too much information, but for our simple case
            # this suffices. this is also the case when we code the decoder part where we upsample the
            # featuremaps.
            # so to make it obvious, we are hardcoding kernel/stride/padding triplet for both parts for
            # the sake of simplicity
            # sidenote 2: since we are also dealing with timesteps, we want to combine both inputs and
            # utilize it in our model. time information allows the model to learn to deal wil different
            # levels of noise properly.
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(num_features=out_channels) if use_bn else nn.Identity(),
                                      act)
        else:
            # otherwise, we will use convtransposed or (conv2d+upsample2d) to create larger featuremaps
            # we can use ksize=4 with stride=2 pad=1 or ksize=2 with stride=2 and padding=0 to double the fmap size
            # the difference between them is that the larger kernel size, results in a smoother output, and its
            # more common in generative models.
            #! use upsample layer insteda of contransposed, 
            self.conv = nn.Sequential(
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1,bias=False),
                                      # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,stride=2, padding=1),
                                      # we use a separate conv layer becasue contransposed is usually only used for upsampling
                                      # the learning part happens in the normal conv layer
                                      # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1,bias=False),
                                      nn.BatchNorm2d(num_features=out_channels) if use_bn else nn.Identity(),
                                      act
                                      )

        self.time_mlp = nn.Sequential(SinusoidalPositionalEncoding(embd_size=self.time_embd_size, device=self.device),
                                      nn.Linear(self.time_embd_size, out_channels),
                                      nn.BatchNorm1d(out_channels),
                                      nn.SiLU())

        # in ou case our skip-connection differs from our output 
        # (channel number is increased for each block) so the input
        # and the output of this block will have different channels
        # we have to use a second conv to make them compatible 
        layer,ksize,stride,padding = (nn.Conv2d, 3,2,1) if is_encoder else (nn.ConvTranspose2d, 4,2,1)
        self.h = nn.Sequential(layer(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=padding, bias=False),
                               # batchnorm is not needed, since we want to apply 
                               # a linear transformation only, and in fact no bn 
                               # improves our loss a bit more and increases our 
                               # inference and convergence speed as well
                               #nn.BatchNorm2d(out_channels)
                              )
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels,3,1,1,bias=False ),
                                   nn.BatchNorm2d(out_channels),
                                   nn.SiLU())
        self.drpout = nn.Identity() if self.drpout is None else nn.Dropout2d(self.drpout)

    def forward (self, x, t):
        identity = x
        # get time embeddigs 
        # time_embeddings = self.time_mlp(t)
        # combine the time embedding and input images, we 
        # add an extra dim to time_embd to make them compatible
        output = self.conv(x) #+ time_embeddings[..., None,None]
        #additional operation 
        output = self.conv2(output) 
        #! some people add the timeembedding to the skip_connection
        identity = self.h(identity)
        # print(f'{output.shape=} {identity.shape=}')
        return self.drpout(output + identity)


x0 = torch.randn(size=(3,1,32,32))
x1 = torch.randn(size=(3,64, 2,2))
t = torch.randint(0,200,size=(3,))

enc0 = ResBlock(1,64)
print(f'{enc0(x0,t).shape=}')
dec0 = ResBlock(64,1,is_encoder=False)
print(f'{dec0(x1,t).shape=}')

class UnetModel(nn.Module):
    def __init__(self, in_channels=1, base_fmap_size=64, embd_size=32, device='cpu') -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_fmap_size = base_fmap_size
        self.embd_size = embd_size
        self.device = device
        # self.dropout_last = dropout_last
        
        self.conv_in = nn.Sequential(nn.Conv2d(in_channels, base_fmap_size,3, padding=1,bias=False),
                                     nn.BatchNorm2d(base_fmap_size),
                                    nn.SiLU())
        fmap = base_fmap_size
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # instead of just multiplying by 2 each time, lets add by a constant value like 64/128
        # this will result in a much smaller model and the roughly the same performance
        self.growth_value=  128#64
        # encoder
        for i in range(5):
            # 0.05 is too small, 0.2 seems too high, 0.1 seems about right
            drpout = None if i<2 else 0.1
            self.encoder.append(ResBlock(fmap, fmap+self.growth_value, 
                                         time_embd_size=embd_size, 
                                         is_encoder=True, 
                                         device=self.device, 
                                         dropout=drpout))
            fmap +=self.growth_value
        # decoder
        for i in range(5):
            drpout = None if i<2 else 0.1
            # likewise instead of dividing by 2, lets subtract
            self.decoder.append(ResBlock(fmap, fmap-self.growth_value, 
                                         time_embd_size=embd_size, 
                                         is_encoder=False, 
                                         device=self.device, 
                                         dropout=drpout))
            fmap -=self.growth_value
        # print(f'{self.encoder=}')
        # print(f'{self.decoder=}')
        #!todo: then add more layers to block so we dont downsample at a rapid pace
        #!todo: then test if all is ok, and merge, but before that test these separately
        # instead of just a single conv layer that produces our final shape, we can use a deeper block
        # this allows us to not drastically shrink the output featuremaps, and hence get a much better
        # result and faster convergence. (we achieve the same loss at nearly one third of the epochs)
        # sidenote: removing the bn from the final_conv, and make it like this ( and use silo), 
        # increased our convergence speed by nearly 3 folds!
        self.final_conv = nn.Sequential(nn.Conv2d(fmap, fmap//2, kernel_size=3,padding=1, bias=False),
                                        #!todo: maybe we want to remove this
                                        nn.BatchNorm2d(fmap//2),
                                        nn.SiLU(),
                                        # note that we dont use any bn or act here, using bn
                                        # just hinders the convergence, think about it, we want
                                        # specific distribution for our images/noise and certainly
                                        # dont want to take all images/noisy inputs in our batch  
                                        # to influence our current image/noise, remember that unet 
                                        # tries to generate the noise we added to our image.
                                        # our noise for each sample is different so it makes sense
                                        # not to distort it with the stats(mean/var) of the whole 
                                        # batch! this is especially the case for more complex data
                                        # such as natural images of the cifar10 dataset. having a 
                                        # bn after this conv just wont allow for meaningful generation!
                                        # even for as many as 2200 epochs!
                                        nn.Conv2d(fmap//2,in_channels,kernel_size=3,padding=1,bias=False)
                                        )
        
    def forward(self, input_images, timesteps):
        out = self.conv_in(input_images)
        skip_connections = []
        
        for l in self.encoder:
            out = l(out, timesteps)
            skip_connections.append(out)
            # print(f'encoder:{out.shape=}')

        for l in self.decoder:
            skip = skip_connections.pop()
            out = l(out+skip, timesteps)
            # print(f'decoder:{out.shape=}')
            
        out = self.final_conv(out)
        # use tanh to make the values be in range -1,1
        # as our inputs range is -1,1
        # by removing this tanh, our loss in cifar10, with 10k imgs, starts at 0.07 instead 
        # of 0.3530!(image was resized to 128x128 instead of 32x32(which becomes 0.15) bytheway) and the patterns are much more visible 
        # this also is the case for mnist. tanh just destroys the output! for somereason!
        # return F.tanh(out)
        return out

x = torch.randn(size=(3,1,32,32))
m = UnetModel()
m(x,t).shape
# print(m) 
# lets create our class


# import math
# import torch
# from torch import nn
# from torch.nn import init
# from torch.nn import functional as F

# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)

# class TimeEmbedding(nn.Module):
#     def __init__(self, T, d_model, dim):
#         assert d_model % 2 == 0
#         super().__init__()
#         emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
#         emb = torch.exp(-emb)
#         pos = torch.arange(T).float()
#         emb = pos[:, None] * emb[None, :]
#         assert list(emb.shape) == [T, d_model // 2]
#         emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
#         assert list(emb.shape) == [T, d_model // 2, 2]
#         emb = emb.view(T, d_model)

#         self.timembedding = nn.Sequential(
#             nn.Embedding.from_pretrained(emb),
#             nn.Linear(d_model, dim),
#             Swish(),
#             nn.Linear(dim, dim),
#         )
#         self.initialize()

#     def initialize(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 init.xavier_uniform_(module.weight)
#                 init.zeros_(module.bias)

#     def forward(self, t):
#         emb = self.timembedding(t)
#         return emb

# class DownSample(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#         self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
#         self.initialize()

#     def initialize(self):
#         init.xavier_uniform_(self.main.weight)
#         init.zeros_(self.main.bias)

#     def forward(self, x, temb):
#         x = self.main(x)
#         return x

# class UpSample(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#         self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
#         self.initialize()

#     def initialize(self):
#         init.xavier_uniform_(self.main.weight)
#         init.zeros_(self.main.bias)

#     def forward(self, x, temb):
#         _, _, H, W = x.shape
#         x = F.interpolate(
#             x, scale_factor=2, mode='nearest')
#         x = self.main(x)
#         return x

# class AttnBlock(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#         self.group_norm = nn.GroupNorm(32, in_ch)
#         self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.initialize()

#     def initialize(self):
#         for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
#             init.xavier_uniform_(module.weight)
#             init.zeros_(module.bias)
#         init.xavier_uniform_(self.proj.weight, gain=1e-5)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         h = self.group_norm(x)
#         q = self.proj_q(h)
#         k = self.proj_k(h)
#         v = self.proj_v(h)

#         q = q.permute(0, 2, 3, 1).view(B, H * W, C)
#         k = k.view(B, C, H * W)
#         w = torch.bmm(q, k) * (int(C) ** (-0.5))
#         assert list(w.shape) == [B, H * W, H * W]
#         w = F.softmax(w, dim=-1)

#         v = v.permute(0, 2, 3, 1).view(B, H * W, C)
#         h = torch.bmm(w, v)
#         assert list(h.shape) == [B, H * W, C]
#         h = h.view(B, H, W, C).permute(0, 3, 1, 2)
#         h = self.proj(h)

#         return x + h

# class ResBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
#         super().__init__()
#         self.block1 = nn.Sequential(
#             nn.GroupNorm(32, in_ch),
#             Swish(),
#             nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
#         )
#         self.temb_proj = nn.Sequential(
#             Swish(),
#             nn.Linear(tdim, out_ch),
#         )
#         self.block2 = nn.Sequential(
#             nn.GroupNorm(32, out_ch),
#             Swish(),
#             nn.Dropout(dropout),
#             nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
#         )
#         if in_ch != out_ch:
#             self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
#         else:
#             self.shortcut = nn.Identity()
#         if attn:
#             self.attn = AttnBlock(out_ch)
#         else:
#             self.attn = nn.Identity()
#         self.initialize()

#     def initialize(self):
#         for module in self.modules():
#             if isinstance(module, (nn.Conv2d, nn.Linear)):
#                 init.xavier_uniform_(module.weight)
#                 init.zeros_(module.bias)
#         init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

#     def forward(self, x, temb):
#         h = self.block1(x)
#         h += self.temb_proj(temb)[:, :, None, None]
#         h = self.block2(h)

#         h = h + self.shortcut(x)
#         h = self.attn(h)
#         return h

# class UNet(nn.Module):
#     def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
#         super().__init__()
#         assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
#         tdim = ch * 4
#         self.time_embedding = TimeEmbedding(T, ch, tdim)

#         self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
#         self.downblocks = nn.ModuleList()
#         chs = [ch]  # record output channel when dowmsample for upsample
#         now_ch = ch
#         for i, mult in enumerate(ch_mult):
#             out_ch = ch * mult
#             for _ in range(num_res_blocks):
#                 self.downblocks.append(ResBlock(
#                     in_ch=now_ch, out_ch=out_ch, tdim=tdim,
#                     dropout=dropout, attn=(i in attn)))
#                 now_ch = out_ch
#                 chs.append(now_ch)
#             if i != len(ch_mult) - 1:
#                 self.downblocks.append(DownSample(now_ch))
#                 chs.append(now_ch)

#         self.middleblocks = nn.ModuleList([
#             ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
#             ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
#         ])

#         self.upblocks = nn.ModuleList()
#         for i, mult in reversed(list(enumerate(ch_mult))):
#             out_ch = ch * mult
#             for _ in range(num_res_blocks + 1):
#                 self.upblocks.append(ResBlock(
#                     in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
#                     dropout=dropout, attn=(i in attn)))
#                 now_ch = out_ch
#             if i != 0:
#                 self.upblocks.append(UpSample(now_ch))
#         assert len(chs) == 0

#         self.tail = nn.Sequential(
#             nn.GroupNorm(32, now_ch),
#             Swish(),
#             nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
#         )
#         self.initialize()

#     def initialize(self):
#         init.xavier_uniform_(self.head.weight)
#         init.zeros_(self.head.bias)
#         init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
#         init.zeros_(self.tail[-1].bias)

#     def forward(self, x, t):
#         # Timestep embedding
#         temb = self.time_embedding(t)
#         # Downsampling
#         h = self.head(x)
#         hs = [h]
#         for layer in self.downblocks:
#             h = layer(h, temb)
#             hs.append(h)
#         # Middle
#         for layer in self.middleblocks:
#             h = layer(h, temb)
#         # Upsampling
#         for layer in self.upblocks:
#             if isinstance(layer, ResBlock):
#                 h = torch.cat([h, hs.pop()], dim=1)
#             h = layer(h, temb)
#         h = self.tail(h)

#         assert len(hs) == 0
#         return h
    
# batch_size = 8
# model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],num_res_blocks=2, dropout=0.1)
# x = torch.randn(batch_size, 3, 32, 32)
# t = torch.randint(1000, (batch_size, ))
# y = model(x, t)

from contextlib import contextmanager
from copy import deepcopy
import math

from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as TF
# from tqdm.notebook import tqdm, trange

# Utilities

@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


# Define the model (a residual U-Net)

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout_last=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True) if dropout_last else nn.Identity(),
            nn.ReLU(inplace=True),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class DiffusionNew(nn.Module):
    def __init__(self,in_channels=3, c=64, embd_size=16,device='cuda'):
        super().__init__()
        #c = 64  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        # self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        # self.class_embed = nn.Embedding(10, 4)
        self.time_mlp = nn.Sequential(SinusoidalPositionalEncoding(embd_size=embd_size, device=device),
                                      nn.Linear(embd_size, embd_size),
                                    #   nn.BatchNorm1d(embd_size),
                                    #   nn.ReLU()
                                      )
        self.net = nn.Sequential(   # 32x32
            #!to use timebeding use in_channels+embd_size below
            ResConvBlock(in_channels , c, c),# ResConvBlock(in_channels +embd_size, c, c)
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool2d(2),  # 32x32 -> 16x16
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),  # 16x16 -> 8x8
                    ResConvBlock(c * 2, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 4),
                    SkipBlock([
                        nn.AvgPool2d(2),  # 8x8 -> 4x4
                        ResConvBlock(c * 4, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 4),
                        nn.Upsample(scale_factor=2),
                    ]),  # 4x4 -> 8x8
                    ResConvBlock(c * 8, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 2),
                    nn.Upsample(scale_factor=2),
                ]),  # 8x8 -> 16x16
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=2),
            ]),  # 16x16 -> 32x32
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout_last=False),
        )

    def forward(self, input,t):
        try:
            tstep = self.time_mlp(t)
            timestep_embd = expand_to_planes(tstep, input.shape)
            # print(f'{input.shape=} {timestep_embd.shape=} {tstep.shape=}')
            # timestep_embed = expand_to_planes(self.timestep_embed(log_snrs[:, None]), input.shape)
            # class_embed = expand_to_planes(self.class_embed(cond), input.shape)
            return self.net(input)
            # return self.net(torch.cat([input,timestep_embd], dim=1))
        except Exception as exp:
            tstep = self.time_mlp(t)
            # timestep_embd = expand_to_planes(tstep, input.shape)
            # print(tstep[..., None, None].shape) #(1,16,1,1)
            # !becasue here we are trying to use a single timestep for a batch of images we fail
            # at the expansion part, so here what I did (im not sure if its ok yet!) is to
            # expand the batch dimension so we get the timesteps aligned for each image and see how it goes!
            # hopefully its ok!
            timestep_embd = tstep[..., None, None].repeat([input.shape[0], 1, input.shape[2], input.shape[3]])
            # print(f'{input.shape=} {timestep_embd.shape=} {tstep.shape=}')
            # timestep_embed = expand_to_planes(self.timestep_embed(log_snrs[:, None]), input.shape)
            # return self.net(torch.cat([input,timestep_embd], dim=1))
            return self.net(input)

#side note:
# I only changed the architecture, and it improved the results drastically! 
# I mean, I didnt even feed it the timesteps! theres no conditioning (like class conditioning)
# or any thing related to timesteps! and it produces more vibrant colors! as early as 20-40 epochs!
# with lr= 0.00002. 0.0002 seems to converge faster!we reach 0.0547 at e40! with
# good, i.e. much more vibrant pictures than before!
# so I guess whent he images are grim/darkish/ it means the model is underfittin!
# not that it lacks paramaters, but it can not process the input properly!(the discriminative power is not there)
#al so the new model doesnt employ the unetarchi tecture the way we  created one, that is
# theres no connection between encoder and decoders featuremaps, its just an ordinary
# hour glass architecture! at epoch 200/400, we have vibrant images, just like cifar, however
# the composition isno t there, i.e. while the overall images are natural looking, thecon tent are demorphed!
#not  yet properly formed. I guesswecan  a ttributed this to sampling/lack of conditioninga t this point
# Next: add timestep information and see how that goes!
# running with timestep with bn seems to make results very blury/grimish/grayish like before!
# it may very well have been the addition of timesteps like this! that contributed to this issue
# lets see how it goes!
# Ok I tested with timsetps, with bn and without it, with silu and relu, it doesnt make any difference
# even without bn and without any activation functions, it just makes the output blury/ grayish
# without timesteps, it just creates very vibrant images very quickly and very well!
# Next revert to base model and use the new sampling method instead!

class DiffusionMnist(nn.Module):
    def __init__(self, in_channels=1, base_fmap_size=64, embd_size=32, num_timesteps=200, linear_scheduler=True, device = 'cpu') -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_fmap_size = base_fmap_size
        self.embd_size = embd_size
        self.num_timesteps = num_timesteps
        self.device = device
        self.linear_scheduler = linear_scheduler
        
        # self.unet_model = UnetModel(in_channels, base_fmap_size, embd_size=embd_size, device=device)
        self.unet_model = DiffusionNew(in_channels, base_fmap_size, embd_size=embd_size)
        # self.unet_model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],num_res_blocks=2, dropout=0.1)
        self.unet_model.to(device)
        # lets initialize our attributes for the forward_diffusion process 
        self._init_parameters()
    
    def forward(self, input_images:torch.Tensor, timesteps:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the forward diffusion followed by unet forward

        Args:
            input_images (torch.Tensor): input noise
            timesteps (torch.Tensor): input timestep

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: returns a tuple of predicted_noises and actual noises 
        """
        noisy_images, actual_noises = self.forward_diffusion(input_images, timesteps)
        predicted_noises = self.forward_unet(noisy_images, timesteps)
        return predicted_noises, actual_noises
    
    def forward_unet(self, input_images:torch.Tensor, timesteps:torch.Tensor) -> torch.Tensor:
        predicted_noise = self.unet_model(input_images, timesteps)
        return predicted_noise
    
    def forward_diffusion(self, input_images:torch.Tensor, timesteps:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check if the input is a batch or a single image
        is_batch = input_images.ndim>3
        # Create a noise tensor with the same dimensions as input_images
        actual_noises = torch.randn_like(input_images)
        # Get sqrt_alphas_cumprod and sqrt_one_minus_alphas_cumprod for current timesteps
        sqrt_alphas_cumprod_t = self._get_value_for_timestep_t(self.sqrt_alphas_cumprod, timestep_indexes=timesteps, use_batch=is_batch)
        sqrt_one_minus_alphas_cumprod_t = self._get_value_for_timestep_t(self.sqrt_one_minus_alphas_cumprod, timesteps, is_batch)
        # now calculate the mean + variance to get the noisy image
        noisy_images = (sqrt_alphas_cumprod_t * input_images) + (sqrt_one_minus_alphas_cumprod_t * actual_noises)
        # return the noisy_image along with the actual noise
        return noisy_images, actual_noises

    @torch.no_grad()
    def display_sample(self, input_channel=1, batch_size=1, image_height=32, image_width=32, num_images=20, title='', fig_size=(8,6)):
        # set the model in eval mode first
        is_training=model.training
        if model.training:
            model.eval()
        with torch.device(self.device):
            # create noise 
            noise = torch.randn(size=(batch_size, input_channel, image_height, image_width))
            # configure out plot size and remove the axis for uncluttered output
            plt.figure(figsize=(fig_size))
            plt.axis("off")
            # set a stepsize so we display only num_images intermediate images for our diffusion process
            step_size = self.num_timesteps//num_images
            # now reverse the timestep in denoising 
            for i in range(0, self.num_timesteps)[::-1]:
                # sidenote: torch.full creates a tensor of the specified size filled with a fill value.
                # its is used when we want to create a tensor of a certain size and fill it with a 
                # specific value. This is useful when we need a tensor of a certain size, but don’t
                # care about the exact values because they’re all going to be the same. we could also 
                # simply use torch.tensor([i])
                # t = torch.full(size=(1,), fill_value=t, dtype=torch.long)
                timestep = torch.tensor([i], dtype=torch.long)
                noise = self._sample(noise, timestep)
                # This is to maintain the natural range of the distribution
                # its important, or otherwise we get a very blury almost all noise image
                noise = torch.clamp(noise, -1.0, 1.0)
                if i%step_size==0:
                    plt.subplot(1, num_images, (i//step_size)+1)
                    img = self._create_image_from_batch(noise, img_shape=(image_height, image_width, input_channel))
                    plt.imshow(img)
                    plt.title(title)
            # show the image
            plt.show()
        # restore the model status
        if is_training:
            model.train()

    @torch.no_grad()
    def gen_images(self, timestep, input_channel=1, batch_size=1, image_height=32, image_width=32):
        #ideally we would refactor dsplayimage and this method so that display image uses this
        #this method would take a previous_noise and thus would be used inside the loop and yeild
        #the result. but for now, im adding this like this
        # note that the timestep should not be exactly the same as  the number of timesteps, that is
        # since our arrays are 0 based, we can have 0 up to num_timsteps-1 only. if we try to get
        # a value = num_timesteps, we will face weird error like "RuntimeError: GET was unable to find an engine to execute this computation"
        # which is not really showing the real cause of error especially when dealing with autocast and fp16
        # training
        # assert timestep<self.num_timesteps, f'Given timestep is too large!. the given timestep({timestep}) must be less than the total number of timesteps({self.num_timesteps})'
        # set the model in eval mode first
        is_training=model.training
        if model.training:
            model.eval()
        img=None
        with torch.device(self.device):
            noise = torch.randn(size=(batch_size, input_channel, image_height, image_width))
            # set a stepsize so we display only num_images intermediate images for our diffusion process
            # step_size = self.num_timesteps//10
            # now reverse the timestep in denoising 
            for i in range(0, self.num_timesteps)[::-1]:
                # create noise
                t = torch.tensor([i], dtype=torch.long)
                noise = self._sample(noise, t)
                # This is to maintain the natural range of the distribution
                # its important, or otherwise we get a very blury almost all noise image
                noise = torch.clamp(noise, -1.0, 1.0)
                if i==timestep:
                    img = self._create_image_from_batch(noise, img_shape=(image_height, image_width, input_channel))
                    # plt.imshow(img)
                    # plt.show()
                    break
        # restore the model status
        if is_training:
            model.train()
        return img
    
    @torch.no_grad()
    def _sample(self, input_images, timesteps):
        # Check whether input_images is a batch of images or a single one
        is_batch = input_images.ndim>3
        # get the betas for current timestep
        betas_t = self._get_value_for_timestep_t(self.betas, timesteps, is_batch)    
        # get the sqrt_one_minus_alphas_cumprod for current timestep as well
        sqrt_one_minus_alphas_cumprod_t = self._get_value_for_timestep_t(self.sqrt_one_minus_alphas_cumprod, timesteps, is_batch)
        # get the sqrt_recip_alphas for current timestep
        sqrt_recip_alphas_t = self._get_value_for_timestep_t(self.sqrt_recip_alphas, timesteps, is_batch)
        # now call the denoising model noise_prediction 
        predicted_noise = self.forward_unet(input_images, timesteps)
        # calculate the model mean
        model_mean =  sqrt_recip_alphas_t * (input_images - betas_t*predicted_noise/sqrt_one_minus_alphas_cumprod_t)
        
        # The if clause here is checking whether the timestep is 0 or not, if it is, 
        # then it returns the model_mean tensor. 
        # This is because at the initial timestep (t=0), the model assumes that there's
        # no noise added yet, so it returns the original data (after some transformations 
        # defined by the model). 
        # If the timestep is not 0, it means the diffusion process has started. In this case,
        # the function generates a noise tensor with the same shape as the input images 
        # (torch.randn_like(input_images, device=self.device)) and adds this noise to the model_mean. 
        # The amount of noise added is scaled by the square root of the posterior_variance_t, 
        # which is a measure of how much the model expects the data to have diffused at this timestep.
        # sidenote2: note that since our timestep is always 1 dimensional its ok to do ==
        # otherwise we'd face an error. torch.all() would work regardless but to convey and
        # make sure timesteps needs to be 1 dimensional here, we use ==
        if timesteps == 0:
            return model_mean.to(self.device)
        else:
            noise = torch.randn_like(input_images, device=self.device)
            # get the posterior variance for the current timestep
            posterior_variance_t = self._get_value_for_timestep_t(self.posterior_variance, timesteps, is_batch)
            return model_mean + torch.sqrt(posterior_variance_t)*noise

    def _init_parameters(self,beta_start=0.001, beta_end=0.02):
        # β
        if self.linear_scheduler:
            self.betas = self._create_betas_linear(start=beta_start, end=beta_end)
        else:
            self.betas = self._create_betas_cosine(self.num_timesteps)
            
        # α 
        self.alphas = 1.0 - self.betas
        # ̅α 
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        # √̅α
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # √1-̅α 
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        # alphas_prev
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], pad=(1,0), value=-1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0/self.alphas)
        # These calculations are part of the reverse process of the diffusion model, 
        # where the model gradually denoises an image starting from pure noise. 
        # The `alphas_cumprod_prev`, `sqrt_recip_alphas`, and `posterior_variance` are 
        # used in the calculation of the Gaussian distribution from which the denoised 
        # image is sampled at each time step.
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # print([t.device for t in (self.betas,self.alphas ,self.alphas_cumprod,
        #                             self.sqrt_alphas_cumprod,
        #                             self.sqrt_one_minus_alphas_cumprod,
        #                             self.alphas_cumprod_prev,
        #                             self.sqrt_recip_alphas,
        #                             self.posterior_variance)])
            
    @torch.no_grad()
    def _create_betas_linear(self, start=0.001, end=0.02)-> torch.Tensor :
        return torch.linspace(start=start, end=end, steps=self.num_timesteps, device=self.device)

    @torch.no_grad()
    def _create_betas_cosine(self, timesteps, s = 0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        ref: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L387
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    @torch.no_grad()
    def _get_value_for_timestep_t(self, tensors:torch.Tensor, timestep_indexes:torch.Tensor, use_batch=True):
        batch_size = timestep_indexes.size(0)
        # print([t.device for t in (tensors, timestep_indexes)])
        values_at_t =  torch.gather(tensors, dim=-1, index=timestep_indexes)
        shape = (batch_size, 1,1,1) if use_batch else (batch_size, 1,1)
        return values_at_t.reshape(shape)

    @torch.no_grad()
    def _create_image_from_batch(self, imgs_tensor:torch.Tensor, img_shape=(32,32,1))->np.ndarray:
        imgs = imgs_tensor.permute(0,2,3,1).detach().cpu()
        img_rows = []
        # how many images do we want in each row
        ncol = int(np.sqrt(imgs.size(0)))
        # print(f'{imgs.size(0)=} {ncol=}')
        for i in range(0, imgs.size(0), ncol):
            # grab ncol images at a time from our batch
            img_row = imgs[i:i+ncol]
            # concatenate them along the column, so we get a row of images
            img_row = torch.cat(img_row.chunk(ncol, dim=0), dim=2).reshape(img_shape[0],-1,img_shape[-1])
            # store them to later stack them and get a full image
            img_rows.append(img_row)
        # print([t.shape for t in img_rows])
        # stack the images along the height and get our final image
        img_grid = torch.cat(img_rows, dim=0).numpy()
        # normalize the image to be in range (0-1) since
        # matplotlib uses 0-1 for floats and 0-255 for int
        # images. we just reverse the process we used for converting
        # (0-1) to (-1,1).
        # rescale the image to 0-1 range
        img_grid = (img_grid+1)/2
        return img_grid

# taken from : https://colab.research.google.com/drive/1IJkrrV-D7boSCLVKhi7t5docRYqORtm3#scrollTo=s8IFYM8fy5h8
@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)

# now lets grab our data
def get_dataset(name='mnist',size=32, mode='val', transforms=None):
    """returns the dataloader object for the specified dataset.

    Args:
        name (str, optional): name of the dataset to load('mnist,cifar10'). Defaults to 'mnist'.
        size (int, optional): image to be resized to. Defaults to 32.
        mode (str, optional): specifies how the dataset to be retrieved. the modes include
        ['train', 'val', 'both']. Defaults to 'val'.

    Returns:
        DataLoader: returns a Dataloader
    """
    if transforms is None:
        transforms = torchvision.transforms.Compose([tfms.Resize(size),
                                                    tfms.ToTensor(),
                                                    # rescale the input to the -1,1 range,
                                                    # !its important to get good result
                                                    tfms.Lambda(lambda x: x*2-1)
                                                    ])
    if name.lower() in 'mnist':
        # we can test mnist and then other datasets such as cifar10 etc 
        dt_tr = torchvision.datasets.MNIST(f'{fldr}/data',True, transform=transforms, download=True )
        dt_val = torchvision.datasets.MNIST(f'{fldr}/data',False, transform=transforms, download=True)
    
    elif name.lower() in 'cifar10':
        dt_tr = torchvision.datasets.CIFAR10(f'{fldr}/data',True, transform=transforms, download=True )
        dt_val = torchvision.datasets.CIFAR10(f'{fldr}/data',False, transform=transforms, download=True )
    
    elif name.lower() in 'stanfordcars':# cars
        # issue https://github.com/pytorch/vision/issues/7545 dataset is no more availabe!
        dt_tr = torchvision.datasets.StanfordCars(f'{fldr}/data',split='train', transform=transforms, download=True )
        dt_val = torchvision.datasets.StanfordCars(f'{fldr}/data',split='test', transform=transforms, download=True )
    
    else:
        raise Exception(f'Unknown dataset name ({name}) entered')
    
    if 'train' == mode:
        return dt_tr
    elif 'val' == mode:
        return dt_val
    elif 'both' == mode:
        # concat the train/val splits 
        return torch.utils.data.ConcatDataset([dt_tr, dt_val])
    else:
        raise Exception(f"Unknown mode ({name}) entered.(valid modes are ['train','val','both'])")

def get_dataloader(dataset, batch_size=32, num_workers=8, drop_last=False):
    # set drop_last=True so, the batch sizes all are the same
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                        shuffle=True, 
                                        num_workers=num_workers, 
                                        pin_memory=True,
                                        drop_last=drop_last)

# now let us train
# - I made the network larger, 5 blocks instead of 4/ 
# - and also I added a second conv in the resblock 
# - after the concat of conv and timeembdedding
# - then trained the model and noticed our 54million model
# - which has roughly the same parameter count for encoder and decoder
# - started to generate pattern, but after some time the same pattern
# - got replicated. signalling we may very well be overfitting so 
# - now im using dropout layers at the end of all resblocks to see how it goes
# - only the first two blocks dont have dropouts and we are using train+val this time
# todo beofre that lets decrease timesteps to 800 and use no drpout, and use val to see how it performs
# todo next we need to change betas_start = 0.0001 and see how it affects the result as well(did it+drp+val)
# ok ok. now booth images
# sidenotes:
# Number of Timesteps: The number of timesteps in a diffusion model corresponds to the number of steps in the Markov chain that transitions from the data distribution to the noise distribution. A larger number of timesteps can potentially result in a more accurate approximation of the data distribution, but it also increases the computational cost and complexity of the model. If you’re finding that your model is not learning effectively or is producing identical images, reducing the number of timesteps could be worth trying. However, this is a hyperparameter that you would typically tune based on the performance of your model on a validation set.
# Starting Value of Betas: The starting value of betas determines the amount of noise added in the first step of the diffusion process. A larger starting value means more noise is added initially, which could make the learning task more difficult for the model. On the other hand, a smaller starting value means less noise is added, which could make the learning task easier but might also result in less diverse generated images. Again, the optimal value for this hyperparameter can depend on your specific task and dataset, and it’s something you would typically tune based on model performance.



#fp16 sometimes mess with the results, and causes high loss! 
# I trained my best model without, so if something weird happens
# disable the fp16 traininghere (it should work ok though so in case
# it ever happened again disable it. for the record for mnist we should
# be getting 0.2170 around 60/80 epochs). with fp16 it takes 16 
# minutes to reach 80 epochs (each epoch takes around 4 minutes
# without fp16 eacy epoch takes around 7 minutes)
use_fp16=True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_name = 'cifar'
load_checkpoint = False
checkpoint_name = f'diffusion_{dataset_name}_newarch_no_ts.pth'
# note large batchsize such as 256 lead to wrose result and much slower convergence!
# try batchsize of 32 and 256 for example and see the very first epochs how the results
# show. batch of 32 is way better than batch 256. this could be casued by batchnorm maybe?
# note that too small of a batch also doesnt work, it results in more noisy output?! try 
# batchsize of 2 for example and see what I mean.
batch_size = 32
num_workers = 8
epochs = 6000
epoch_start=0
step_size=7000
interval = 20

# resize image
image_size = 32
transforms2 = torchvision.transforms.Compose([tfms.Resize(image_size),
                                              #tfms.Grayscale(),
                                              tfms.ToTensor(),
                                              # rescale the input to the -1,1 range,
                                              # !its important to get good result
                                              tfms.Lambda(lambda x: x*2-1)
                                              ])

dataset = get_dataset(dataset_name, size=image_size, mode='val',transforms=transforms2)
# !higher numbers like 800 seem not to perform well, at least in my limited experiments
# needs more testing, higher timestep seems to affect loss at least in cifar case
# 200 works prefect for mnist
# I noticed with a low timestep_num like 100, the generation takes longer
# to generate good images, when we used num_timesteps=200, as early 
# as 40 epochs we had prefect numbers formed, however, when t=100, 
# even til epoch 80 we had giberish yet clean/smooth images (this shows
# the importance of timesteps. the higher the timesteps, the better the 
# results the earlier! likewise it affects time loss as well. 
# so for mnist, a timestep of 200/400 seem like a good choice
# for cifar seems a higher number is a btter choice, like 400/800
#todo: make model larger, it worked and did better we ultimately reached 0.210 using mse loss
#todo: and around 0.09xx from epoch 2600 we used mse loss and trained until 5480 epochs at which
#todo point I ended the training because I used a lower dropout (0.05) and it overfitted the model
#todo so it started repeating a single image so I gave up. prior to that it was starting to create 
#todo much better images, but since it was slow, I lowered the lr to 0.00001 after 1072 epochs
#next i plan on using 500 for timesteps and use attenstions to see if that makes anydifference
#also I used val for cifar10 only
num_timesteps = 500
embd_size = 16#64
# the learning rate is very important, 
# and 1e-4 seems to work just fine, 
# anything larger like 1e-3 e.g. wont 
# work and results in noise and high loss (0.3793) # 20-0.25
# in cifar, with t=200, loss becomes nan in epoch 40, while with t=400 it goes strong
# so I increased the t to 600, and started the training to see how it goes
# with large lr like 0.001, if we dont use bn in the final_conv, we get nans.
# with larger imagesize (i.e. 64) large lr like 0.001 causes nans! even with bn in fnal_con 
# and even with t=800(32isok)
#after 1740 epochs
lr = 0.0002 #0.00002

# mnist is 1 channel, and cifar10 is 3!
in_channels = 1 if 'mnist' in dataset_name else 3
base_fmap_size = 64
model = DiffusionMnist(in_channels=in_channels, 
                       base_fmap_size=base_fmap_size,
                       embd_size=embd_size, 
                       num_timesteps=num_timesteps,
                       linear_scheduler=True,
                       device=device)
# to increase performance
# model.compile()
model_ema = copy.deepcopy(model)
# sidenote: recall 
# the variance schedule beta(β) tells us how much noise we want to add in each time steps.
# we linearly increase it until we reach a maximum value of 0.02. if we wouldnt increase it at all,
# it would take for ever to endup with pure noise (full noise image). therefore the authors
# defined a new term alpha(α) which is simply (1-β), we can think of it as, how much information
# we get to keep about an image when transitioning to another/next image.
model._init_parameters(beta_start=0.0001,beta_end=0.02)

optimizer = torch.optim.Adam(model.parameters(), lr = lr)
# 0.0001 is small enough and lowering it would imepede the convergence further
# so I just set it at 3000 to mean donot change it! why use it then? to test with
# different cases! feel free to choose and play with other schedulers and optimizers
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,gamma=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

# calculate model parameters
num_params = np.sum([p.numel() for p in model.parameters()])
# num_params_enc = np.sum([p.numel() for p in model.unet_model.encoder.parameters()])
# num_params_dec = np.sum([p.numel() for p in model.unet_model.decoder.parameters()])

ema_decay = 0.998

if load_checkpoint and Path(f"{fldr}/{checkpoint_name}").exists():
    checkpoint = torch.load(f"{fldr}/{checkpoint_name}")
    print(f'{checkpoint.keys()}')
    epoch_start = checkpoint["epoch"]
    use_fp16 = checkpoint.get("use_fp16",False)
    model.unet_model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    model_ema.unet_model.load_state_dict(checkpoint["model_ema"])

current_time = datetime.now().strftime('%Y%m%d_%H_%M_%S')

print(f'running on     : {device}/{model.device}')
print(f'experiment date: {current_time}')
print(f'dataset length : {len(dataset):,}')
print(f'image size     : {image_size}')
print(f'in_channels    : {in_channels}')
print(f'base_fmap_size : {base_fmap_size}')
print(f'checkpointname : {checkpoint_name}')
print(f'is resumed     : {load_checkpoint}')
print(f'uses FP16      : {use_fp16}')
print(f'batch_size     : {batch_size}')
print(f'num_epochs     : {epochs}')
print(f'epoch_start    : {epoch_start}')
print(f'interval       : {interval}')
print(f'n_timestep     : {model.num_timesteps}')
print(f'embd_size      : {model.embd_size}')
print(f'learning_rate  : {lr}')
print(f'step_size      : {step_size}')
print(f'model n_params : {num_params:,}')
# print(f'enc n_params  : {num_params_enc:,}')
# print(f'dec n_params  : {num_params_dec:,}')

for epoch in tqdm(range(epoch_start, epochs)):
    losses = []
    model.train()
    for i, (imgs,_) in tqdm(enumerate(get_dataloader(dataset,
                                                     batch_size=batch_size,
                                                     num_workers=num_workers))):
        # sidenote: since torch 2.0.0 we can use torch.device as a context manager!
        # but it only works at the tensor creation time! i.e. before a tensor is created
        # ithas to be called.
        with torch.cuda.amp.autocast(enabled=use_fp16):
            imgs = imgs.to(model.device)
            # pick a timestep
            t = torch.randint(low=0, high=num_timesteps, size=(imgs.size(0),),device=device).long()
            # making the times close to the end more probable
            # instead of using a uniform distribution for taking the t
            # we instead pick the ones with higher probablities (which means
            # the ones at the very end have higher probabilities)
            # probs = torch.linspace(0,1,steps=num_timesteps,device=device).softmax(dim=-1)
            # t = torch.multinomial(probs, num_samples=imgs.size(0),replacement=True).long()
            predicted_noises, noises = model(imgs, t)
            loss = F.mse_loss(predicted_noises, noises)

            losses.append(loss.item())
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            #update ema
            ema_update(model, model_ema, 0.95 if epoch < 20 else ema_decay)
            scaler.update()
            
    scheduler.step()
    
    if epoch%interval==0:
        with torch.cuda.amp.autocast(enabled=use_fp16):
            model.eval()
            # save image for each epoch
            img_gen = model.gen_images(0,
                                    model.in_channels, 
                                    batch_size=64,
                                    image_height=image_size,
                                    image_width=image_size)
            dir_path = f"{fldr}/imgs_gen_{current_time}/"
            fname = f"{dataset_name}_img_{epoch}.jpg"
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            # we need to convert our 0-1 range image to a proper format pil supports
            # otherwise we get Cannot handle this data type: (1, 1, 3), <f4 which simply
            # means, our image has 32-bit floating point numbers in it. pil requires
            # uint8 numbers, i.e. 0-255. so to convert our 0-1 range to 0-255 we simply
            # do this (img*255).astype(np.unit8)
            Image.fromarray((img_gen * 255).astype(np.uint8)).save(os.path.join(dir_path, fname))
            print(f'Epoch: {epoch}/{epochs} | Loss: {np.mean(losses):.4f} | lr:{scheduler.get_last_lr()[-1]:.1e}')
            model.display_sample(input_channel=model.in_channels,
                                batch_size=64,
                                image_height=image_size,
                                image_width=image_size,
                                num_images=10,
                                fig_size=(64,32),
                                title=f'Epoch: {epoch} | Loss: {np.mean(losses):.4f}')
            torch.save({"epoch":epoch,
                        "use_fp16":use_fp16,
                        "state_dict":model.unet_model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "scheduler":scheduler.state_dict(),
                        "scaler":scaler.state_dict(),
                        'model_ema': model_ema.state_dict(),},
                    f"{fldr}/{checkpoint_name}")

# mnist: 
# we startedwith a loss of 0.2835 and achieved a loss of 0.2028 at 1760 epochs, 
# we saw that as early as 20 epochs we had a somewhat good result which got constant
# improvement. since our lr was small, it obviously took a lot, the more the model is
# trained, the more prominent/sharper the final images become. 
# we also noticed that proper range for our images matter, if we dont rescale our final
# images from (-1,1) back to (0,1) we will get darker images becasue matplotlib uses 0-1
# for float images, and 0-255 for integer images, it clips the values smaller than 0 so
# it causes the images not to display accurately. this is more prominent when we train a
# color image  like from cifar10 datasets.
# we also see that if we try different shapes, like different widths, hieght we get 
# the output but they dont look good. larger sizes also show this fact that our model
# works well on the image dimensions it was trained on! we trained it 32x32 so it performs
# well on this resolution. try 32x64, and 64x64 and see the result
# for cifar10, the loss ddint go down like mnist and stayed the same for 1000 epochs at 0.3533
# so I lowered the lr again. followed by removing mean/std etc
#%%
model.display_sample(input_channel=model.in_channels,
                    batch_size=1,
                    image_height=image_size,
                    image_width=image_size,
                    num_images=10,
                    fig_size=(64,32),
                    title=f'Epoch: {epoch} | Loss: {np.mean(losses):.4f}')

#%%
# test with otherpeoples implementation
# Imports

from contextlib import contextmanager
from copy import deepcopy
import math
from datetime import datetime
from IPython import display
from matplotlib import pyplot as plt
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm, trange


# Utilities

@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


# Define the model (a residual U-Net)

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout_last=False): #!it was True by default (to get the best result itmust be True)
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            # nn.Dropout2d(0.1, inplace=True), #! this must be enabled to get the best results
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True) if dropout_last else nn.Identity(),
            nn.ReLU(inplace=True),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.class_embed = nn.Embedding(10, 4)

        self.net = nn.Sequential(   # 32x32
            ResConvBlock(3+16+4, c, c),# 3+16+4
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool2d(2),  # 32x32 -> 16x16
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),  # 16x16 -> 8x8
                    ResConvBlock(c * 2, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 4),
                    SkipBlock([
                        nn.AvgPool2d(2),  # 8x8 -> 4x4
                        ResConvBlock(c * 4, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 4),
                        nn.Upsample(scale_factor=2),
                    ]),  # 4x4 -> 8x8
                    ResConvBlock(c * 8, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 2),
                    nn.Upsample(scale_factor=2),
                ]),  # 8x8 -> 16x16
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=2),
            ]),  # 16x16 -> 32x32
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout_last=False),
        )

    def forward(self, input, log_snrs, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(log_snrs[:, None]), input.shape)
        class_embed = expand_to_planes(self.class_embed(cond), input.shape)
        return self.net(torch.cat([input,class_embed, timestep_embed], dim=1))

# Define the noise schedule and sampling loop

def get_alphas_sigmas(log_snrs):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given the log SNR for a timestep."""
    return log_snrs.sigmoid().sqrt(), log_snrs.neg().sigmoid().sqrt()


def get_ddpm_schedule(t):
    """Returns log SNRs for the noise schedule from the DDPM paper."""
    return -torch.special.expm1(1e-4 + 10 * t**2).log()


@torch.no_grad()
def sample(model, x, steps, eta, classes):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * log_snrs[i], classes).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


# Visualize the noise schedule

%config InlineBackend.figure_format = 'retina'
plt.rcParams['figure.dpi'] = 100

t_vis = torch.linspace(0, 1, 1000)
log_snrs_vis = get_ddpm_schedule(t_vis)
alphas_vis, sigmas_vis = get_alphas_sigmas(log_snrs_vis)

print('The noise schedule:')

plt.plot(t_vis, alphas_vis, label='alpha (signal level)')
plt.plot(t_vis, sigmas_vis, label='sigma (noise level)')
plt.legend()
plt.xlabel('timestep')
plt.grid()
plt.show()

plt.plot(t_vis, log_snrs_vis, label='log SNR')
plt.legend()
plt.xlabel('timestep')
plt.grid()
plt.show()


# Prepare the dataset

batch_size = 100
img_size = 32
tf = transforms.Compose([
    transforms.Resize(size=(img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
train_set = datasets.CIFAR10('data', train=True, download=True, transform=tf)
train_dl = data.DataLoader(train_set, batch_size, shuffle=True,
                           num_workers=4, persistent_workers=True, pin_memory=True)
val_set = datasets.CIFAR10('data', train=False, download=True, transform=tf)
val_dl = data.DataLoader(val_set, batch_size,
                         num_workers=4, persistent_workers=True, pin_memory=True)


# Create the model and optimizer
# sidenote:
# I noticed when I trained the plain version (i.e with no conditioning, and timestep)
# as we trained more,the images kept getting more blurier/grimish/darker! possibly
# signifying overfitting! this is clearly visible by viewing the saved samples in 
# /imgs_gen_newarch_20240408_21_47_07 directory where contains the samples for
# our plain version. I trained for 1000 epochs, lets now go for the next round
# 
# now im going to enble class conditioning part and see how that affects the results
# it creates better image, but as we train more, the images seem toget   blurier/grimish/darker
# basically the quality decreases, up around 150 epochs, the images seem vibrant and 
# somewhat distinguishable,but as we train more, they become, worse! (sometimes better though!
# look at the result you;ll see the network fixes some images while ruins others)
# it seems the model is overfitting! also  size=32x32 shouldbe  enough, if images are developed
# properly,they should be clear! if not it means the model hasnt learned properly!
# so for the next round im going to use img_size=32x32 to make training much faster!
# 720 epochs took around 12 hours now!(with img-size=64) see the result in (imgs_gen_newarch_20240409_14_03_45)
#
# now im going to enable thetime step part and see how that affects the results 
# right off the bat, it seems the images are more vibrant and staying more vibrant than 
# the plain mode where no conditioning (class or timestep) was used. see up to epoch 250
# for this run(imgs_gen_newarch_20240410_07_42_35) vs the plain version(imgs_gen_newarch_20240408_21_47_07)
# the results are just way better, even at 750 epochs, they are still vibrant and resemble
# actual images. so the timestep is absolutely important.(by the way it took around 6 hours to do 1025 epochs!) 
#
# now im going to enable both conditionings (class conditioning and timestep) and see
# how it affects the result, it improved the results, using conditioning 
# we can specify the class we want to generate and hence we can better examine the output
# using the timesteps, it gives us vibrant colors and as we continue, we achieve better image generations
# compare the images in /imgs_gen_newarch_20240410_13_43_51 which are trained with 
# timesteps embedding only with our new results in /imgs_gen_newarch_20240410_07_42_35
# which contains class embeddings as well. the quality is the same, and class embedding
# allows us to generate images for each class individually which is very helpful in generation
# but for the model itself, the timesteps is needed but class emebedding is optional!
# we achieved a loss of 0.032852 in epoch 1058.
#
#
# next lets disable all the dropouts in all layers and see how that affects the output, whether what we see as grimish images
# linked to overfitting or not!
# 
#
seed = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.manual_seed(0)

model = Diffusion().to(device)
model_ema = deepcopy(model)
print(f'model: {model}')
print('Model parameters:', sum(p.numel() for p in model.parameters()))

opt = optim.Adam(model.parameters(), lr=2e-4)
scaler = torch.cuda.amp.GradScaler()
epoch = 0

# Use a low discrepancy quasi-random sequence to sample uniformly distributed
# timesteps. This considerably reduces the between-batch variance of the loss.
rng = torch.quasirandom.SobolEngine(1, scramble=True)


# Actually train the model

ema_decay = 0.998

# The number of timesteps to use when sampling
steps = 500
losses = []
# The amount of noise to add each timestep when sampling
# 0 = no noise (DDIM)
# 1 = full noise (DDPM)
eta = 1.
fldr = "/media/hossein/SSD1/code_dl"
current_time = datetime.now().strftime('%Y%m%d_%H_%M_%S')
img_dir_path = f"{fldr}/imgs_gen_newarch_{current_time}/"

if not os.path.exists(img_dir_path):
    os.makedirs(img_dir_path)
    print(f"directory '{img_dir_path}' created!")
    
def eval_loss(model, rng, reals, classes):
    # Draw uniformly distributed continuous timesteps
    t = rng.draw(reals.shape[0])[:, 0].to(device)

    # Calculate the noise schedule parameters for those timesteps
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)
    weights = log_snrs.exp() / log_snrs.exp().add(1)

    # Combine the ground truth images and the noise
    alphas = alphas[:, None, None, None]
    sigmas = sigmas[:, None, None, None]
    noise = torch.randn_like(reals)
    noised_reals = reals * alphas + noise * sigmas
    targets = noise * alphas - reals * sigmas

    # Compute the model output and the loss.
    with torch.cuda.amp.autocast():
        v = model(noised_reals, log_snrs, classes)
        return (v - targets).pow(2).mean([1, 2, 3]).mul(weights).mean()


def train():
    for i, (reals, classes) in enumerate(tqdm(train_dl)):
        opt.zero_grad()
        reals = reals.to(device)
        classes = classes.to(device)

        # Evaluate the loss
        loss = eval_loss(model, rng, reals, classes)

        # Do the optimizer step and EMA update
        scaler.scale(loss).backward()
        scaler.step(opt)
        ema_update(model, model_ema, 0.95 if epoch < 20 else ema_decay)
        scaler.update()

        if i % 50 == 0:
            tqdm.write(f'Epoch: {epoch}, iteration: {i}, loss: {loss.item():g}')


@torch.no_grad()
@torch.random.fork_rng()
@eval_mode(model_ema)
def val():
    tqdm.write('\nValidating...')
    torch.manual_seed(seed)
    rng = torch.quasirandom.SobolEngine(1, scramble=True)
    total_loss = 0
    count = 0
    for i, (reals, classes) in enumerate(tqdm(val_dl,leave=False)):
        reals = reals.to(device)
        classes = classes.to(device)

        loss = eval_loss(model_ema, rng, reals, classes)

        total_loss += loss.item() * len(reals)
        count += len(reals)
    loss = total_loss / count
    losses.append(loss)
    tqdm.write(f'Validation: Epoch: {epoch}, loss: {loss:g}')


@torch.no_grad()
@torch.random.fork_rng()
@eval_mode(model_ema)
def demo():
    tqdm.write('\nSampling...')
    torch.manual_seed(seed)

    noise = torch.randn([100, 3, img_size, img_size], device=device)
    fakes_classes = torch.arange(10, device=device).repeat_interleave(10, 0)
    fakes = sample(model_ema, noise, steps, eta, fakes_classes)

    grid = utils.make_grid(fakes, 10).cpu()
    filename = f'{img_dir_path}/demo_{epoch:05}.png'
    TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(filename)
    display.display(display.Image(filename))
    tqdm.write('')


def save():
    filename = 'cifar_diffusion_plain_class_condition_and_timestep.pth'
    obj = {
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
    }
    torch.save(obj, filename)

epochs = 2000
try:
    val()
    demo()
    while epoch<epochs:
        print('Epoch', epoch)
        train()
        epoch += 1
        if epoch % 5 == 0:
            val()
            demo()
        save()
except KeyboardInterrupt:
    pass
#%%
try:
    val()
    demo()
except KeyboardInterrupt:
    pass
#%%
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
# import wandb

device = "cuda"

#define ResNet style convolutional block for UNET
class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


# process and downscale the image feature maps
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# process and upscale the image feature maps
class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

#define embedding layer 
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


#implement Context UNET 
class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(8), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )
    
    #implement multi hot enconding function to produce multi category images 
    def one_hot(self,c,num_classes):
        c_return = torch.zeros(len(c),num_classes)
        for i,value in enumerate(c):
            c_return[i, value] = 1.0
        return c_return     

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = self.one_hot(c, self.n_classes)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c.to(device)* context_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)


        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1+ temb1, down2) 
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


# returns pre-computed schedules for DDPM sampling, training process.
def ddpm_schedules(beta1, beta2, T):

    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "log_alpha_t": log_alpha_t,
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


#implement diffusion model
class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))
    
    def sample_single(self, n_sample, size, device, c, guide_w = 0.0):
           
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
       
        #c = c.repeat(int(n_sample/c.shape[0]))
    
        # don't drop context at test time
        context_mask = torch.zeros(1).to(device)
        if c.dim()== 1:
            c = c.repeat(2)
        else:
            c = c.repeat(2,1)

        context_mask = context_mask.repeat(2)

        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, * size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


    def sample(self, n_sample, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


def train_cifar():

    #define hyperparameters
    n_epoch = 1
    batch_size = 256
    n_T = 400 # 500
    device = "cuda"
    n_classes = 10
    n_feat = 128 
    lrate = 1e-4
    save_model = True
    save_dir = './louisdata/diffusion_outputs10/'
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #instantiate model
    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor()]) 

    #load dataset
    dataset = CIFAR10("./datapics", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (3, 32, 32), device, guide_w=w)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")
        # optionally save model
        if save_model and ep%1 == 0:
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train_cifar()



#%%

import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt='%I:%M:%S')

import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(image_size,dataset_path,batch_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    # dataset = torchvision.datasets.CIFAR10(dataset_path,False,download=True,transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        # Downsample
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        # bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # upsample
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        # Downsample
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        # bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # upsample
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # this algorithm is from the paper
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(run_name, epochs,batch_size,image_size, dataset_path, device = "cuda", lr = 3e-4):
    setup_logging(run_name)
    device = device
    dataloader = get_data(image_size, dataset_path, batch_size)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", run_name))
    l = len(dataloader)

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))


fldr = "/media/hossein/SSD1/code_dl"
def launch():
    train(run_name="DDPM_Uncondtional", epochs = 500,
          batch_size = 2, image_size = 64, dataset_path = f"{fldr}/data/landscape_dataset",
          device = "cuda",lr = 3e-4)

launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("models/DDPM_Uncondtional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # for i in range(3):
    #     x = diffusion.sample(model, 8)
    #     print(x.shape)
    #     plt.figure(figsize=(32, 32))
    #     plt.imshow(torch.cat([
    #         torch.cat([i for i in x.cpu()], dim=-1),
    #     ], dim=-2).permute(1, 2, 0).cpu())
    #     plt.axis('off')
    #     plt.savefig(f'samples/ddpm_unconditional_{i}.png', bbox_inches='tight')
    # plt.show()






























#%%
# let's break down the process of how a diffusion model works step by step:
# 1. The process begins with an initial image. This could be a random noise 
#    image or a specific image provided as input.
# 2. In the first step, a small amount of noise is added to the image. 
#    This is done to introduce variability and prevent the model from 
#    simply memorizing the training data. The amount of noise added is
#    controlled by a noise schedule, which determines how much noise to
#    add at each step.
# 3. After the noise has been added, the model's task is to remove this 
#    noise and recover the original image. This is done using a denoising
#    model, which is a neural network trained to predict the original image
#    given the noisy image. The denoising model is applied iteratively, with
#    each iteration making the image less noisy.
# 4. Steps 2 and 3 are repeated multiple times. With each iteration, the 
#    image becomes less noisy and more like the original image. The number
#    of iterations is typically a hyperparameter that is set before training
#    begins.
# 5. After a certain number of iterations, the denoising process is stopped. 
#    The image at this point is the final output of the diffusion model. It 
#    should be a clean, denoised version of the original image.
# 6. If the diffusion model is being used for text-to-image generation, a text
#    encoder may be used to convert text prompts into a format that the model 
#    can understand. This is typically done before the noise addition step, 
#    and the output of the text encoder is used to guide the denoising process.
# 7. Some diffusion models also include an autoencoder, which is a type of 
#    neural network that can learn to compress and decompress data. 
#    The autoencoder is used to convert the image into a lower-dimensional 
#    representation before the noise addition step, and then to reconstruct
#    the image from this representation after the denoising step.
# In summary, a diffusion model works by adding noise to an image and then iteratively denoising the image until it resembles the original image. This process allows the model to generate new images that are similar to the training data.
# %%


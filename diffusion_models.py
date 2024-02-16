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

import matplotlib
import matplotlib.pyplot as plt
import PIL 
import PIL.Image as Image 

import torch
import torch.nn as nn
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
# 
#%%
# ref https://huggingface.co/docs/diffusers/en/tutorials/basic_training
# 
# 
# 
# 
# 

# %%

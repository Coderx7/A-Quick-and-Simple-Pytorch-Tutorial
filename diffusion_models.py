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
# text2image = StableDiffusionPipeline.from_pretrained(repo_model_name,
#                                                 variant='fp16',
#                                                 torch_dtype= torch.float16,
#                                                 # incase our download is interuppted due to bad connection
#                                                 # lets resume from where we left off last time
#                                                 resume_download=True).to('cuda')
# for sdxl-base-1.0 DiffusionPipeline works!
text2image = DiffusionPipeline.from_pretrained(repo_model_name,
                                                variant='fp16',
                                                torch_dtype= torch.float16,
                                                # incase our download is interuppted due to bad connection
                                                # lets resume from where we left off last time
                                                resume_download=True).to('cuda')
#
# by the way we could also use AutoPipelineForText2Image as well, everything stays the same!
# text2image = AutoPipelineForText2Image.from_pretrained(repo_model_name,
#                                                 variant='fp16',
#                                                 torch_dtype= torch.float16,
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
# text2image.enable_attention_slicing("max")
#
# now we simply pass our textual prompt, describing the image we want and get the result
# the output is a dictionary with two keys, "images" in pil format, and "nsfw_content_detected"
# which is a boolean value denoting whether ther result contains anything not safe for work and if
# it does, it returns a black image instead! (its detection can not be trusted though! its not accurate)
prompt = "a beautiful day in a lush forest"
result = text2image(prompt)
print(f'{result.nsfw_content_detected=}')
plt.imshow(result.images[0])
plt.show()
#%%
# now if we want we can send a list of prompts and get a list of images
prompt = ["a beautiful day in a lush forest","a dog sleeping at the beach"]
result = text2image(prompt)
print(f'{result.nsfw_content_detected=}')
def display_images(prompt, result):
    for msg, img in zip(prompt,result.images):
        plt.imshow(img)
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
# The "vae" ([`AutoencoderKL`]) is  the Variational Auto-Encoder (VAE) model to encode and decode 
#   images to and from latent representations.
# The "text_encoder" ([`~transformers.CLIPTextModel`]) is the Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
# The "tokenizer" ([`~transformers.CLIPTokenizer`]) is a `CLIPTokenizer` to tokenize text.
# The "unet" ([`UNet2DConditionModel`]) is a `UNet2DConditionModel` to denoise the encoded image latents.
# The scheduler ([`SchedulerMixin`]) is a scheduler to be used in combination with `unet` to denoise 
#   the encoded image latents. 
#   It can be one of:
#   [`DDIMScheduler`],
#   [`LMSDiscreteScheduler`],
#   or [`PNDMScheduler`].
#
# The "safety_checker" ([`StableDiffusionSafetyChecker`]) is a Classification module that estimates 
#   whether generated images could be considered offensive or harmful.
# The "feature_extractor" ([`~transformers.CLIPImageProcessor`]) is a `CLIPImageProcessor` to extract
# features from generated images, its used as inputs to the `safety_checker`.
# The "requires_safety_checker" is a boolean value specifying whether to run safty check on the output or not
#    the default is Ture. 
# we ignore the ones with _, as they are implementation details and we will get to them later on inshaallah.
# 
# this command in jupyeter notebook allows us to see the implementation details of our pipeline
# which if you have a look at, will find a lot of useful comments concerning how certain sections work!
??text2image
# and we can see a few intersting arguments we can utilize to have more control on the result
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
prompt = ["an 8k photorealistc photography of a horse in a lush forest",
          "a Vintage-style photo of a family playing at the beach"]
# resolution also plays an important role in good outcome, the larger the better!
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
timesteps = None
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
    "ugly low quality extra limbs bad anatomy poorly rendered face deformed bad anatomy "
    "bad proportions blurry cloned face cropped disfigured duplicate out of frame "
    "extra arms extra fingers extra legs fused fingers gross proportions long neck "
    "lowres malformed limbs missing arms missing legs morbid mutated hands mutation mutilated "
    "poorly drawn face poorly drawn hands too many fingers watermark worst quality"]*len(prompt) 
# we can generate multiple images for a single prompt and chose all or the best one if we wish
num_images_per_prompt = 1
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

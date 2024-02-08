# in the name of God the most compassionate the most merciful 
# geremey howard part 9,9a and 9b and 10 watch them first to get the initial idea. 
# then feel free to watch any of the following:
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
#%%
# lets import our modules that we may need 
import os
import sys
import numpy

import matplotlib.pyplot as plt 
import PIL.Image as Image 

import torch
import torch.nn as nn 

import transformers as trans
import evaluate 
import diffusers as dfs
#
# in the first step we are going to see how we can use huggingface diffusers library/module to 
# create/finetune/train diffusion models 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

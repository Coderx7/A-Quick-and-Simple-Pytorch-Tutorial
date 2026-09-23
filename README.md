# A-Quick-and-Simple-Pytorch-Tutorial
This repo will contain simple tutorials for getting familiar with Pytorch quickly for beginners. 
This actually acts as a personal note for myself as well, as I review my old recollection of different algorithms and concepts
in Pytorch. I tried to explain everything so in case later on I forget something, I can quickly recall it or see  the refs I 
find useful. 
I'll tidy things up when I get the time, the following section will be updated as I finish different parts. 
Some sections are already done (e.g. syle transfer, RNNs, GANs, but they need full explanations, so when that is done, I'll push the changes to this repo. 
Hope this comes handy to some of you dear fellow software engineers/deeplearning researchers. 
Have a wonderful day/night :)

Subjcts : 
- [x] Introduction to Pytorch basics
- [x] Introduction on Networks:
  - [x] training and testing (including augmentation)
  - [x] changing and finetuning architectures
  - [x] saving and loading models
- Autoencoders
  - [x] Autoencoder(AE)
  - [x] Deep MLP Autoencoder(MLPAE)
  - [x] Convolutional Autoencoder(ConvAE)
  - [x] Sparse Autoencoder(SAE) (l1penalty, kldivergance)
  - [x] Denoising Autoencoder(DAE)
  - [x] Contractive Autoencoder(CAE)
  - [x] Variational Autoencoder(VAE)
  - [x] Conditional Variational Autoencoder(Cond-VAE)
  - [x] Disentagled(beta) Variational Autoencoder(B-VAE)
  - To do: 
    [x] Sequence to Sequence Autoencoder
    [x] Cyclical Annealing Schedule 
- [x] MultiTask Learning 
- [x] GANs (GAN, DCGAN, CGAN, CycleGAN, StarGAN, StyleGAN, WGAN, etc) 
- [x] RNNs(RNN, LSTM, GRU) (NLP and Vision)
  - [x] Text Generation
  - [x] Sentiment Analysis 
  - [x] Seq2Seq
  - [x] Attention Mechanism 
  - [x] Transformers
  - [x] Image Captioning 
  - [-] CTC Loss
  - [x] Word Embedding 
  - [-] NER(Named Entity Recognition) 
  - [x] Misc 
- [x] Style transfer 
- [x] Adversarial Attacks (Examples)
- [x] Object Detection 
- [x] Semantic Segmentation 
- [x] Siamese Networks 
- [x] Autograd introduction 
- [x] Datasets Introduction 
- Misc
  - [x] Concepts



### Update 2026:  

I finally found the time to merge the updates that were long over due,   
they were originally supposed to follow shortly after my first commit here,  
But personal reasons got in the way.  

This update brings in the tutorials I originally wrote for myself back in 2018/2019,   
I didnt plan on publishing them publicly initially, as they were only my  
practice files, reminders and definitely not in a shape fit for others hence why  
It took a fair amount of work to consolidate and clean them up into something usable.  

While organizing them, I also added a few more architectures, so the repo now covers  
a reasonable range of topics, all heavily commented and explained.  

Each file/section/chapter includes the references I used while writing it. aside  
from the original papers and their implementations, many of the exercises   
and projects here are simply me implementing concepts I was learning at the time  
some inspired by online courses such as Coursera (deeplearning course) or   
Udacity (Pytorch introduction) circa 2017~2019.   
Please refer to each file for further references.   

I used LLMs such as ChatGPT, Gemini, Claude, Microsoft Copilot, Grok, and DeepSeek as   
a secondary aid throughout my work post-2023. I mainly used these for debugging, proofreading,  
and sanity-checking my explanations, mostly concerning newer architectures that I had not   
previously implemented, such as new GAN architectures, and, to a lesser degree, my existing  
explanations especially when I had made an incorrect assumption or explanation that needed to be corrected.  

Some chapters need more edits/proofing than others, but you should be able to read/experiment  
nonetheless. hopefully I'll take care of them when my time allows me and do a cleanup later on.  


[TOPICS.md](./TOPICS.md) contains an auto generated summary of what these tutorials cover  
I generated it using claude. I'll leave the original readme as it shows the majority  
of topics being covered here.   

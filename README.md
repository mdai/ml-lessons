# Introductory lessons to Deep Learning for medical imaging by [MD.ai](https://www.md.ai)

The following are several Jupyter notebooks covering the basics of downloading and parsing annotation data, and training and evaluating different deep learning models for classification, semantic and instance segmentation and object detection problems in the medical imaging domain. The notebooks can be run on Google's colab with GPU (see instruction below).  

- Lesson 1. Classification of chest vs. adominal X-rays using TensorFlow/Keras [Github](https://github.com/mdai/ml-lessons/blob/master/lesson1-xray-images-classification.ipynb) | [Annotator](https://public.md.ai/annotator/project/PVq9raBJ)
- Lesson 2. Lung X-Rays Semantic Segmentation using UNets. [Github](https://github.com/mdai/ml-lessons/blob/master/lesson2-lung-xrays-segmentation.ipynb) |
[Annotator](https://public.md.ai/annotator/project/aGq4k6NW/workspace) 
- Lesson 3. RSNA Pneumonia detection using Kaggle data format [Github](https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-kaggle.ipynb) | [Annotator](https://public.md.ai/annotator/project/LxR6zdR2/workspace) 
- Lesson 3. RSNA Pneumonia detection using MD.ai python client library [Github](https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-mdai-client-lib.ipynb) | [Annotator](https://public.md.ai/annotator/project/LxR6zdR2/workspace)

*Note that the mdai client requires an access token, which authenticates you as the user. To create a new token or select an existing token, to go a specific MD.ai domain (e.g., public.md.ai), register, then navigate to the "Personal Access Tokens" tab on your user settings page to create and obtain your access token.*

## The MD.ai Annotator 
The MD.ai annotator is a powerful web based application, to store and view anonymized medical images (e.g, DICOM) on the cloud, create annotations collaboratively, in real-time, and export annotations, images and labels for training. The MD.ai python client library can be used to download images and annotations, prepare the datasets, and then used to train and evaluate deep learning models. Further documentation and videos are available at https://docs.md.ai/

- MD.ai Annotator Example Project URL: https://public.md.ai/annotator/project/aGq4k6NW/workspace
- MD.ai python client libray URL: https://github.com/mdai/mdai-client-py

![MD.ai Annotator](https://docs.md.ai/img/annotator_homepage.png)

## Running Jupyter notebooks Colab  

It’s easy to run a Jupyter notebook on Google's Colab with free GPU use (time limited).  
For example, you can add the Github jupyter notebook path to https://colab.research.google.com/notebook: 
Select the "GITHUB" tab, and add the Lesson 1 URL: https://github.com/mdai/ml-lessons/blob/master/lesson1-xray-images-classification.ipynb

To use the GPU, in the notebook menu, go to Runtime -> Change runtime type -> switch to Python 3, and turn on GPU.  See more [colab tips.](https://www.kdnuggets.com/2018/02/essential-google-colaboratory-tips-tricks.html)

## Advanced: How to run on Google Cloud Platform with Deep Learning Images

You can also run the notebook with powerful GPUs on the Google Cloud Platform. In this case, you need to authenticate to the Google Cloug Platform, create a private virtual machine instance running a Google's Deep Learning image, and import the lessons. See instructions below. 

[GCP Deep Learnings Images How To](running_on_gcp.md)

---

&copy; 2018 MD.ai, Inc.  
Licensed under the Apache License, Version 2.0

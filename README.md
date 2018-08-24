# Introductory lessons to Deep Learning for medical imaging by [MD.ai](https://www.md.ai). 

The following are few Jupyter notebooks covers the basics of parsing annotation data (using our annotator or Kaggle competition data formats), training several different deep learning models for classification, semantic and instance segmentation and object detection problems in the medical imaging domain.  

- Lesson 1. Classification of chest vs. adominal X-rays using TensorFlow/Keras [Github](https://github.com/mdai/ml-lessons/blob/master/lesson1-xray-images-classification.ipynb) [Annotator](https://public.md.ai/annotator/project/PVq9raBJ)

- Lesson 2. Lung X-Rays Semantic Segmentation using UNets. [Github](https://github.com/mdai/ml-lessons/blob/master/lesson2-lung-xrays-segmentation.ipynb)
[Annotator](https://public.md.ai/annotator/project/aGq4k6NW/workspace) 

- Lesson 3. RSNA Pneumonia detection using Kaggle data format [Github](https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-kaggle.ipynb) [Annotator](https://public.md.ai/annotator/project/LxR6zdR2/workspace) 
  
- Lesson 3. RSNA Pneumonia detection using MD.ai python client library [Github](https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-mdai-client-lib.ipynb) [Annotator](https://public.md.ai/annotator/project/LxR6zdR2/workspace)

*Note that the mdai client requires an access token, which authenticates you as the user. To create a new token or select an existing token, navigate to the "Personal Access Tokens" tab on your user settings page at the specified MD.ai domain (e.g., public.md.ai).*

## The MD.ai Annotator 
The MD.ai annotator a web-base tool is used to store and view anonymized medical images (e.g, DICOM) on the cloud, create annotations collaboratively, in real-time, and export annotations, images and labels for training. The MD.ai python client library can be used to download images and annotations, prepare the datasets, and then used to train and evaluate deep learning models. Further documentation and videos are available at https://docs.md.ai/

- MD.ai Annotator Example Project URL: https://public.md.ai/annotator/project/aGq4k6NW/workspace
- MD.ai python client libray URL: https://github.com/mdai/mdai-client-py

![MD.ai Annotator](/images/annotator.png)

## Running Jupyter notebooks Colab  

It’s easy to create run a Jupyter notebook on Google's Colab with free GPU use (time limited). 
Just add your Github path to https://colab.research.google.com/notebook, select "GITHUB" tab, and enter the following lesson 1 URL: https://github.com/mdai/ml-lessons/blob/master/lesson1-xray-images-classification.ipynb. 

And in order to use the GPU, in the notebook menu, go to Runtime -> Change runtime type -> switch to Python 3, and turn on GPU.  See more [Colab tips](https://www.kdnuggets.com/2018/02/essential-google-colaboratory-tips-tricks.html)

## Advanced: How to run on Google Cloud Platform with Deep Learning Images

[GCP Deep Learnings Images How To](running_on_gcp.md)

---

&copy; 2018 MD.ai, Inc.  
Licensed under the Apache License, Version 2.0

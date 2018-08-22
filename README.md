# ml-lessons

Introductory Deep learning/ML lessons for medical images by [MD.ai](https://www.md.ai). They can be run on Google's Colab Jupyter notebook server, using the GPU, for free! 

- Lesson 1. Classification of chest vs. adominal X-rays using TensorFlow/Keras [Launch in Colab](https://colab.research.google.com/github/mdai/ml-lessons/blob/master/lesson1-xray-images-classification.ipynb) 
- Lesson 2. Lung X-Rays Semantic Segmentation using UNets. [Launch in Colab](https://colab.research.google.com/github/mdai/ml-lessons/blob/master/lesson2-lung-xrays-segmentation.ipynb)
- Lesson 3. 
  - RSNA Pneumonia detection using Kaggle data format [Launch in Colab](https://colab.research.google.com/github/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-kaggle.ipynb) 
  - RSNA Pneumonia detection using MD.ai python client library [Launch in Colab](https://colab.research.google.com/github/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-mdai-client-lib.ipynb) 

## MD.ai Annotator 
The MD.ai annotator is used to view the DICOM images, and to create the image level annotation. The MD.ai python client library is then used to download images and annotations, prepare the datasets, then are then used to train the model for classification. 

![MD.ai Annotator](/images/annotator.png)

## Colab Tips 
In order to use the GPU, in the menu, go to Runtime -> Change runtime type -> switch to Python 3, and turn on GPU.*
[Colab tips](https://www.kdnuggets.com/2018/02/essential-google-colaboratory-tips-tricks.html)

## Advanced: How to run on Google Cloud Platform with Deep Learning Images

[GCP Deep Learnings Images How To](running_on_gcp.md)

---

&copy; 2018 MD.ai, Inc.  
Licensed under the Apache License, Version 2.0

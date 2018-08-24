# Introductory Deep learning/ML lessons for medical images by [MD.ai](https://www.md.ai).

**Note that the mdai client requires an access token, which authenticates you as the user. To create a new token or select an existing token, navigate to the "Personal Access Tokens" tab on your user settings page at the specified MD.ai domain (e.g., public.md.ai).**

- Lesson 1. Classification of chest vs. adominal X-rays using TensorFlow/Keras [Github](https://github.com/mdai/ml-lessons/blob/master/lesson1-xray-images-classification.ipynb) [Annotator](https://public.md.ai/annotator/project/PVq9raBJ)

- Lesson 2. Lung X-Rays Semantic Segmentation using UNets. [Github](https://github.com/mdai/ml-lessons/blob/master/lesson2-lung-xrays-segmentation.ipynb)
[Annotator](https://public.md.ai/annotator/project/aGq4k6NW/workspace) 

- Lesson 3. RSNA Pneumonia detection using Kaggle data format [Github](https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-kaggle.ipynb) [Annotator](https://public.md.ai/annotator/project/LxR6zdR2/workspace) 
  
- Lesson 3. RSNA Pneumonia detection using MD.ai python client library [Github](https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-mdai-client-lib.ipynb) [Annotator](https://public.md.ai/annotator/project/LxR6zdR2/workspace)

## MD.ai Annotator 
The MD.ai annotator is used to view the DICOM images, and to create the image level annotation. The MD.ai python client library is then used to download images and annotations, prepare the datasets, then are then used to train the model for classification. 
- MD.ai Annotator example URL: https://public.md.ai/annotator/project/aGq4k6NW/workspace
- MD.ai python client libray URL: https://github.com/mdai/mdai-client-py
- MD.ai documentation URL: https://docs.md.ai/


![MD.ai Annotator](/images/annotator.png)

## Colab Tips 
They can be run on Google's Colab Jupyter notebook server, using the GPU, for free! Just go to colab's website, and enter a notebook's github URL. 

In order to use the GPU, in the menu, go to Runtime -> Change runtime type -> switch to Python 3, and turn on GPU.  
[Colab tips](https://www.kdnuggets.com/2018/02/essential-google-colaboratory-tips-tricks.html)

## Advanced: How to run on Google Cloud Platform with Deep Learning Images

[GCP Deep Learnings Images How To](running_on_gcp.md)

---

&copy; 2018 MD.ai, Inc.  
Licensed under the Apache License, Version 2.0

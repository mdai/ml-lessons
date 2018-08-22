# ml-lessons

Introductory Deep learning/ML lessons for medical images by [MD.ai](https://www.md.ai). 

- Lesson 1. Classification of chest vs. adominal X-rays using TensorFlow/Keras. 
- Lesson 2. Lung X-Rays Semantic Segmentation using UNets. 
- Lesson 3. RSNA Pneumonia detection. 

## MD.ai Annotator 
The MD.ai annotator is used to view the DICOM images, and to create the image level annotation. The MD.ai python client library is then used to download images and annotations, prepare the datasets, then are then used to train the model for classification. 

![MD.ai Annotator](/images/annotator.png)

## How to run on Colab 

[Colab](https://colab.research.google.com/) -> File->Open Notebook, Select Github tab, and paste the url of a lesson, e.g., use 
`https://github.com/mdai/ml-lessons/blob/master/lesson1-xray-images-classification.ipynb` for lesson 1. 

[Colab tips](https://www.kdnuggets.com/2018/02/essential-google-colaboratory-tips-tricks.html)

## Advanced: How to run on Google Cloud Platform with Deep Learning Images

[GCP Deep Learnings Images How To](running_on_gcp.md)

---

&copy; 2018 MD.ai, Inc.  
Licensed under the Apache License, Version 2.0

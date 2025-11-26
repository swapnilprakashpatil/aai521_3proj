# SpaceNet 8: Flood Detection Challenge Using Multiclass Segmentation using Computer Vision and Deep Learning

# Docs: https://spacenet.ai/sn8-challenge/

## Overview

The SpaceNet 8 Challenge focuses on flood detection using satellite imagery. Participants are tasked with developing models that can accurately identify flooded areas in high-resolution satellite images. The challenge involves multiclass segmentation, where the goal is to classify each pixel in the image into different categories such as water, land, and flooded areas.

## Objectives

- Develop a deep learning model capable of performing multiclass segmentation on satellite images.
- Accurately identify and classify flooded areas in the images.
- Evaluate model performance using appropriate metrics for segmentation tasks.

## Dataset

The dataset for the SpaceNet 8 Challenge consists of high-resolution satellite images captured before and after flood events. The images are annotated with pixel-level labels indicating different classes such as water, land, and flooded areas. Participants can access the dataset through the SpaceNet website.

# Training Data: /dataset/train/ contains tif images Pre event and Post Event and annotations from Germany and Louisiana Eastern USA

/dataset/train/Louisiana-East_Training_Public
/dataset/train/Germany_Training_Public

# Test Data: /dataset/test/ contains tif images Pre event and Post Event from Louisiana USA

/dataset/test/

## Instructions

1. Create a new notebook in your Jupyter environment.
2. Load the training dataset from the specified directory.
3. Preprocess the images and annotations for training.

# EDA - Exploratory Data Analysis

- Explore the dataset to understand the distribution of classes, image sizes, and any potential challenges in the data.
- Check for missing or corrupted images and annotations.
- Analyze the distribution of pixel values across different classes.
- Assess the variability in lighting conditions, weather, and seasonal changes in the images.
- Evaluate the alignment between pre-event and post-event images.
- Check the annotation quality and consistency across the dataset.
- Identify any potential outliers or anomalies in the data.
- Perform statistical analysis on the dataset to summarize key characteristics.
- Generate histograms and visualizations to understand class distributions.
- Create summary statistics for image dimensions and resolutions.
- Explore correlations between different classes and features in the images.
- Generate sample visualizations of images with their corresponding annotations.
- Create a checklist for data quality and completeness.
- Identify any potential biases in the dataset.
- Explore temporal aspects if multiple time points are available.
- Investigate the geographic distribution of the images if location data is available.
- Analyze the impact of different weather conditions on image quality.
- Explore the presence of shadows and their effect on segmentation.
- Visualize sample images and their corresponding annotations to get insights into the data.
- Check for class imbalance and consider techniques to address it if necessary.
- Analyze the spatial resolution and quality of the satellite images.
- Identify any patterns or features that may help in distinguishing flooded areas from non-flooded areas.
- Summarize findings and insights from the EDA process.
- Document your findings and insights from the EDA process.

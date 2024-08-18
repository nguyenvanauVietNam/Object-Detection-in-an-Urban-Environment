# Object-Detection-in-an-Urban-Environment


#Note:AuNV note
https://sagemaker-us-east-1-251792761470.s3.amazonaws.com/tf2-object-detection-2024-08-17-15-15-49-969/output/model.tar.gz

mobilenet-> upload model.tar.gz to AWS and run training
https://sagemaker-us-east-1-251792761470.s3.amazonaws.com/tf2-object-detection-2024-08-18-14-12-02-225/output/model.tar.gz

ml.t3.xlarge :Type machine  3,825 USD
![Demo](data\animation.gif)

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Training & Deployment Process with AWS](#training--deployment-process-with-aws)
5. [Model Selection](#model-selection)
6. [Results](#results)
7. [Future Work & Possible Improvement](#future-work--possible-improvement)

## Introduction
This project focuses on utilizing transfer learning using the TensorFlow Object Detection API and AWS Sagemaker to train models for detecting and classifying objects in an urban environment. The data used is sourced from the Waymo Open Dataset, providing a robust basis for training.

## Dataset
The dataset consists of front camera images from the Waymo Open Dataset. These data are stored in TFRecord format, which is a simple format for storing a sequence of binary records. The TFRecord format is used to enhance data reading and processing efficiency.

## Methodology
This project uses AWS services to manage the training and deployment process:
- **AWS Sagemaker**: For running Jupyter notebooks, training the model, deploying it, and performing inference.
- **AWS Elastic Container Registry (ECR)**: To build the Docker image and create the container required for running this project.
- **AWS Simple Storage Service (S3)**: Used to save logs for creating visualizations and store the data in a public S3 bucket.

## Training & Deployment Process with AWS
The model training and deployment processes were conducted using AWS services. AWS Sagemaker was utilized to run Jupyter notebooks, facilitating the training, deployment, and inference processes. Docker images were built and containers created using AWS Elastic Container Registry (ECR), while logs were saved to AWS Simple Storage Service (S3) for visualization purposes.

## Model Selection
In this project, several object detection models from the TensorFlow 2 Object Detection Model Zoo were tested:

| Model                 | Config File        |
|-----------------------|--------------------|
| Model_EfficientNet    | [File](Model_EfficientNet\1_model_training\source_dir\pipeline.config) |
| Model_MobileNet | [File](Model_MobileNet\1_model_training\source_dir\pipeline.config) |
| Model_ResNet   | [File](Model_ResNet\1_model_training\source_dir\pipeline.config) |

These pre-trained models, trained on the COCO 2017 dataset, were selected for testing. Modifications were made to the `pipeline.config` files to ensure compatibility with the Waymo Open Dataset, which has 3 classes: Cars, Pedestrians, and Cyclists, as opposed to the 90 classes in the COCO dataset.

For consistency and due to budget constraints, the training steps were limited to 2000 for all models. The same batch size of 8 and Momentum Optimizer were used across the experiments.

## Results
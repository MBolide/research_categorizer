# Article Classifier API

## Overview
This application provides a **Django Rest Framework (DRF) API** that leverages a machine learning model to classify research articles based on their abstract. The goal is to allow users to submit article abstracts and receive predicted categories through a simple API interface.

## Setup

### 1. Install Required Packages
All necessary dependencies are listed in the `requirements.txt` file. To install them, simply run the following command:
```bash
pip install -r requirements.txt
```
### 2. Model and Tokenizer Setup
In order to make predictions, the model, tokenizer, and classification dictionary must be loaded. These files are included in the `model-finetuned` directory, which has been provided as an archive.

**Instructions**:
- Unzip the `model-finetuned` folder.
- Place the unzipped folder inside the `api` directory of the project.
  
This step is crucial.

### 2. Running the Application
Once the setup is complete and the application is running, you will be able to submit article abstracts for classification directly from the application's homepage.

## Usage
After starting the application, simply navigate to the home page where you can input the abstract of an article and get the predicted category in return.

## Limitations

The model has been evaluated using several metrics, including accuracy, precision, recall, F1 score, Hamming score, and Hamming loss (considering that this is a multilabel classification problem)

The evaluation results reveal areas for improvement. While precision scores are relatively high, recall is low, which reduces the F1 score. This indicates that while the model is good at identifying relevant labels when it makes predictions, it misses many true labels, suggesting a need for better balance between precision and recall. Overall, accuracy also needs improvement.

One way to enhance the model’s performance is through hyperparameter tuning. With more time for training and more computational resources, it should be possible to fine-tune the parameters, leading to improved results. 

Additionally, the dataset's imbalanced nature presents a significant challenge. Although this reflects real-world scenarios, it hampers the model's ability to classify underrepresented categories. For instance, some categories appear only six times in the dataset, making it difficult for the model to learn and predict these categories effectively. As a result, the model tends to favor categories it has seen more frequently, and it may struggle with categories it has seen less often, even with hyperparameter optimization.

There are methods to mitigate this imbalance, such as resampling techniques to balance the dataset or applying data augmentation to generate more instances of underrepresented classes. These approaches could help the model achieve better generalization across all categories.

The selection of learning rate and batch size has also impacted the results. It is possible that the model may have become stuck in a local minimum during training. Experimenting with a different optimizer, perhaps one more specialized than Adam, and exploring momentum adjustments, could also lead to better performance.

Finally, another source of potential improvement comes from outside the model itself: the cutoff probability for when a category is considered is currently set at 0.4. This threshold is external to the model but still influences the end result a lot (to get the probabilities we have just applied sigmoid function to the model's end results to put them on the scale between 0 and 1). By lowering this probability threshold, we may be able to classify some of the more underrepresented categories correctly. However, the trade-off here is that this will also increase the false positive rate, making the probability threshold yet another important consideration in the overall performance.

Side Note: It apparently at least classifies correctly the article abstract from the Kaggle page of the dataset :) 


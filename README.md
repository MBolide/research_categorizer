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

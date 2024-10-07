import torch
from transformers import AutoTokenizer, DistilBertModel
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import itertools
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

#Download the needed packages 
nltk.download("stopwords")
nltk.download("wordnet")

# Add our custom model definition
class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 176)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# Initialize and load model
model = DistilBERTClass()
device = torch.device('cpu')
model.load_state_dict(torch.load('api/model-finetuned/distilbert_multilabel_state.pth', map_location=device, weights_only=True))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('api/model-finetuned/Tokenizer')

## Load categories
with open('api/model-finetuned/mlb_classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Initialize a new MultiLabelBinarizer and set its classes_ attribute
mlb = MultiLabelBinarizer()
mlb.classes_ = classes

#Needed for text pre-processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def remove_stop_words(text):
    word_list = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in word_list if word not in stop_words])


def predict_categories(abstract):
     #Preprocess the abstract 
    # Remove the newline characters
    abstract = abstract.replace('\n', ' ').strip()

    # Remove special characters
    abstract = re.sub(r'[^\w\s]', '', abstract)

    # Convert to lowercase
    abstract = abstract.lower()

    #Perform lemmatization
    abstract = remove_stop_words(abstract)

     #Tokenization
    inputs = tokenizer(abstract, return_tensors='pt', truncation=True, padding=True)
    fin_outputs=[]
    # Step 4: Make prediction with the model
    with torch.no_grad():
        outputs = model(**inputs)
        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    #Process the output to predicted categories
    final_outputs = np.array(fin_outputs) >=0.4
    final_outputs = final_outputs.astype(int)
    predicted_categories_nested = mlb.inverse_transform(final_outputs)
    predicted_categories = list(itertools.chain(*predicted_categories_nested))
    if predicted_categories:
        return predicted_categories
    return "No categories detected for this abstract"

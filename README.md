# Natural-Language-Processing-with-Disaster-Tweets
Predict which Tweets are about real disasters and which ones are not using a dataset of 10,000 hand-classified tweets.

## Approach

The approach can be broken down into the following steps:

1. **Microsoft DeBERTa model**: Microsoft DeBERTa model (microsoft/deberta-v3-large) is imported from the transformers library. The tokenizer and model objects are created with the help of the AutoTokenizer and AutoModelForSequenceClassification classes, respectively.
2. **Data Analysis and Exploration**: The training dataset is loaded and explored using Pandas. The target column is converted to float and the missing values are filled with '[N]'. The 'input' column is created by adding the '[CLS]' token to the beginning of the 'text' column.
3. **Data pre-processing**: The Dataset class from the datasets library is used to create a dataset from the training data. A function tok_func is defined to tokenize the input text using the DeBERTa tokenizer. The map method of the Dataset class is used to apply this function to each element of the dataset in parallel.
4. **Model Performance Metrics**: A function compute_metrics is defined to compute the F1 score metric using the load method from the datasets library. This function takes the predicted and actual labels as input and returns the computed metric.
5. **Vectorization of the pre-processed text**: The pre-processed text is converted into token ids using the DeBERTa tokenizer. This is done by applying the map method to the dataset with the tok_func function as an argument.
6. **Building a machine learning model**: The train_model function is defined to train a DeBERTa model on the pre-processed and tokenized data. This function takes the pre-processed dataset, tokenizer, learning rate, epochs, batch size, model name, gradient accumulation steps, reshuffle, epochs2, bs2, gradient accumulation steps2, and fp16 as input parameters. The DeBERTa model is instantiated with the AutoModelForSequenceClassification class and the trainer is created with the Trainer class from the transformers library. The training method of the trainer object is called to train the model.
7. **Predictions**: The trained model is used to make predictions on the test data. The test dataset is loaded and pre-processed in the same way as the training data. The predict method of the trainer object is used to make predictions on the pre-processed test data. The predicted values are thresholded at 0.6 and converted to binary values.

## Implementation Details
The code first reads the training data, performs data cleaning and preprocessing, and then trains the DeBERTa model on the preprocessed data. The model is trained in two stages, with the second stage using a smaller batch size and fewer epochs. Finally, the trained model is used to make predictions on a test set, and the predictions are saved to a CSV file for submission.

**Packages**: transformers, datasets and PyTorch

**External sources of code**: Pre-trained models from the Hugging Face Transformers library - the Microsoft DeBERTa model for sequence classification, functions, and classes from the PyTorch and Datasets libraries.

**Parameters/Architecture**: 
Learning rate: 5e-6

Batch size: 4

Gradient accumulation steps: 1

Number of epochs: 2 (for the first training stage), 1 (for the second training stage)

Weight decay: 0.01

Loss function: binary cross-entropy

Evaluation metric: F1 score

**Model architecture**: Microsoft DeBERTa model for sequence classification with 24 layers and 540M parameters

**Fine-tuning strategy**: The model is first trained on a training set, and then further fine-tuned on a validation set.

## Experiments
* Initially we started training the data with distilbert-base-uncased model (DistilBertTokenizer and TFDistilBertModel) as it is designed to be faster and use less memory and we got accuracy of 83.29%. This was the best accuracy we got after hyperparameter tuning and the parameters were, binary cross-entropy loss and Adam optimizer with a learning rate schedule of exponential decay starting from 1e-5 to 1e-4. 
* Next, we used a much larger and powerful model than BERT - microsoft/deberta-v3-large. DeBERTa model explained above was tried to improve by using different learning rates, batch size, number of epochs and reshuffling. The above described parameters gave the best accuracy of 84.55%. 
* To check if we can get higher accuracy if we cleaned the data, we performed data cleaning by defining special tokens using a predefined list that includes mentions, URLs, and hashtags. These tokens were then added to the tokenizer, and the data was cleaned by removing special characters, replacing URLs, wrapping mentions and hashtags, removing non-ASCII characters, and removing special cases. After cleaning the data, it was trained using the same best performing model mentioned above but it did not improve the accuracy and it was almost the same 84.49%.

## Conclusion
* While the approach described in the proposal is well-structured and involves several key steps, there are still areas for further exploration and improvement.

* One area for potential improvement is data preprocessing. While the current approach involves a few basic preprocessing steps, such as filling missing values and adding special tokens, other techniques like stemming, lemmatization, and stop word removal could also be explored. These preprocessing techniques could help improve the quality of the tokenized text and potentially lead to better model performance.

* Another area for potential improvement is the use of regularization techniques. The current approach uses weight decay as a form of regularization, but other techniques like dropout or L1/L2 regularization could also be tried to prevent overfitting. However, for large datasets, the effectiveness of these techniques may be limited.

* In addition, the current approach relies on a single classification model. Using an ensemble of models, each trained on different subsets of the data, could potentially improve performance and provide more robust predictions.

* Interestingly, cleaning the data by defining special tokens using a predefined list that includes mentions, URLs, and hashtags did not improve the model's accuracy significantly. This result suggests that these things store semantic information about texts and should not be ignored during preprocessing.

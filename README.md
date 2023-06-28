# TextCNN-with-RoBERTa
This model performs a sentimental analysis on the tweets regarding Roe vs Wade

[Link to Sentiment Analysis Report](https://wandb.ai/samyak152002/MYM%20TWEETS%20ANALYSIS/reports/Sentimental-Analysis-Using-RoBERTa-and-TextCNN--Vmlldzo0NzU4MTIz)


Model Development
I have used a BERT model to classify tweets according to Neutral, Positive, and Negative for abortion and fine-tuned it using a Convolutional Neural Network.
The model used in the code is a TextCNN (Convolutional Neural Network) for sentiment analysis. It combines the features of a traditional CNN with pre-trained word embeddings to classify the sentiment of text data.

Here is a high-level overview of the model architecture:

Input Encoding: The text data is tokenized and encoded using the AutoTokenizer provided by the Hugging Face library. This step converts the input text into a sequence of token IDs and attention masks.

Word Embeddings: The model uses pre-trained word embeddings called GloVe (Global Vectors for Word Representation) to represent each token in the input sequence. GloVe embeddings capture semantic relationships between words based on their co-occurrence statistics in a large corpus of text.

Convolutional Layers: The model applies convolutional filters of different kernel sizes to the word embeddings. These filters slide over the input sequence and extract local features, capturing patterns at different scales. The number of filters and kernel sizes are hyperparameters that can be tuned.

Max Pooling: After applying the convolutional filters, the model performs max pooling over the resulting feature maps. Max pooling selects the maximum value within each window of the feature map, reducing the dimensionality and extracting the most salient features.

Fully Connected Layers: The pooled features are flattened and passed through a series of fully connected layers. These layers learn to map the extracted features to the sentiment labels (positive, negative, or neutral). Dropout regularization is applied between the fully connected layers to prevent overfitting.

Training: The model is trained using the AdamW optimizer and the cross-entropy loss function. Class weights are calculated to handle class imbalance in the training data. The model is trained for a specified number of epochs, with batch-wise updates to the weights based on the gradients computed during backpropagation.

Evaluation: The model is evaluated on the validation and test datasets using accuracy, F1 score, and confusion matrix. These metrics provide insights into the model's performance in correctly predicting the sentiment of the text data.

The model also utilizes a pre-trained language model called "cardiffnlp/twitter-roberta-base-stance-abortion" from the Hugging Face library. This language model provides contextualized representations of the input text, which are combined with the TextCNN outputs to improve the overall performance of the sentiment classification task.

Overall, the TextCNN model with pre-trained word embeddings and the addition of contextualized representations from the language model allows for effective sentiment analysis of text data. The combination of convolutional filters, max pooling, and fully connected layers enables the model to capture local and global features, providing valuable insights into the sentiment expressed in the text.

Model Training
Data Preparation: Clean and preprocess the training data, tokenize it, and encode it with token IDs and attention masks.

Word Embeddings: Utilize pre-trained word embeddings like GloVe to capture semantic relationships between words.

Model Architecture: Construct the TextCNN model with convolutional layers, max pooling, fully connected layers, and dropout regularization.

Training Loop: Train the model using the AdamW optimizer, iterating over batches of training examples, calculating loss, and updating parameters.

Class Weights: Account for class imbalance by assigning higher weights to underrepresented classes.

Regularization: Apply dropout regularization between fully connected layers to prevent overfitting.

Fine-tuning: Jointly train the TextCNN model and a pre-trained language model to improve performance.
Model Evaluation
Validation Set: Use a separate validation set during training to monitor performance and tune hyperparameters.

Test Set: Evaluate the trained model on a separate test set to measure its performance on unseen data.

Metrics: Compute evaluation metrics such as accuracy, precision, recall, F1 score, and confusion matrix for both validation and test sets.
Fine Tuning and Optimization
Data Preparation: The code starts by importing the required libraries and loading the dataset using the load_dataset function from the Hugging Face datasets library. The dataset used is "tweet_eval," and the specific task is "stance_abortion". The dataset is split into train, validation, and test sets.

Text Preprocessing: A preprocessing function called preprocess is defined to remove usernames, links, and certain pronouns/articles from the text. This function is applied to the training, validation, and test texts.

Tokenization: The AutoTokenizer class from the Transformers library is used to tokenize the text data. The tokenizer is initialized with the model name "cardiffnlp/twitter-roberta-base-stance-abortion". The train, validation, and test texts are tokenized and encoded using the tokenizer.

Word Embeddings: Pre-trained word embeddings (GloVe) are loaded from a file ("glove.twitter.27B.100d.txt") and stored in the word_embeddings variable. The GloVe embeddings are used to initialize an embedding layer in the model.

Model Architecture: The model architecture is defined using the TextCNN class, which is a convolutional neural network for text classification. The model consists of an embedding layer, convolutional layers with different kernel sizes, max pooling, and fully connected layers. The model takes input size, embed size, kernel sizes, number of channels, and word embeddings as parameters.

Training Loop: The code sets up the training loop with parameters such as learning rate, batch size, and number of epochs. The model and the pre-trained RoBERTa model for sequence classification are moved to the device (GPU if available). The optimizer (AdamW) and the loss function (CrossEntropyLoss) are defined. The training loop iterates over the data loader, computes the forward pass, calculates the loss, performs backpropagation, and updates the model parameters. Training loss and accuracy are logged using WandB


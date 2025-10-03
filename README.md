# Spam_Detection_For_ML_Models

<h3> <font color = 'red'> Spam_detector using ML Models </fonot> </h3>
<hr>

Spam - whether in the form of emails, messages, etc. - is a nuisance. Thanks to deep learning algorithms, the problem is now well under control. Here I show with a Recurrent Neural Network (RNN) model how fast and uncomplicated a model can be calculated that can distinguish spam from non-spam. The used dataset is the "SMS Spam Collection", which can be found at Kaggle under "https://www.kaggle.com/ishansoni/sms-spam-collection-dataset".


<h3>  Data preprocessing </h3>
 
     1 . Cleaning the text 
     2 . changing the text format to lower cases
     3 . stemming 
     4 . Lemmatization
     5 . stopwords
     
<h3> Vectorization <h3>
   
     1 . one_hot_encoding
     2 . Bag_of_words
     3 . TF_IDF
     4 . Word_Embeddings
         a . word_2_vec
         b . Glove 

 <h3>  Build Data loaders </h3>
 So far, data is processed and prepared as vectorized form.
 Now, it is turn to build data loaders that will feed the batches of datasets into our model.

 To do so,

 A custom data loader class was built
 3 data loaders were instantiated : for train, validation, and test
 
 <h3>  Custom Data Loader </h3>
  Sequence here means a vectorized list of words in an email.
 As we prepared 6,000 e-mails, we have 6,000 sequences.

 As sequences have different lengths, it is required to pass the length of each sequence into our model not to train our model on dummy numbers ( 0s for padding ).

 Consequently, we need custom data loaders that return lengths of each sequence along with sequences and labels.

 Plus, The data loader should sort the batch by each sequence's length and returns the longest one first in the batch to use torch's pack_padded_sequence() (you will see this     function in model code)
 
<hr> 
Using SVM = Train Accuracy_score = 0.9820877084620135  Test Accuracy_score = 0.921865154379332 

<img src="/images/2.png" width="700" height="400">
 
<hr>
 
Using Naive Bayes = Train Accuracy_score = 0.9163063619518221  Test Accuracy_score = 0.8216761184625079 

<img src="/images/3.png" width="700" height="400">
 
<hr>
 
Using Logistic Regression = Train Accuracy_score = 0.9879555281037677  Test Accuracy_score = 0.9464398235664776 

<img src="/images/4.jfif" width="700" height="400">
 
<hr>
 
Using random Forest = Train Accuracy_score = 0.9984558369363805  Test Accuracy_score = 0.926591052299937 
 
<img src="/images/5.png" width="700" height="400">
<hr>
 
Using Decision Tree = Train Accuracy_score = 0.9984558369363805  Test Accuracy_score = 0.9117832388153749 

<img src="/images/6.PNG" width="700" height="400">

 <hr>
 
 <h3> Project URL : https://email-spam-1.herokuapp.com/ </h3>

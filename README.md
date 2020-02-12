# Generative-Topic-Modeling-using-Latent-Dirichlet-Allocation.

This program is to generate topics out of a dateset of news and articles by using Latent Dirichlet Allocation technique.

There are 2 json files which are the mixture of news categories. One is imported to fit the model, and the other one is imported to test the model whether how accurate the model is.

File example:

<img width="500" alt="train and test docs exam" src="https://user-images.githubusercontent.com/45326221/56854811-8563a400-690a-11e9-9720-2b1b90ae3925.PNG">


**Here is how the program works:**
- Take two file name strings as inputs, the file path of text_train.json, and the
file path of text_test.json
- Use LDA to generate 3 topics from created count vectors of train_file.json.
- Test the clustering model performance using test_file:
  + Predict the cluster ID for each document in test_file.
  + Use the first label in the ground-truth label list of each test document as one layer clustering is used in this project.
  + Apply majority vote rule to dynamically map the predicted cluster IDs to the ground-truth labels in test_file.
  + Calculate precision/recall/f-score for each label.
- Return topic distribution and the original ground-truth labels of each document in

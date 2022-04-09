# -*- coding: utf-8 -*-
"""
Emotion Classification

Basic multi-class perceptron model with tf-idf

Group 7, Simone Beckmann, Julia Evans
"""

import csv
import re
import math
from eval_test import evaluation

    
class MCP():
    
    def __init__(self, corpus):
        """
        initiates MCP & creates the sets of words and emotions
        all_scores: dict {emotion : score of emotion} used for predicting
        dict_perceptron: dict of Perceptron objects: {emotion : Perceptron}
        wordset: set of all words from corpus
        training_data: list of lists [[label, sentence]]
        idf_scores: dict of idf score for each feature {word : idf score}
        """
        
        self.all_scores = { }
        self.dict_perceptron = { }
    
        U = Utterance(corpus)
        wordset = U.create_feature_vector()
        emotions = U.labels
        
        self.training_data = U.training_data
        self.idf_scores = U.idf_scores
        
        for em in emotions:
            self.dict_perceptron[em] = Perceptron(em, wordset)          
    
    
    def predict(self, features):
        """
        returns emotion with highest score or alphabetically first emotion
        if two or more emotions have the same high score
        features: one string (sentence)
        all_scores: dict {emotion : score of emotion}
        """
             
        for emotion, perceptron in self.dict_perceptron.items():
            self.all_scores[emotion] = perceptron.get_scores(features)
        
        prediction = [emotion for emotion, score in self.all_scores.items() 
                        if score == max(self.all_scores.values())]
        
        return min(prediction)
    
    
    def train(self):
        """
        adjusts the weights of features based on the predictions
        training_data: list of lists [[label, sentence]]
        lst_label_sentence: [label, sentence]
        """
        
        for lst_label_sentence in self.training_data:
            
            sentence = lst_label_sentence[1]
            pred_label = self.predict(sentence)
            training_label = lst_label_sentence[0]  
                        
            if training_label != pred_label:
                
                for word in sentence.split():
                    idf = (1 + math.log(self.idf_scores.get(word, 1)))
                    tf = math.log(1 + sentence.split().count(word))
                    score = tf * idf
                    
                    self.dict_perceptron[training_label].dict_weights[word] += score
                    self.dict_perceptron[pred_label].dict_weights[word] -= score
                   
                

class Perceptron():
    
    def __init__(self, emotion, wordset):
        """
        initializes the feature weights dictionary
        wordset: set of feature types from corpus
        dict_weights: {feature : weight}
        """

        self.emotion = emotion
        self.wordset = wordset
        self.dict_weights = { }
        
        for word in self.wordset:
            self.dict_weights[word] = 0

                      
    def get_scores(self, features):
        """
        returns the sum of the weights for all words in features
        features: one string (sentence)
        """
        
        list_features = features.split()
        weight_sum = 0
        
        for feature in list_features:
            feature = re.sub(r"(\.|\?|\!|;|:|,|\(|\)|'s|\")","",feature.lower())
            weight_sum += self.dict_weights.get(feature, 0)

        return weight_sum
        
    
class Utterance():
    
    def __init__(self,filename):
        self.filename = filename
        self.feat_set = [ ]
        self.labels = [ ]
        self.training_data = [ ]
        self.idf_scores = { }


    def create_feature_vector(self):    
        """
        removes punctuation and applies casefolding to features
        extracts emotion labels
        calculates idf score for each word
        returns a set of feature types
        """
        
        documents = 0
        doc_word_count = { }

        with open(self.filename) as csv_file:
            data_reader = csv.reader(csv_file, delimiter=',')
            data = [ ]
            
            for row in data_reader:
                if len(row) == 2 and not re.match(r"\W.*", row[0]):
                    documents += 1

                    data.append(row[1].split())
                    self.labels.append(row[0])
                    
                    sent = re.sub(r"\. ", " ", row[1])
                    sent = re.sub(r"(\.|\?|\!|;|:|,|\(|\)|'s|\")","",sent.lower())
                    self.training_data.append([row[0],re.sub(r"(\.|\?|\!|;|:|,|\(|\)|'s|\")","",sent.lower())])
                    
                    for word in list(set(sent.split())):
                        doc_word_count.setdefault(word, 0)
                        doc_word_count[word] += 1
                        
        for word in doc_word_count:
            self.idf_scores[word] = documents / (doc_word_count[word] + 1)
                
        self.labels = set(self.labels)
        
        self.feat_set = [re.sub(r"(\.|\?|\!|;|:|,|\(|\)|'s|\")", "" , word.lower()) 
                    for sent in data for word in sent]
        
        return set(self.feat_set)
    
    
# initialize MCP with training data  
MCP_Basic = MCP("isear-train.csv") 

# 20 iterations of training step
for i in range(20):
    print("Iteration: ", str(i))
    MCP_Basic.train()


#### EVALUATION ####
def get_predictions(gold_file):
    """
    reads file, gets predictions for features
    returns list of lists [pred_label, gold_label] for evaluation
    """
    
    with open(gold_file) as csv_file:
        gold_csv_reader = csv.reader(csv_file, delimiter=',')
        eval_list = [ ]
        
        for row in gold_csv_reader:
            if len(row) == 2 and not re.match(r"\W.*", row[0]):
                pred_label = MCP_Basic.predict(row[1])
                eval_list.append([pred_label, row[0]])
                
    return eval_list


our_eval_data = get_predictions("isear-test.csv")
print(evaluation(our_eval_data))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:39:50 2021

@author: divyakapur
"""

#!pip install transformers
#!pip install torch
#!pip install sentence-transformers

import os
import torch
import tensorflow as tf
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

# To disable all logging output from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# Disabling parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# For response generation, I used the DialoGPT model
model_name = "microsoft/DialoGPT-large"
# For sentence transformation, I used the stsb roberta-large model
sentence_transformer_model_name = "stsb-roberta-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
sentence_transformer_model = SentenceTransformer(sentence_transformer_model_name)

# Initialize chat history 
chat_history = None
chat_history_strings = []


for chat_round in range(1000):
    input_chat = input(">> You: ")
    # 'Bye' keyword breaks out of the loop
    if input_chat.lower() == "goodbye":
        print("Bot: Goodbye.👋")
        break
    new_chat_input = tokenizer.encode(input_chat + tokenizer.eos_token, return_tensors='pt')
    if chat_round > 0:
        # If the user asks the same question more than once
        if (chat_history_strings.count(input_chat.lower()) > 0):
            print("Bot: You already asked me that. Are you glitching?")
            continue
        corpus_embeddings = sentence_transformer_model.encode(chat_history_strings, convert_to_tensor=True)
        question_embedding = sentence_transformer_model.encode(input_chat, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(question_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.numpy()
        # If the similarity score between the last question and the new question is 
        # below the threshold
        if cos_scores[-1] < 0.25:
            print("Bot: Hmm... I'm not sure about that. Can you tell me more about this topic?")
            bot_inputs = torch.cat([chat_history, new_chat_input], dim=-1)
            continue
        bot_inputs = torch.cat([chat_history, new_chat_input], dim=-1)
    else:
        bot_inputs = new_chat_input
    chat_history_strings.append(input_chat.lower())
    chat_history = model.generate(bot_inputs, max_length=1250, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(chat_history[:, bot_inputs.shape[-1]:][0], skip_special_tokens=True)
    print("Bot: {}".format(bot_response))
    
      

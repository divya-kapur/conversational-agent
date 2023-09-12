# conversational-agent
This project presents a Conversational Agent that utilizes state-of-the-art NLP techniques to interact intelligently with users

**Project Overview:** Leveraging cutting-edge response generation and sentence transformer models, this project aims to build a conversational agent that responds dynamically to user input, answering a wide range of questions, including multi-turn dialogues, common sense inquiries, and philosophical discussions. It is designed to adapt its responses based on context and can request additional information when needed. The conversational agent can be interacted with through a user interface. Users can input questions, engage in conversations, and receive dynamic responses. The agent's contextual awareness and adaptability enhance the quality of interactions.

**Models**

To build an intelligent conversational agent, I used the following NLP models:

- *DialoGPT*: The core model for generating responses is Microsoft's DialoGPT, a large-scale pretrained dialogue generation model. It is trained on a massive dataset comprising 147 million conversations from Reddit discussion threads and is an autoregressive model that uses a multi-layer transformer architecture. This architecture enables the model to focus on information representation at various levels, enhancing contextual understanding.
- *STSB Roberta-Large Sentence Transformer*: To facilitate context-awareness and question relevance assessment, the STSB Roberta-Large Sentence Transformer model is used. It encodes the chat history (corpus) and the new user question. By converting embeddings to cosine similarity scores, the model assesses how similar the new question is to the chat history. This approach allows the agent to remember past interactions and detect out-of-context questions. The STSB Roberta-Large model is trained on extensive datasets, including SNLI+MultiNLI and the STS benchmark dataset, which encompass diverse sentence pairs and semantic evaluations.

**Features**

- *Interactive Conversations*: The conversational agent can engage in up to 1000 rounds of dialogue. Users can conclude the conversation at any point by entering the keyword 'goodbye.'
- *Contextual Understanding*: Each user input is analyzed to determine if it has been previously asked. If it has, the agent responds accordingly. The input is then encoded using the STSB Roberta-Large model, generating a similarity score to assess relevance to the ongoing conversation.
- *Adaptive Responses*: If a user input appears to be out of context, the interface prompts the user to provide additional information. This ensures that the agent can respond appropriately by understanding the context of the query.
- *Multi-turn Dialogues*: The project supports multi-turn conversations, enabling users to engage in meaningful dialogues with the agent.

**References**

- [DialoGPT: Large-Scale Generative Pretraining for Conversational Response Generation](https://arxiv.org/pdf/1911.00536.pdf)
- [STSB Roberta-Large Sentence Embeddings](https://huggingface.co/sentence-transformers/stsb-roberta-large)


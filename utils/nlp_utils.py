import json
import re
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer,util
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer,DistilBertTokenizer, BertForQuestionAnswering, BertTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import logging

# MongoDB setup
mongo_client = MongoClient('mongodb://localhost:27017/')
mongo_db = mongo_client['your_database_name']
inventory_collection = mongo_db['inventory']


# Load the BART model and tokenizer
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


qa_model = pipeline('question-answering', model='deepset/roberta-base-squad2')

# Load models
sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
flan_t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# NLP setup
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')

sentence_transformer = SentenceTransformer('all-MiniLM-L12-v2')

# Constants
TOP_K = 5
MAX_LENGTH = 200
NUM_BEAMS = 4
LENGTH_PENALTY = 2.0

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = [word for word in text.split() if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def product_to_text(product):
    return " | ".join([f"{key}: {value}" for key, value in product.items() if key != 'id'])

def get_formatted_inventory():
    try:
        return list(inventory_collection.find({}, {'_id': 0}))
    except Exception as e:
        print(f"Error fetching inventory data: {e}")
        return None



# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")

import json
import nltk
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load models (do this once at the start of your application)
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
flan_t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def preprocess_text(text):
    # Implement more robust text preprocessing
    text = text.lower()
    text = ' '.join(nltk.word_tokenize(text))
    return text

def get_general_answer(data, question):
    try:
        chatbot_data = json.loads(data) if isinstance(data, str) else data
        pdf_data = chatbot_data.get('pdf_data', [])
        folder_data = chatbot_data.get('folder_data', [])
        web_data = chatbot_data.get('web_data', {})
    except json.JSONDecodeError:
        return "I'm sorry, but there seems to be an issue with our data. Please try again later."

    # Preprocess and structure the data
    structured_data = preprocess_data(pdf_data, folder_data, web_data)

    # Find the most relevant information using a hybrid approach
    relevant_info = find_relevant_info_hybrid(question, structured_data)

    # Generate the answer
    answer = generate_answer(question, relevant_info)

    # Post-process the answer
    answer = post_process_answer(answer, question)

    return answer





# def get_relevant_items(question, inventory_data, top_k=5):
#     inventory_texts = [
#         f"{item['name']} {item['category']} ${item['price']} {item['quantity']} in stock"
#         for item in inventory_data
#     ]
#     preprocessed_inventory = [preprocess_text(text) for text in inventory_texts]

#     question_embedding = sentence_transformer.encode(preprocess_text(question), convert_to_tensor=True)
#     inventory_embeddings = sentence_transformer.encode(preprocessed_inventory, convert_to_tensor=True)

#     similarities = util.pytorch_cos_sim(question_embedding, inventory_embeddings)[0]
#     top_k_indices = similarities.argsort(descending=True)[:top_k]

#     return [inventory_data[i] for i in top_k_indices]


def get_relevant_items(question, inventory_data, top_k=5):
    try:
        inventory_texts = [
            f"{item['name']} {item['category']} ${item['price']} {item['quantity']} in stock"
            for item in inventory_data
        ]
        preprocessed_inventory = [preprocess_text(text) for text in inventory_texts]

        question_embedding = sentence_transformer.encode(preprocess_text(question), convert_to_tensor=True)
        inventory_embeddings = sentence_transformer.encode(preprocessed_inventory, convert_to_tensor=True)

        logging.debug(f"Question embedding shape: {question_embedding.shape}")
        logging.debug(f"Inventory embeddings shape: {inventory_embeddings.shape}")

        if question_embedding.shape[0] == 0 or inventory_embeddings.shape[0] == 0:
            raise ValueError("Empty embedding detected")

        similarities = util.pytorch_cos_sim(question_embedding, inventory_embeddings)[0]
        top_k_indices = similarities.argsort(descending=True)[:top_k]

        return [inventory_data[i] for i in top_k_indices]
    except Exception as e:
        logging.error(f"Error in get_relevant_items: {str(e)}")
        return []


def preprocess_data(pdf_data, folder_data, web_data):
    structured_data = []

    logging.debug(f"Preprocessing PDF data: {len(pdf_data)} items")
    logging.debug(f"Preprocessing folder data: {len(folder_data)} items")
    logging.debug(f"Preprocessing web data: {bool(web_data)}")

    # Process PDF and folder data
    for item in pdf_data + folder_data:
        if isinstance(item, dict) and 'text' in item:
            sentences = nltk.sent_tokenize(item['text'])
            structured_data.extend([{'type': 'text', 'content': sent, 'source': 'pdf/folder'} for sent in sentences])

    # Process web data
    if web_data:
        logging.debug(f"Web data keys: {web_data.keys()}")
        if isinstance(web_data, list) and len(web_data) > 0:
            web_data = web_data[0]  # Take the first item if it's a list
        
        if isinstance(web_data, dict):
            if 'title' in web_data:
                structured_data.append({'type': 'title', 'content': web_data['title'], 'source': 'web'})
            
            if 'sections' in web_data:
                for section in web_data['sections']:
                    if isinstance(section, dict):
                        if 'heading' in section:
                            structured_data.append({'type': 'heading', 'content': section['heading'], 'source': 'web'})
                        if 'content' in section:
                            structured_data.extend([{'type': 'web_content', 'content': item, 'source': 'web'} for item in section['content']])
            
            if 'sub_pages' in web_data:
                for sub_page in web_data['sub_pages']:
                    structured_data.extend(preprocess_data([], [], [sub_page]))  # Recursive call for sub-pages

    logging.info(f"Preprocessed data: {len(structured_data)} items")
    logging.debug(f"First few preprocessed items: {structured_data[:5]}")
    return structured_data

def find_relevant_info_hybrid(question, structured_data):
    # Semantic search using sentence transformers
    question_embedding = sentence_transformer.encode(question, convert_to_tensor=True)
    content_list = [item['content'] for item in structured_data]
    content_embeddings = sentence_transformer.encode(content_list, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(question_embedding, content_embeddings)[0]

    # Keyword-based search using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([question] + content_list)
    keyword_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    # Combine scores (you can adjust the weights)
    combined_scores = 0.7 * semantic_scores.numpy() + 0.3 * keyword_scores

    # Get top results
    top_k = min(7, len(combined_scores))
    best_match_indices = combined_scores.argsort()[-top_k:][::-1]

    return [structured_data[i] for i in best_match_indices]

# def find_relevant_info_hybrid(question, structured_data):
#     try:
#         content_list = [item['content'] for item in structured_data]
#         if not content_list:
#             logging.warning("No content to process in structured_data")
#             return []

#         # Semantic search using sentence transformers
#         question_embedding = sentence_transformer.encode(question, convert_to_tensor=True)
#         content_embeddings = sentence_transformer.encode(content_list, convert_to_tensor=True)

#         logging.debug(f"Question embedding shape: {question_embedding.shape}")
#         logging.debug(f"Content embeddings shape: {content_embeddings.shape}")

#         semantic_scores = util.pytorch_cos_sim(question_embedding, content_embeddings)[0]

#         # Keyword-based search using TF-IDF
#         vectorizer = TfidfVectorizer()
#         tfidf_matrix = vectorizer.fit_transform([question] + content_list)
#         keyword_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

#         logging.debug(f"Semantic scores shape: {semantic_scores.shape}")
#         logging.debug(f"Keyword scores shape: {keyword_scores.shape}")

#         if semantic_scores.shape != keyword_scores.shape:
#             raise ValueError(f"Shape mismatch: semantic_scores {semantic_scores.shape}, keyword_scores {keyword_scores.shape}")

#         # Combine scores (you can adjust the weights)
#         combined_scores = 0.7 * semantic_scores.numpy() + 0.3 * keyword_scores

#         # Get top results
#         top_k = min(7, len(combined_scores))
#         best_match_indices = combined_scores.argsort()[-top_k:][::-1]

#         return [structured_data[i] for i in best_match_indices]
#     except Exception as e:
#         logging.error(f"Error in find_relevant_info_hybrid: {str(e)}")
#         return []

def generate_answer(question, relevant_info):
    context = "\n".join([f"{item['type']} ({item['source']}): {item['content']}" for item in relevant_info])
    
    input_text = f"""
    Question: {question}
    
    Context:
    {context}
    
    Task: Answer the question based on the given context. If the information is not available or if you're not sure, say so. Provide a detailed and accurate response.
    """

    input_ids = flan_t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
    
    with torch.no_grad():
        outputs = flan_t5_model.generate(
            input_ids,
            max_length=200,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_beams=4,
            early_stopping=True
        )

    answer = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def post_process_answer(answer, question):
    # Ensure the answer directly addresses the question
    if not any(keyword in answer.lower() for keyword in question.lower().split()):
        answer = f"To answer your question about {question.lower()}: {answer}"

    # Add confidence statement if the answer seems uncertain
    uncertainty_keywords = ['might', 'maybe', 'possibly', 'not sure', 'could be']
    if any(keyword in answer.lower() for keyword in uncertainty_keywords):
        answer += " Please note that this answer is based on the available information and may not be completely certain."

    return answer

# need to improve on this part(high priority)
def split_complex_query(query):
    sentences = sent_tokenize(query)
    
    # Identify question sentences
    questions = [s for s in sentences if s.strip().endswith('?')]
    
    # If no questions are found, treat the entire query as one question
    if not questions:
        questions = [query]
    
    return questions





def get_inventory_rag_answer(data, query):
    try:
        chatbot_data = json.loads(data) if isinstance(data, str) else data
        inventory_data = chatbot_data['db_data'][0]['text']
        inventory_data = json.loads(inventory_data) if isinstance(inventory_data, str) else inventory_data
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        logging.error(f"Error parsing inventory data: {e}")
        return "I apologize, but there seems to be an issue with our inventory data. Please try again later."

    questions = split_complex_query(query)
    
    final_answer = ""
    for question in questions:
        relevant_items = get_relevant_items(question, inventory_data)
        
        inventory_summary = "\n".join([
            f"{item['name']}: ${item['price']:.2f}, {item['quantity']} in stock, Category: {item['category']}"
            for item in relevant_items
        ])

        answer = generate_answer(question, inventory_summary)
        answer = post_process_answer(answer)
        final_answer += answer + " "

    return final_answer.strip()


# def get_inventory_rag_answer(data, query):
#     try:
#         chatbot_data = json.loads(data) if isinstance(data, str) else data
#         inventory_data = chatbot_data['db_data'][0]['text']
#         inventory_data = json.loads(inventory_data) if isinstance(inventory_data, str) else inventory_data

#         if not inventory_data:
#             raise ValueError("Empty inventory data")

#         logging.debug(f"Inventory data size: {len(inventory_data)}")
#         logging.debug(f"Query: {query}")

#         questions = split_complex_query(query)
        
#         final_answer = ""
#         for question in questions:
#             relevant_items = get_relevant_items(question, inventory_data)
            
#             if not relevant_items:
#                 final_answer += "I couldn't find any relevant inventory items for this question. "
#                 continue

#             inventory_summary = "\n".join([
#                 f"{item['name']}: ${item['price']:.2f}, {item['quantity']} in stock, Category: {item['category']}"
#                 for item in relevant_items
#             ])

#             answer = generate_answer(question, inventory_summary)
#             answer = post_process_answer(answer)
#             final_answer += answer + " "

#         return final_answer.strip() if final_answer else "I'm sorry, but I couldn't generate a relevant answer based on the available inventory data."

#     except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
#         logging.error(f"Error parsing inventory data: {e}")
#         return "I apologize, but there seems to be an issue with our inventory data. Please try again later."
#     except Exception as e:
#         logging.error(f"Unexpected error in get_inventory_rag_answer: {str(e)}")
#         return "I encountered an unexpected error while processing your question. Please try again later."
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



# def get_general_answer(data, question):
#     try:
#         chatbot_data = json.loads(data)
#         pdf_data = chatbot_data.get('pdf_data', [])
#         folder_data = chatbot_data.get('folder_data', [])
#         web_data = chatbot_data.get('web_data', [])  # Add this line to get web data
#     except json.JSONDecodeError:
#         logging.error("Failed to decode chatbot data")
#         return "I'm sorry, but there seems to be an issue with our data. Please try again later."

#     chunks = []
#     for item in pdf_data:
#         if 'text' in item:
#             text_chunks = nltk.sent_tokenize(item['text'])
#             chunks.extend(text_chunks)
    
#     for item in folder_data:
#         if 'text' in item:
#             text_chunks = nltk.sent_tokenize(item['text'])
#             chunks.extend(text_chunks)
    
#     for item in web_data:  # Add this loop to process web data
#         if 'text' in item:
#             text_chunks = nltk.sent_tokenize(item['text'])
#             chunks.extend(text_chunks)

#     if not chunks:
#         return "I'm sorry, but I don't have enough information to answer your question."

#     preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
#     question_embedding = sentence_transformer_model.encode(preprocess_text(question), convert_to_tensor=True)
#     chunk_embeddings = sentence_transformer_model.encode(preprocessed_chunks, convert_to_tensor=True)

#     similarity_scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
#     top_k = min(TOP_K, len(similarity_scores))
#     best_match_indices = similarity_scores.topk(top_k).indices.tolist()
#     context = " ".join([chunks[i] for i in best_match_indices])

#     qa_input = {'question': question, 'context': context}
#     result = qa_model(qa_input)

#     input_text = f"question: {question} context: {context}"
#     input_ids = bart_tokenizer.encode(input_text, return_tensors='pt')
#     summary_ids = bart_model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
#     complete_answer = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     return complete_answer

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")

# --------------------------------------------------------
#  second iteration of get general answer function
# ---------------------------------------------------------
# def get_general_answer(data, question):
#     try:
#         chatbot_data = json.loads(data) if isinstance(data, str) else data
#         pdf_data = chatbot_data.get('pdf_data', [])
#         folder_data = chatbot_data.get('folder_data', [])
#         web_data = chatbot_data.get('web_data', [])
#     except json.JSONDecodeError:
#         logging.error("Failed to decode chatbot data")
#         return "I'm sorry, but there seems to be an issue with our data. Please try again later."

#     chunks = []
    
#     # Process PDF data
#     for item in pdf_data:
#         if 'text' in item:
#             text_chunks = nltk.sent_tokenize(item['text'])
#             chunks.extend(text_chunks)
    
#     # Process folder data
#     for item in folder_data:
#         if 'text' in item:
#             text_chunks = nltk.sent_tokenize(item['text'])
#             chunks.extend(text_chunks)

#      # Process web data
#     for page in web_data:
#         chunks.append(page['title'])
#         for section in page['sections']:
#             chunks.append(section['heading'])
#             chunks.extend(section['content'])


#     preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
#     question_embedding = sentence_transformer_model.encode(preprocess_text(question), convert_to_tensor=True)
#     chunk_embeddings = sentence_transformer_model.encode(preprocessed_chunks, convert_to_tensor=True)

#     similarity_scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
#     top_k = min(TOP_K, len(similarity_scores))
#     best_match_indices = similarity_scores.topk(top_k).indices.tolist()
#     context = " ".join([chunks[i] for i in best_match_indices])

#     qa_input = {'question': question, 'context': context}
#     result = qa_model(qa_input)

#     input_text = f"question: {question} context: {context}"
#     input_ids = bart_tokenizer.encode(input_text, return_tensors='pt')
#     summary_ids = bart_model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
#     complete_answer = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     return complete_answer

# def get_general_answer(data, question):
#     try:
#         chatbot_data = json.loads(data) if isinstance(data, str) else data
#         pdf_data = chatbot_data.get('pdf_data', [])
#         folder_data = chatbot_data.get('folder_data', [])
#         web_data = chatbot_data.get('web_data', {})  # Change this to {} as it's a dictionary in your data structure
#     except json.JSONDecodeError:
#         logging.error("Failed to decode chatbot data")
#         return "I'm sorry, but there seems to be an issue with our data. Please try again later."

#     chunks = []
    
#     # Process PDF data
#     for item in pdf_data:
#         if isinstance(item, dict) and 'text' in item:
#             text_chunks = nltk.sent_tokenize(item['text'])
#             chunks.extend(text_chunks)
    
#     # Process folder data
#     for item in folder_data:
#         if isinstance(item, dict) and 'text' in item:
#             text_chunks = nltk.sent_tokenize(item['text'])
#             chunks.extend(text_chunks)

#     # Process web data
#     if isinstance(web_data, dict):
#         chunks.append(web_data.get('title', ''))
#         for section in web_data.get('sections', []):
#             chunks.append(section.get('heading', ''))
#             chunks.extend(section.get('content', []))

#     if not chunks:
#         return "I'm sorry, but I don't have enough information to answer your question."

#     preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks if isinstance(chunk, str)]
#     question_embedding = sentence_transformer_model.encode(preprocess_text(question), convert_to_tensor=True)
#     chunk_embeddings = sentence_transformer_model.encode(preprocessed_chunks, convert_to_tensor=True)

#     similarity_scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
#     top_k = min(TOP_K, len(similarity_scores))
#     best_match_indices = similarity_scores.topk(top_k).indices.tolist()
#     context = " ".join([chunks[i] for i in best_match_indices if i < len(chunks)])

#     qa_input = {'question': question, 'context': context}
#     result = qa_model(qa_input)

#     input_text = f"question: {question} context: {context}"
#     input_ids = bart_tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True)
#     summary_ids = bart_model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
#     complete_answer = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     return complete_answer


import json
import logging
import nltk
from sentence_transformers import SentenceTransformer, util

def get_general_answer(data, question):
    try:
        chatbot_data = json.loads(data) if isinstance(data, str) else data
        pdf_data = chatbot_data.get('pdf_data', [])
        folder_data = chatbot_data.get('folder_data', [])
        web_data = chatbot_data.get('web_data', {})
    except json.JSONDecodeError:
        logging.error("Failed to decode chatbot data")
        return "I'm sorry, but there seems to be an issue with our data. Please try again later."

    # Preprocess and structure the data
    structured_data = preprocess_data(pdf_data, folder_data, web_data)

    # Find the most relevant information
    relevant_info = find_relevant_info(question, structured_data)

    # Generate the answer
    answer = generate_answer(question, relevant_info)

    return answer

# def preprocess_data(pdf_data, folder_data, web_data):
#     structured_data = []

#     # Process PDF and folder data
#     for item in pdf_data + folder_data:
#         if isinstance(item, dict) and 'text' in item:
#             sentences = nltk.sent_tokenize(item['text'])
#             structured_data.extend([{'type': 'text', 'content': sent} for sent in sentences])

#     # Process web data
#     if isinstance(web_data, dict):
#         structured_data.append({'type': 'title', 'content': web_data.get('title', '')})
#         for section in web_data.get('sections', []):
#             heading = section.get('heading', '')
#             content = section.get('content', [])
#             structured_data.append({'type': 'heading', 'content': heading})
#             structured_data.extend([{'type': 'web_content', 'content': item} for item in content])

#     return structured_data


def preprocess_data(pdf_data, folder_data, web_data):
    structured_data = []

    logging.info(f"Preprocessing - PDF data items: {len(pdf_data)}")
    logging.info(f"Preprocessing - Folder data items: {len(folder_data)}")
    logging.info(f"Preprocessing - Web data present: {bool(web_data)}")

    # Process PDF and folder data
    for item in pdf_data + folder_data:
        if isinstance(item, dict) and 'text' in item:
            sentences = nltk.sent_tokenize(item['text'])
            structured_data.extend([{'type': 'text', 'content': sent, 'source': 'pdf/folder'} for sent in sentences])

    # Process web data
    if isinstance(web_data, dict):
        structured_data.append({'type': 'title', 'content': web_data.get('title', ''), 'source': 'web'})
        for section in web_data.get('sections', []):
            structured_data.append({'type': 'heading', 'content': section.get('heading', ''), 'source': 'web'})
            structured_data.extend([{'type': 'web_content', 'content': item, 'source': 'web'} for item in section.get('content', [])])

    logging.info(f"Preprocessing complete - Structured data items: {len(structured_data)}")

    return structured_data


def find_relevant_info(question, structured_data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embedding = model.encode(question, convert_to_tensor=True)
    
    content_list = [item['content'] for item in structured_data]
    content_embeddings = model.encode(content_list, convert_to_tensor=True)
    
    similarity_scores = util.pytorch_cos_sim(question_embedding, content_embeddings)[0]
    top_k = min(5, len(similarity_scores))
    best_match_indices = similarity_scores.topk(top_k).indices.tolist()
    
    return [structured_data[i] for i in best_match_indices]

def generate_answer(question, relevant_info):
    context = " ".join([f"{item['type']}: {item['content']}" for item in relevant_info])
    
    # Here you would typically use a more advanced language model like GPT-3 or a fine-tuned BART model
    # For this example, we'll use a simple template-based approach
    if any(item['type'] == 'web_content' for item in relevant_info):
        for item in relevant_info:
            if item['type'] == 'heading':
                service = item['content']
            elif item['type'] == 'web_content' and item['content'].startswith('$'):
                return f"The price for {service} is {item['content']}."
    
    return f"Based on the available information: {context}"

# You would still need to implement or import qa_model, bart_model, and bart_tokenizer for more advanced answer generation

# def get_general_answer(data, question):
#     try:
#         chatbot_data = json.loads(data) if isinstance(data, str) else data
#         web_data = chatbot_data.get('web_data', [])
#         pdf_data = chatbot_data.get('pdf_data', [])
#     except json.JSONDecodeError:
#         logging.error("Failed to decode chatbot data")
#         return "I'm sorry, but there seems to be an issue with our data. Please try again later."

#     # Prepare the web data
#     sentences = []
#     for item in web_data:
#         if item['tag'] in ['h1', 'h2', 'h3']:
#             sentences.append(item['text'])
#         elif item['tag'] in ['p', 'li']:
#             sentences.extend(item['text'].split('. '))

#     # Prepare the PDF data
#     for item in pdf_data:
#         if 'text' in item:
#             sentences.extend(item['text'].split('. '))

#     if not sentences:
#         return "I'm sorry, but I don't have enough information to answer your question."

#     # Encode the question and sentences
#     question_embedding = model.encode(question, convert_to_tensor=True)
#     sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

#     # Compute cosine similarities
#     cosine_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]

#     # Get the top 3 most similar sentences
#     top_results = torch.topk(cosine_scores, k=min(3, len(sentences)))
    
#     # Prepare the context from top similar sentences
#     context = " ".join([sentences[idx] for idx in top_results.indices])

#     # Use the question answering pipeline
#     result = qa_pipeline(question=question, context=context)

#     return result['answer']


# need to improve on this part(high priority)
def split_complex_query(query):
    sentences = sent_tokenize(query)
    
    # Identify question sentences
    questions = [s for s in sentences if s.strip().endswith('?')]
    
    # If no questions are found, treat the entire query as one question
    if not questions:
        questions = [query]
    
    return questions


def get_relevant_items(question, inventory_data, top_k=5):
    inventory_texts = [
        f"{item['name']} {item['category']} ${item['price']} {item['quantity']} in stock"
        for item in inventory_data
    ]
    preprocessed_inventory = [preprocess_text(text) for text in inventory_texts]

    question_embedding = sentence_transformer.encode(preprocess_text(question), convert_to_tensor=True)
    inventory_embeddings = sentence_transformer.encode(preprocessed_inventory, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(question_embedding, inventory_embeddings)[0]
    top_k_indices = similarities.argsort(descending=True)[:top_k]

    return [inventory_data[i] for i in top_k_indices]

def generate_answer(question, context):
    input_text = f"""Task: Answer the following question about our inventory. Use the provided information to give a detailed and accurate response.

    Question: {question}

    Relevant Inventory Information:
    {context}

    Instructions:
    1. Directly address the question asked.
    2. Provide specific details from the inventory when relevant.
    3. If asked about multiple items or categories, include information on all relevant items.
    4. If the question cannot be answered with the given information, say so politely.
    5. Use complete sentences and proper grammar.
    6. Keep the answer concise but informative.
    """

    input_ids = flan_t5_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).input_ids
    
    with torch.no_grad():
        outputs = flan_t5_model.generate(
            input_ids,
            max_length=300,
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

def post_process_answer(answer):
    answer = re.sub(r'\s+', ' ', answer).strip()
    answer = answer[0].upper() + answer[1:]  # Capitalize first letter
    if not answer.endswith(('.', '!', '?')):
        answer += '.'
    return answer

# def get_inventory_rag_answer(data, query):
#     try:
#         inventory_data = json.loads(data)['db_data'][0]['text']
#         inventory_data = json.loads(inventory_data)
#     except (json.JSONDecodeError, KeyError, IndexError):
#         return "I apologize, but there seems to be an issue with our inventory data. Please try again later."

#     # Split the complex query into individual questions
#     questions = split_complex_query(query)
    
#     final_answer = ""
#     for question in questions:
#         relevant_items = get_relevant_items(question, inventory_data)
        
#         inventory_summary = "\n".join([
#             f"{item['name']}: ${item['price']}, {item['quantity']} in stock, Category: {item['category']}"
#             for item in relevant_items
#         ])

#         answer = generate_answer(question, inventory_summary)
#         answer = post_process_answer(answer)
#         final_answer += answer + " "

#     return final_answer.strip()





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
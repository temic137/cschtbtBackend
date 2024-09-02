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



def get_general_answer(data, question):
    try:
        chatbot_data = json.loads(data)
        pdf_data = chatbot_data.get('pdf_data', [])
    except json.JSONDecodeError:
        logging.error("Failed to decode chatbot data")
        return "I'm sorry, but there seems to be an issue with our data. Please try again later."

    chunks = []
    for item in pdf_data:
        if 'text' in item:
            text_chunks = nltk.sent_tokenize(item['text'])
            chunks.extend(text_chunks)

    if not chunks:
        return "I'm sorry, but I don't have enough information to answer your question."

    preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
    question_embedding = sentence_transformer_model.encode(preprocess_text(question), convert_to_tensor=True)
    chunk_embeddings = sentence_transformer_model.encode(preprocessed_chunks, convert_to_tensor=True)

    similarity_scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
    top_k = min(TOP_K, len(similarity_scores))
    best_match_indices = similarity_scores.topk(top_k).indices.tolist()
    context = " ".join([chunks[i] for i in best_match_indices])

    qa_input = {'question': question, 'context': context}
    result = qa_model(qa_input)

    input_text = f"question: {question} context: {context}"
    input_ids = bart_tokenizer.encode(input_text, return_tensors='pt')
    summary_ids = bart_model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    complete_answer = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return complete_answer

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

def get_inventory_rag_answer(data, query):
    try:
        inventory_data = json.loads(data)['db_data'][0]['text']
        inventory_data = json.loads(inventory_data)
    except (json.JSONDecodeError, KeyError, IndexError):
        return "I apologize, but there seems to be an issue with our inventory data. Please try again later."

    # Split the complex query into individual questions
    questions = split_complex_query(query)
    
    final_answer = ""
    for question in questions:
        relevant_items = get_relevant_items(question, inventory_data)
        
        inventory_summary = "\n".join([
            f"{item['name']}: ${item['price']}, {item['quantity']} in stock, Category: {item['category']}"
            for item in relevant_items
        ])

        answer = generate_answer(question, inventory_summary)
        answer = post_process_answer(answer)
        final_answer += answer + " "

    return final_answer.strip()





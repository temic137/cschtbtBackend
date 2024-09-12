from flask import Blueprint, request, jsonify, session, current_app,logging
from models import Chatbot
from werkzeug.utils import secure_filename
from extensions import db
from utils.nlp_utils import preprocess_text, get_general_answer, get_inventory_rag_answer, get_formatted_inventory
from utils.file_utils import extract_text_from_pdf, read_text_file,extract_folder_content,extract_text_from_url
from utils.api_utils import fetch_real_time_data
import json
import uuid
import os
from functools import wraps
from transformers import pipeline
import logging

logging.basicConfig(level=logging.ERROR)
chatbot_bp = Blueprint('chatbot', __name__)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            current_app.logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({"error": "An unexpected error occurred"}), 500
    return decorated_function

@chatbot_bp.route('/create_chatbot', methods=['POST'])
@login_required
@handle_errors
def create_chatbot():
    data = request.json
    name = data.get('name')

    new_chatbot = Chatbot(
        id=str(uuid.uuid4()),
        name=name,
        user_id=session['user_id'],
        data=json.dumps([])
    )
    db.session.add(new_chatbot)
    db.session.commit()

    return jsonify({"message": "Chatbot created successfully", "chatbot_id": new_chatbot.id}), 201

@chatbot_bp.route('/chatbots', methods=['GET'])
@login_required
@handle_errors
def get_chatbots():
    user_id = session['user_id']
    chatbots = Chatbot.query.filter_by(user_id=user_id).all()
    chatbot_list = [{"id": c.id, "name": c.name} for c in chatbots]

    return jsonify(chatbot_list), 200



@chatbot_bp.route('/train_chatbot/<chatbot_id>', methods=['POST'])
@login_required
@handle_errors
def train_chatbot(chatbot_id):
    try:
        chatbot = Chatbot.query.get(chatbot_id)
        if not chatbot or chatbot.user_id != session['user_id']:
            return jsonify({"error": "Chatbot not found or unauthorized"}), 404
        
        file = request.files.get('file')
        api_url = request.form.get('api_url')
        folder_path = request.form.get('folder_path')
        website_url = request.form.get('website_url')
        
        pdf_data = []
        db_data = []
        folder_data = []
        web_data = []
        
        if file:
            filename = secure_filename(file.filename)
            upload_folder = current_app.config['UPLOAD_FOLDER']
            
            os.makedirs(upload_folder, exist_ok=True)
            
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            file_extension = os.path.splitext(filename)[1].lower()
            
            try:
                if file_extension == '.pdf':
                    pdf_text = extract_text_from_pdf(filepath)
                    pdf_data.extend(pdf_text)
                elif file_extension in ['.txt', '.md', '.rst']:
                    raw_text = read_text_file(filepath)
                    pdf_data.append({'page': 'file', 'text': raw_text})
                else:
                    return jsonify({"error": f"Unsupported file type: {file_extension}"}), 400
            except Exception as e:
                current_app.logger.error(f"Error processing file {filename}: {str(e)}")
                return jsonify({"error": f"Error processing file {filename}: {str(e)}"}), 500
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        if api_url:
            try:
                real_time_data = fetch_real_time_data(api_url)
                if real_time_data:
                    real_time_text = json.dumps(real_time_data, indent=2)
                    db_data.append({'page': 'real_time', 'text': real_time_text})
            except Exception as e:
                current_app.logger.error(f"Error fetching data from API {api_url}: {str(e)}")
                return jsonify({"error": f"Error fetching data from API: {str(e)}"}), 500
        
        if folder_path:
            try:
                folder_data = extract_folder_content(folder_path)
            except Exception as e:
                current_app.logger.error(f"Error extracting content from folder {folder_path}: {str(e)}")
                return jsonify({"error": f"Error extracting content from folder: {str(e)}"}), 500
        
        if website_url:
            try:
                extracted_data = extract_text_from_url(website_url)
                if isinstance(extracted_data, list) and extracted_data and extracted_data[0].get('tag') == 'error':
                    return jsonify({"error": extracted_data[0]['text']}), 400
                web_data = extracted_data
            except Exception as e:
                current_app.logger.error(f"Error extracting text from URL {website_url}: {str(e)}")
                return jsonify({"error": f"Error extracting text from URL: {str(e)}"}), 500
        
        inventory_data = get_formatted_inventory()
        if inventory_data:
            db_data.append({'page': 'inventory', 'text': inventory_data})
        
        if not pdf_data and not db_data and not folder_data and not web_data:
            return jsonify({"error": "No data provided. Please upload a file, provide a folder path, provide an API URL, provide a website URL, or ensure MongoDB has data."}), 400
       
        new_data = {
            "pdf_data": pdf_data,
            "db_data": db_data,
            "folder_data": folder_data,
            "web_data": web_data
        }
        
        # If chatbot.data is empty, initialize it as a list
        if not chatbot.data:
            chatbot.data = []
        
        # If chatbot.data is a string, parse it first
        if isinstance(chatbot.data, str):
            chatbot.data = json.loads(chatbot.data)
        
        # Append the new data to the existing data
        if isinstance(chatbot.data, list):
            chatbot.data.append(new_data)
        else:
            chatbot.data = [chatbot.data, new_data]
        
        # Convert the entire data structure back to a JSON string
        chatbot.data = json.dumps(chatbot.data)
        
        db.session.commit()
        
        return jsonify({"message": "Chatbot trained successfully"}), 200
    
    except Exception as e:
        current_app.logger.error(f"Unexpected error in train_chatbot: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@chatbot_bp.route('/delete_chatbot/<chatbot_id>', methods=['DELETE'])
@login_required
@handle_errors
def delete_chatbot(chatbot_id):
    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot or chatbot.user_id != session['user_id']:
        return jsonify({"error": "Chatbot not found or unauthorized"}), 404
    
    db.session.delete(chatbot)
    db.session.commit()
    
    return jsonify({"message": "Chatbot deleted successfully"}), 200



@chatbot_bp.route('/chatbot/<chatbot_id>/ask', methods=['POST'])
@handle_errors
def chatbot_ask(chatbot_id):
    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot:
        return jsonify({"error": "Chatbot not found"}), 404

    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Ensure chatbot.data is a string
        chatbot_data_str = chatbot.data if isinstance(chatbot.data, str) else json.dumps(chatbot.data)

        # Parse the JSON string
        chatbot_data = json.loads(chatbot_data_str)
        logging.info(f"Parsed chatbot data: {chatbot_data}")
        logging.info(f"Final chatbot data structure: {type(chatbot_data)}")

        # Handle both list and dictionary cases
        if isinstance(chatbot_data, list):
            # If it's a list, use the last item (most recent data)
            chatbot_data = chatbot_data[-1]
        elif not isinstance(chatbot_data, dict):
            raise ValueError("Invalid chatbot data format")

        logging.info(f"Raw chatbot data: {chatbot.data}")
        # Now chatbot_data should be a dictionary
        if any(keyword in question.lower() for keyword in ["proce", "inventory", "stock", "available", "category", "type"]):
            answer = get_inventory_rag_answer(json.dumps(chatbot_data), question)
        else:
            answer = get_general_answer(json.dumps(chatbot_data), question)

        return jsonify({"answer": answer})

    
    except json.JSONDecodeError as e:
        current_app.logger.error(f"JSON decode error: {str(e)}")
        return jsonify({"error": "Invalid chatbot data format"}), 500
    except Exception as e:
        current_app.logger.error(f"Error in chatbot_ask: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


@chatbot_bp.route('/get_chatbot_script/<chatbot_id>')
@handle_errors
def get_chatbot_script(chatbot_id):
    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot:
        return jsonify({"error": "Chatbot not found"}), 404
    
    integration_code = f"""

<div id="chatbot-icon" onclick="toggleChat()" style="position: fixed; bottom: 20px; right: 20px; width: 50px; height: 50px; background-color: #4a5568; border-radius: 50%; display: flex; justify-content: center; align-items: center; cursor: pointer; z-index: 1000;" aria-label="Toggle chat">
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
    </svg>
</div>

<div id="chatbot-container" style="position: fixed; bottom: 20px; right: 20px; width: 300px; height: 400px; border: 1px solid #ccc; border-radius: 10px; overflow: hidden;">
    <div id="chatbot-header" style="background-color: #4a5568; color: white; padding: 10px; font-weight: bold;">
        Chat with {chatbot.name}
    </div>
    <div id="chatbot-messages" style="height: 300px; overflow-y: auto; padding: 10px;"></div>
        <div id="chatbot-input" style="display: flex; padding: 10px;">
            <input type="text" id="chatbot-text" style="flex-grow: 1; padding: 5px; border: 1px solid #ccc; border-radius: 3px;" placeholder="Type your message..." aria-label="Chat message">
            <button id="send-button" onclick="sendMessage()" style="background-color: #4a5568; color: white; border: none; padding: 5px 10px; margin-left: 5px; cursor: pointer; border-radius: 3px;">Send</button>
    </div>
</div>

<script>

function toggleChat() {{
        const chatContainer = document.getElementById('chatbot-container');
        chatContainer.style.display = chatContainer.style.display === 'none' ? 'block' : 'none';
    }}
    
function sendMessage() {{
    var message = document.getElementById('chatbot-text').value;
    if (message.trim() === '') return;

    appendMessage('You: ' + message);
    document.getElementById('chatbot-text').value = '';

    fetch('{request.host_url}chatbot/{chatbot_id}/ask', {{
        method: 'POST',
        headers: {{
            'Content-Type': 'application/json',
        }},
        body: JSON.stringify({{ question: message }})
    }})
    .then(response => response.json())
    .then(data => {{
        appendMessage('Bot: ' + data.answer);
    }})
    .catch((error) => {{
        console.error('Error:', error);
        appendMessage('Bot: Sorry, I encountered an error.');
    }});
}}

function appendMessage(message) {{
    var messageDiv = document.createElement('div');
    messageDiv.textContent = message;
    messageDiv.style.marginBottom = '10px';
    messageDiv.style.padding = '5px';
    messageDiv.style.borderRadius = '5px';
    messageDiv.style.backgroundColor= '#e6f3ff';
    document.getElementById('chatbot-messages').appendChild(messageDiv);
    document.getElementById('chatbot-messages').scrollTop = document.getElementById('chatbot-messages').scrollHeight;
}}
</script>
 """

    return jsonify({ 'integration_code' : integration_code })





"""
    <div id="chatbot-container" style="position: fixed; bottom: 20px; right: 20px; width: 300px; height: 400px; border: 1px solid #ccc; border-radius: 10px; overflow: hidden;">
        <div id="chatbot-header" style="background-color: #4a5568; color: white; padding: 10px; font-weight: bold;">
            Chat with {chatbot.name}
        </div>
        <div id="chatbot-messages" style="height: 300px; overflow-y: auto; padding: 10px; background-color:white;"></div>
        <div id="chatbot-input" style="display: flex; padding: 10px;">
            <input type="text" id="chatbot-text" style="flex-grow: 1; padding: 5px;" placeholder="Type your message...">
            <button onclick="sendMessage()" style="background-color: #4a5568; color: white; border: none; padding: 5px 10px; margin-left: 5px;">Send</button>
        </div>
    </div>

    <script>
    function sendMessage() {{
        var message = document.getElementById('chatbot-text').value;
        if (message.trim() === '') return;

        appendMessage('You: ' + message);
        document.getElementById('chatbot-text').value = '';

        fetch('{request.host_url}chatbot/{chatbot_id}/ask', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
            }},
            body: JSON.stringify({{ question: message }})
        }})
        .then(response => response.json())
        .then(data => {{
            appendMessage('Bot: ' + data.answer);
        }})
        .catch((error) => {{
            console.error('Error:', error);
            appendMessage('Bot: Sorry, I encountered an error.');
        }});
    }}

    function appendMessage(message) {{
        var messageDiv = document.createElement('div');
        messageDiv.textContent = message;
        document.getElementById('chatbot-messages').appendChild(messageDiv);
        document.getElementById('chatbot-messages').scrollTop = document.getElementById('chatbot-messages').scrollHeight;
    }}
    </script>
     """
    
from flask import Blueprint, request, jsonify, session, current_app,logging
from models import Chatbot
from werkzeug.utils import secure_filename
from extensions import db
from utils.nlp_utils import preprocess_text, get_general_answer, get_inventory_rag_answer, get_formatted_inventory
from utils.file_utils import extract_text_from_pdf, read_text_file
from utils.api_utils import fetch_real_time_data
import json
import uuid
import os
from functools import wraps

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
    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot or chatbot.user_id != session['user_id']:
        return jsonify({"error": "Chatbot not found or unauthorized"}), 404
    
    file = request.files.get('file')
    api_url = request.form.get('api_url')
    
    pdf_data = []
    db_data = []
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension == '.pdf':
            pdf_text = extract_text_from_pdf(filepath)
            pdf_data.extend(pdf_text)
        elif file_extension in ['.txt', '.md', '.rst']:
            raw_text = read_text_file(filepath)
            pdf_data.append({'page': 'file', 'text': raw_text})
        else:
            os.remove(filepath)
            return jsonify({"error": "Unsupported file type"}), 400
        
        os.remove(filepath)
    
    if api_url:
        real_time_data = fetch_real_time_data(api_url)
        if real_time_data:
            real_time_text = json.dumps(real_time_data, indent=2)
            db_data.append({'page': 'real_time', 'text': real_time_text})
    
    inventory_data = get_formatted_inventory()
    if inventory_data:
        db_data.append({'page': 'inventory', 'text': inventory_data})
    
    if not pdf_data and not db_data:
        return jsonify({"error": "No data provided. Please upload a file, provide an API URL, or ensure MongoDB has data."}), 400
    
    # Structure the data in JSON format
    chatbot.data = json.dumps({
    "pdf_data": pdf_data,
    "db_data": db_data
    })
    
    db.session.commit()
    
    return jsonify({"message": "Chatbot trained successfully"}), 200



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

    chatbot_data = chatbot.data if isinstance(chatbot.data, str) else json.dumps(chatbot.data)

    if any(keyword in question.lower() for keyword in ["price", "inventory", "stock", "available", "category", "type"]):
        answer = get_inventory_rag_answer(chatbot_data, question)
    else:
        answer = get_general_answer(chatbot_data, question)

    return jsonify({"answer": answer})



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
    
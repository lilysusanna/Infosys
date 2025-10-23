from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import requests
import json
import PyPDF2
from werkzeug.utils import secure_filename
import openai

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/static/*": {"origins": "*"}
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('knowledge_base', exist_ok=True)

# AI Configuration - Using OpenAI API for intelligent responses
# Prefer a hardcoded key if provided, else fall back to environment variable
HARDCODED_OPENAI_API_KEY = " "# <--- OPTIONAL: put your OpenAI API key here to skip env setup

openai.api_key = HARDCODED_OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")

if not openai.api_key:
    # Helpful console guidance if key is missing; the app will still run but AI calls may return fallback responses
    print("WARNING: No OpenAI API key configured. Set HARDCODED_OPENAI_API_KEY in app.py or set OPENAI_API_KEY env var.")
    print("Set it in PowerShell with: $env:OPENAI_API_KEY = 'your_key_here'")

# Global variables to store document data
documents = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def get_ai_response(query, document_context=""):
    """Get AI-powered response using OpenAI"""
    try:
        # Create a comprehensive prompt
        system_prompt = """You are a helpful MSME (Micro, Small and Medium Enterprises) support assistant. 
        Provide accurate, concise, and helpful responses about business registration, compliance, and MSME-related queries.
        If document context is provided, use it to give specific answers. Be brief but comprehensive."""
        
        user_prompt = f"""
        Query: {query}
        
        {f"Document Context: {document_context}" if document_context else ""}
        
        Please provide a helpful, accurate response. If the query is about uploaded documents, use the context provided.
        If it's a general MSME question, provide relevant information about business registration, compliance, or MSME services.
        """
        
        # Use the newer OpenAI API format
        client = openai.OpenAI(api_key=openai.api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"AI API Error: {e}")
        # Return a fallback response instead of the error message
        if "rate limit" in str(e).lower():
            return "I'm currently experiencing high demand. Please try again in a moment."
        elif "authentication" in str(e).lower() or "api key" in str(e).lower():
            return "There's an issue with the AI service configuration. Please check the API key."
        else:
            return "I'm here to help with MSME-related questions! What would you like to know about business registration, compliance, or MSME services?"

def process_document(file_path, filename):
    """Process uploaded document for AI analysis"""
    global documents
    
    # Extract text from PDF
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        return False, "No text could be extracted from the PDF"
    
    # Store document data for AI context
    doc_id = str(uuid.uuid4())
    doc_data = {
        'id': doc_id,
        'filename': filename,
        'text': text
    }
    
    documents.append(doc_data)
    
    # Save to knowledge base
    knowledge_file = os.path.join('knowledge_base', f"{doc_id}.json")
    with open(knowledge_file, 'w', encoding='utf-8') as f:
        json.dump(doc_data, f, ensure_ascii=False, indent=2)
    
    return True, f"Document processed successfully and ready for AI analysis!"

def get_document_context():
    """Get all document text for AI context"""
    if not documents:
        return ""
    
    context_parts = []
    for doc in documents:
        context_parts.append(f"Document: {doc['filename']}\nContent: {doc['text'][:2000]}...")  # Limit to 2000 chars per doc
    
    return "\n\n".join(context_parts)

@app.route('/')
def index():
    return send_from_directory('build', 'index.html')

@app.route('/test')
def test():
    return send_from_directory('.', 'test.html')

@app.route('/static/css/<path:filename>')
def static_css(filename):
    return send_from_directory('build/static/css', filename)

@app.route('/static/js/<path:filename>')
def static_js(filename):
    return send_from_directory('build/static/js', filename)

@app.route('/favicon.ico')
def favicon():
    return '', 404

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle document upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        file.save(file_path)
        
        # Process the document
        success, message = process_document(file_path, filename)
        
        if success:
            return jsonify({
                'message': message,
                'filename': filename,
                'file_id': file_id
            })
        else:
            return jsonify({'error': message}), 400
    
    return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with AI"""
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get document context if available
    document_context = get_document_context()
    
    # Get AI response
    ai_response = get_ai_response(message, document_context)
    
    # If AI response is the fallback message, try a simple rule-based response
    if "I'm here to help with MSME-related questions!" in ai_response:
        ai_response = get_simple_response(message, document_context)
    
    return jsonify({
        'response': ai_response,
        'has_document_context': len(documents) > 0
    })

def get_simple_response(message, document_context=""):
    """Simple rule-based responses as fallback"""
    message_lower = message.lower()
    
    # Document-based responses
    if document_context and any(word in message_lower for word in ['what', 'summarize', 'explain', 'tell me about']):
        return f"Based on your uploaded document, I can see it contains relevant information. Here are the key points: {document_context[:500]}..."
    
    # General MSME responses
    if any(word in message_lower for word in ['register', 'registration', 'business']):
        return """To register your business as an MSME:
1. Visit the Udyam registration portal (udyamregistration.gov.in)
2. Provide your Aadhaar number and PAN card
3. Fill in business details (name, address, type of business)
4. Submit the application
5. You'll receive your Udyam Registration Number (URN)

Required documents: PAN card, Aadhaar card, business address proof, and bank account details."""
    
    elif any(word in message_lower for word in ['document', 'documents', 'required']):
        return """For MSME registration, you need:
• PAN Card
• Aadhaar Card  
• Business address proof
• Bank account details
• Business type and activity description
• Investment in plant & machinery details"""
    
    elif any(word in message_lower for word in ['benefit', 'benefits', 'advantage']):
        return """MSME registration benefits include:
• Priority in government tenders
• Collateral-free loans up to ₹10 lakhs
• Access to government schemes and subsidies
• Tax benefits and exemptions
• Skill development programs
• Credit guarantee schemes"""
    
    elif any(word in message_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm your MSME support assistant. I can help you with business registration, compliance, and MSME-related queries. What would you like to know?"
    
    else:
        return "I'm here to help with MSME-related questions! You can ask me about business registration, required documents, benefits, or upload documents for specific guidance."

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of uploaded documents"""
    doc_list = [{'id': doc['id'], 'filename': doc['filename']} for doc in documents]
    return jsonify({'documents': doc_list})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify, render_template_string
import logging
from loguru import logger
from sentence_transformers import SentenceTransformer
import torch
torch.set_num_threads(1)
from chatbot_qa import DocumentRetrievalQA

# Initialize the Flask application
app = Flask(__name__)

api_key = os.environ["API_KEY"]

# Initialization code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1').to(device)
local_folder = "vector_store_1"
qa_system = DocumentRetrievalQA(model, device, local_folder)
qa_system.set_openai_api_key(api_key)

# Setup standard logging
logging.basicConfig(level=logging.INFO)
logger.add("file.log", rotation="500 MB")  # Loguru example for logging to a file

@app.route('/')
def index():
    company_names = [
        'Sunshine Gloria',
        'Deep Mind',
        'Tropical Flowers',
        'Moonlight Shadow'
    ]

    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Document Retrieval QA</title>
    </head>
    <body>
        <h1>Hi, I'm Chatty, your friendly AI chatbot.</h1>

        <p>You can ask me for the financial status of the following companies:</p>

        <ul>
            {% for company_name in company_names %}
                <li>{{ company_name }}</li>
            {% endfor %}
        </ul>

        <form id="qa-form" action="/ask" method="post">
            <textarea name="question" rows="4" cols="50" placeholder="Type your question here..."></textarea><br>
            <input type="submit" value="Ask">
        </form>

        <div id="response"></div>

        <script>
            const form = document.querySelector('#qa-form');
            form.onsubmit = async (event) => {
                event.preventDefault();
                const formData = new FormData(form);
                document.getElementById('response').innerHTML = 'Loading...'; // Show loading message
                const response = await fetch('/ask', {
                    method: 'POST',
                    body: formData
                });
                const answers = await response.json();
                document.getElementById('response').innerHTML = answers.join('<br>'); // Display the response
            };
        </script>
    </body>
    </html>
    ''', company_names=company_names)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    try:
        answers = qa_system.answer_query(question)
        return jsonify(answers)
    except Exception as e:
        logger.error(f"Error processing question '{question}': {str(e)}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

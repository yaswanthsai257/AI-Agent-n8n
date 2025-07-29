import os
import json
import uuid
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv

# Using Mistral as requested
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- Initial Setup ---
load_dotenv()
app = Flask(__name__)
CORS(app, expose_headers=["X-Conversation-Id"])

# --- In-Memory Chat History ---
chat_histories = {}

# --- Configure LangChain with Mistral AI ---
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY is not set in the .env file.")

llm = ChatMistralAI(api_key=api_key)

# --- THE FIX: New and Correct Complex Example ---
system_prompt_text = """
You are an expert n8n workflow developer. Your sole purpose is to generate or modify the JSON code for an n8n workflow based on a user's request.
Analyze the user's request, your previous responses, and the examples below. Your output MUST match the exact JSON structure of the examples, especially for branching logic and chained connections.

**RULES:**
1.  Your final response must be ONLY the raw JSON object, starting with `{` and ending with `}`. Do not include "```json" or any other text.
2.  The `connections` object is critical. The IF node has two outputs. The first (index 0) is for TRUE, the second (index 1) is for FALSE.
3.  Chain multiple steps correctly. A node connects to B, then B connects to C. Do not connect A to both B and C from the same output.

--- EXAMPLE 1: SIMPLE WORKFLOW ---
USER REQUEST: "When a webhook gets a POST request, add the incoming data as a new row in Google Sheets."
CORRECT JSON OUTPUT:
{ "name": "Webhook to Sheets", "nodes": [ { "parameters": {}, "name": "Start", "type": "n8n-nodes-base.start", "typeVersion": 1, "position": [ 250, 300 ] }, { "parameters": { "path": "webhook-test" }, "name": "Webhook", "type": "n8n-nodes-base.webhook", "typeVersion": 1, "position": [ 400, 300 ] }, { "parameters": { "sheetId": "{{ $credentials.googleSheet.sheetId }}", "range": "A:A" }, "name": "Google Sheets", "type": "n8n-nodes-base.googleSheets", "typeVersion": 4, "position": [ 600, 300 ] } ], "connections": { "Webhook": { "main": [ [ { "node": "Google Sheets", "type": "main", "index": 0 } ] ] } } }
--- EXAMPLE 1 END ---

--- EXAMPLE 2: NEW AND CORRECT COMPLEX WORKFLOW ---
USER REQUEST: "When a new form is submitted on our website, check if the customer's message contains 'urgent'. If it does, create a high-priority Asana task, then send a Slack notification to #support-leads, and finally send an email to the customer using Gmail. If not, add the customer's information to a Google Sheet, and then send a welcome email to the customer using Sendinblue."
CORRECT JSON OUTPUT:
{ "name": "New Customer Inquiry", "nodes": [ { "parameters": {}, "name": "Start", "type": "n8n-nodes-base.start", "typeVersion": 1, "position": [ 250, 300 ] }, { "parameters": { "path": "customer-inquiry" }, "name": "Webhook", "type": "n8n-nodes-base.webhook", "typeVersion": 1, "position": [ 400, 300 ] }, { "parameters": { "conditions": { "boolean": [ { "value1": "{{$json.body.message}}", "operation": "contains", "value2": "urgent" } ] } }, "name": "IF - Is it urgent?", "type": "n8n-nodes-base.if", "typeVersion": 1, "position": [ 620, 300 ] }, { "parameters": { "projectId": "{{ $credentials.asana.projectId }}", "name": "New Urgent Lead: {{ $json.body.name }}" }, "name": "Create Asana Task", "type": "n8n-nodes-base.asana", "typeVersion": 1, "position": [ 840, 200 ] }, { "parameters": { "channel": "#support-leads", "text": "🔥 New URGENT inquiry from {{ $json.body.name }}!" }, "name": "Send Slack Alert", "type": "n8n-nodes-base.slack", "typeVersion": 1, "position": [ 1060, 200 ] }, { "parameters": { "to": "{{ $json.body.email }}", "subject": "Re: Your Urgent Inquiry", "text": "We have received your urgent request and will be in touch shortly." }, "name": "Send Gmail Confirmation", "type": "n8n-nodes-base.gmail", "typeVersion": 1, "position": [ 1280, 200 ] }, { "parameters": { "sheetId": "{{ $credentials.googleSheets.sheetId }}", "range": "Leads!A:C", "values": { "values": [ [ "{{ $json.body.name }}", "{{ $json.body.email }}", "{{ $json.body.message }}" ] ] } }, "name": "Add to Google Sheet", "type": "n8n-nodes-base.googleSheets", "typeVersion": 4, "position": [ 840, 400 ] }, { "parameters": { "recipientEmail": "{{ $json.body.email }}", "templateId": 123 }, "name": "Send Welcome Email", "type": "n8n-nodes-base.sendinblue", "typeVersion": 1, "position": [ 1060, 400 ] } ], "connections": { "Webhook": { "main": [ [ { "node": "IF - Is it urgent?", "type": "main", "index": 0 } ] ] }, "IF - Is it urgent?": { "main": [ [ { "node": "Create Asana Task", "type": "main", "index": 0 } ], [ { "node": "Add to Google Sheet", "type": "main", "index": 1 } ] ] }, "Create Asana Task": { "main": [ [ { "node": "Send Slack Alert", "type": "main", "index": 0 } ] ] }, "Send Slack Alert": { "main": [ [ { "node": "Send Gmail Confirmation", "type": "main", "index": 0 } ] ] }, "Add to Google Sheet": { "main": [ [ { "node": "Send Welcome Email", "type": "main", "index": 0 } ] ] } } }
--- EXAMPLE 2 END ---

Now, generate or modify the JSON for the new user request based on the conversation so far.
"""

system_message = SystemMessage(content=system_prompt_text)

# --- API Endpoint with Memory ---
@app.route("/api/generate-workflow-stream", methods=["POST"])
def generate_workflow_stream():
    data = request.get_json()
    user_prompt = data.get("prompt")
    conversation_id = data.get("conversation_id") or str(uuid.uuid4())

    if not user_prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if conversation_id not in chat_histories:
        chat_histories[conversation_id] = [system_message]
    
    chat_histories[conversation_id].append(HumanMessage(content=user_prompt))
    messages_to_send = chat_histories[conversation_id]

    def stream_response():
        ai_full_response = ""
        try:
            for chunk in llm.stream(messages_to_send):
                content = chunk.content
                cleaned_chunk = content.replace("```json", "").replace("```", "")
                ai_full_response += cleaned_chunk
                yield cleaned_chunk
            
            chat_histories[conversation_id].append(AIMessage(content=ai_full_response))
        except Exception as e:
            print(f"An error occurred during streaming: {e}")
            yield json.dumps({"error": "Failed to stream response from AI."})

    resp = Response(stream_response(), mimetype='text/plain')
    resp.headers['X-Conversation-Id'] = conversation_id
    return resp

# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=True, port=5001)
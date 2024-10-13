import argparse
import os
import uuid
import traceback
import gradio as gr
from llama_index.llms.openai_like import OpenAILike
from llama_parse import LlamaParse
from pinecone import Pinecone, ServerlessSpec

# Initialize global variables
pc = None
openailike_client = None
parser = None
uploaded_files = []

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Chat-with-document demo")
parser.add_argument("--llama-parse-api-key", help="LlamaCloud API Key")
parser.add_argument("--pinecone-api-key", help="Pinecone API Key")
parser.add_argument("--openai-api-base", default="https://api.openai.com/v1", help="OpenAI API Base URL")
parser.add_argument("--openai-api-key", help="OpenAI API Key")
parser.add_argument("--pinecone-index-name", default="pdf-parser-index", help="Pinecone Index Name")
parser.add_argument("--pinecone-region", default="us-east-1", help="Pinecone Region")
args = parser.parse_args()

# Set environment variables from command-line arguments
args.llama_parse_api_key = args.llama_parse_api_key or os.getenv("LLAMA_PARSE_API_KEY")
args.pinecone_api_key = args.pinecone_api_key or os.getenv("PINECONE_API_KEY")
args.openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")

def initialize_services():
    global pc, openailike_client, parser
    if not args.pinecone_api_key or not args.llama_parse_api_key:
        raise ValueError("API keys are not provided.")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=args.pinecone_api_key)
    if args.pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=args.pinecone_index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=args.pinecone_region)
        )
    
    # Initialize OpenAI client
    openailike_client = OpenAILike(
        model="gpt-4o-mini",
        is_chat_model=True,
        api_base=args.openai_api_base,
        api_key=args.openai_api_key
    )

    # Initialize LlamaParse
    parser = LlamaParse(api_key=args.llama_parse_api_key)

def get_embedding(text):
    return pc.inference.embed(model="multilingual-e5-large", inputs=[text]).data[0]["values"]

def handle_chat(message):
    if not pc or not openailike_client or not parser:
        return "💡 Please initialize the settings first."
    
    try:
        query_embedding = get_embedding(message)
        index = pc.Index(args.pinecone_index_name)
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        contexts = "\n\n".join(item.metadata['text'] for item in results['matches'])
        
        response = openailike_client.stream_chat(messages=[
            {"role": "system", "content": f"Use the following context to answer: {contexts}"},
            {"role": "user", "content": message}
        ])
        return "".join(chunk.message.content for chunk in response)

    except Exception as e:
        return f"💡 Error: {str(e)}"

def parse_and_ingest(new_files):
    global uploaded_files
    if not new_files:
        return "No new files uploaded."
    
    index = pc.Index(args.pinecone_index_name)
    documents = parser.load_data(new_files)

    for doc in documents:
        embedding = get_embedding(doc.text)
        index.upsert(vectors=[{
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {"text": doc.text}
        }])
    
    uploaded_files.extend(new_files)
    return f"Parsed and ingested {len(documents)} documents."

def clear_uploaded_files():
    global uploaded_files
    uploaded_files = []
    return "Uploaded files cleared."

# Gradio interface
with gr.Blocks(title="🦙 Chat-with-document Demo") as demo:
    gr.Markdown("# 🦙 Chat-with-document Demo")

    with gr.Tab("Chat"):
        chat_input = gr.Textbox(label="Ask a question")
        chat_output = gr.Textbox(label="Response", interactive=False)
        chat_button = gr.Button("Send")
        chat_button.click(fn=lambda msg: handle_chat(msg), inputs=chat_input, outputs=chat_output)

    with gr.Tab("Document"):
        file_input = gr.File(label="Upload PDF", file_count="multiple")
        parse_button = gr.Button("Parse and Ingest")
        parse_output = gr.Textbox(label="Status", interactive=False)
        parse_button.click(fn=lambda files: parse_and_ingest(files), inputs=file_input, outputs=parse_output)

    with gr.Tab("Settings"):
        llama_parse_api_key = gr.Textbox(label="LlamaCloud API Key")
        pinecone_api_key = gr.Textbox(label="Pinecone API Key")
        pinecone_region = gr.Textbox(label="Pinecone Region")
        pinecone_index_name = gr.Textbox(label="Pinecone Index Name")
        openai_api_base = gr.Textbox(label="OpenAI Base URL")
        openai_api_key = gr.Textbox(label="OpenAI API Key")
        update_button = gr.Button("Update Settings")
        update_button.click(fn=initialize_services, inputs=[llama_parse_api_key, pinecone_api_key, pinecone_region, pinecone_index_name, openai_api_base, openai_api_key])

# Start the Gradio app
demo.launch()

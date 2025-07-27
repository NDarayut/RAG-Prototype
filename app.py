import gradio as gr
import os
import shutil
import fitz  # PyMuPDF
from simple_rag import generate_data_store, ask_question, DATA_PATH, CHROMA_PATH
import tempfile
from typing import List, Tuple, Optional

# Global variables to store chat history and processed files
chat_history = []
uploaded_files_info = []

def clear_vector_store():
    """Clear the Chroma vector store"""
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            return "✅ Vector store cleared successfully."
        else:
            return "ℹ️ No vector store found to clear."
    except Exception as e:
        return f"❌ Failed to clear vector store: {e}"

def process_uploaded_files(files):
    """Process uploaded PDF files and generate vector store"""
    global uploaded_files_info
    
    if not files:
        return "⚠️ No files uploaded.", ""
    
    # Clear existing data
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
    os.makedirs(DATA_PATH, exist_ok=True)
    
    uploaded_files_info = []
    file_info_text = "📂 **Uploaded Files:**\n\n"
    
    try:
        for file in files:
            if file is None:
                continue
                
            # Copy file to data directory
            file_name = os.path.basename(file.name)
            file_path = os.path.join(DATA_PATH, file_name)
            shutil.copy2(file.name, file_path)
            
            uploaded_files_info.append({
                'name': file_name,
                'path': file_path
            })
            
            file_info_text += f"- 📄 `{file_name}`\n"
        
        # Generate vector store
        generate_data_store()
        
        status_msg = f"✅ Successfully processed {len(uploaded_files_info)} file(s) and generated vector store!"
        
        return status_msg, file_info_text
        
    except Exception as e:
        return f"❌ Error processing files: {e}", ""

def get_pdf_preview(file_path: str, max_pages: int = 2) -> List[any]:
    """Generate preview images from PDF"""
    try:
        doc = fitz.open(file_path)
        images = []
        
        for page_num in range(min(len(doc), max_pages)):
            page = doc.load_page(page_num)
            # Render page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 1.5x scale
            img_bytes = pix.tobytes("png")
            
            # Save to temporary file for Gradio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_file.write(img_bytes)
            temp_file.close()
            images.append(temp_file.name)
        
        doc.close()
        return images
    except Exception as e:
        print(f"Error generating PDF preview: {e}")
        return []

def chat_with_docs(message: str, history: Optional[List] = None) -> Tuple[str, List]:
    """Handle chat interactions with documents"""
    global chat_history
    
    if not message.strip():
        return "", history or []
    
    if not uploaded_files_info:
        response = "⚠️ Please upload and process PDF files first before asking questions."
        if history is None:
            history = []
        history.append([message, response])
        return "", history
    
    try:
        # Get answer from RAG system
        answer, context_chunks = ask_question(message)
        
        # Format response with context
        response = f"**Answer:** {answer}\n\n"
        
        if context_chunks:
            response += "**📖 Sources:**\n"
            for i, chunk in enumerate(context_chunks, 1):
                response += f"\n{i}. **{chunk['filename']}** (Page {chunk['page']})\n"
                response += f"```\n{chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}\n```\n"
        
        # Add to chat history
        chat_history.append({
            "question": message,
            "answer": answer,
            "context": context_chunks
        })
        
        # Update Gradio chat history
        if history is None:
            history = []
        history.append([message, response])
        
        return "", history
        
    except Exception as e:
        error_msg = f"❌ Error processing question: {e}"
        if history is None:
            history = []
        history.append([message, error_msg])
        return "", history

def clear_chat():
    """Clear chat history"""
    global chat_history
    chat_history = []
    return []

def show_file_previews():
    """Generate file preview gallery"""
    if not uploaded_files_info:
        return []
    
    preview_images = []
    for file_info in uploaded_files_info:
        images = get_pdf_preview(file_info['path'], max_pages=2)
        preview_images.extend(images)
    
    return preview_images

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="DocChat - PDF Question Answering",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-container {
            height: 500px;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 💬 DocChat – Ask Anything From Your PDFs
            
            Upload your PDF documents and ask questions about their content using advanced RAG (Retrieval-Augmented Generation) technology.
            """
        )
        
        with gr.Tab("📂 Upload & Process"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_upload = gr.File(
                        label="Upload PDF Files",
                        file_count="multiple",
                        file_types=[".pdf"],
                        height=150
                    )
                    
                    process_btn = gr.Button("⚡ Process Files & Generate Vector Store", variant="primary")
                    
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=2
                    )
                
                with gr.Column(scale=1):
                    files_info = gr.Markdown("No files uploaded yet.")
                    
                    clear_vector_btn = gr.Button("🗑️ Clear Vector Store", variant="secondary")
                    clear_status = gr.Textbox(label="Clear Status", interactive=False, lines=1)
            
            # File previews
            with gr.Row():
                preview_gallery = gr.Gallery(
                    label="📖 PDF Previews (First 2 pages of each file)",
                    show_label=True,
                    elem_id="preview_gallery",
                    columns=3,
                    rows=2,
                    height="auto"
                )
        
        with gr.Tab("💬 Chat"):
            with gr.Row():
                chatbot = gr.Chatbot(
                    label="Chat with your documents",
                    height=400,
                    elem_classes=["chat-container"]
                )
            
            with gr.Row():
                with gr.Column(scale=4):
                    msg = gr.Textbox(
                        label="Ask a question",
                        placeholder="e.g., What is the main conclusion of the document?",
                        lines=2
                    )
                with gr.Column(scale=1):
                    submit_btn = gr.Button("🚀 Ask", variant="primary")
                    clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
        
        with gr.Tab("ℹ️ Instructions"):
            gr.Markdown(
                """
                ## How to use DocChat:
                
                1. **Upload PDFs**: Go to the "Upload & Process" tab and select your PDF files
                2. **Process Files**: Click "Process Files & Generate Vector Store" to index your documents
                3. **Ask Questions**: Switch to the "Chat" tab and start asking questions about your documents
                4. **View Sources**: Each answer includes the source passages from your PDFs
                
                ## Features:
                - 📄 **Multi-PDF Support**: Upload and query multiple documents at once
                - 🔍 **Semantic Search**: Find relevant information even with different wording
                - 📖 **Source Citations**: See exactly which parts of your documents were used to generate answers
                - 🖼️ **PDF Previews**: Visual preview of your uploaded documents
                - 💾 **Vector Store Management**: Clear and rebuild your document index as needed
                
                ## Tips:
                - Ask specific questions for better results
                - Try rephrasing questions if you don't get the expected answer
                - Check the source citations to verify the information
                """
            )
        
        # Event handlers
        process_btn.click(
            fn=process_uploaded_files,
            inputs=[file_upload],
            outputs=[status_output, files_info]
        ).then(
            fn=show_file_previews,
            outputs=[preview_gallery]
        )
        
        clear_vector_btn.click(
            fn=clear_vector_store,
            outputs=[clear_status]
        )
        
        submit_btn.click(
            fn=chat_with_docs,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            fn=chat_with_docs,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        clear_chat_btn.click(
            fn=clear_chat,
            outputs=[chatbot]
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
#pip install langchain gradio openai

import os
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
import gradio as gr

os.environ['OPENAI_API_KEY'] = '**********' #Add your Open A key here

loader = PyPDFLoader("whitepaper.pdf")  #add your pdf here in the same directory.
documents = loader.load()

def summarize_pdf (pdf_file_path, custom_prompt=""):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    
    return summary


def main():
    input_pdf_path = gr.inputs.Textbox(label="Enter the PDF file path")
    output_summary = gr.outputs.Textbox(label="Summary")
    
    interface = gr.Interface(
        fn = summarize_pdf,
        inputs = input_pdf_path,
        outputs = output_summary,
        title = "PDF Summarizer",
        description = "This app allows you to summarize your PDF files.",
    ).launch(share=True)

main()
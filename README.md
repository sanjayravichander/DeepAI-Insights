## DeepDive AI - Research Paper Insights
## Overview
DeepDive AI is a Retrieval-Augmented Generation (RAG) application designed to help users extract insights from complex AI research papers. Users can upload a research paper in PDF format, ask questions, and receive concise, context-aware answers based on the content of the paper. The application uses Generative AI to generate responses and Semantic Search to retrieve relevant sections from the paper.

This project is particularly useful for researchers, students, and AI enthusiasts who want to quickly understand and navigate scientific literature.

## Features
## Paper Upload: Users can upload AI research papers in PDF format.

## Query-Based Retrieval: The application uses semantic search to retrieve specific sections or paragraphs related to user queries.

## Summary Generation: Generates concise summaries of key sections (e.g., abstract, methodology, results).

## Interactive Q&A: Provides answers to natural language questions based on the paper's content.

## Citation Assistance: Offers citation suggestions for specific sections or ideas in the paper.

## Generative AI Integration: Uses a Generative AI model (instead of OpenAI) to generate responses.

## Tech Stack
## Text Extraction: PyPDF2

## Vectorization: Sentence Transformers (all-MiniLM-L6-v2)

## Vector Database: FAISS (for efficient similarity search)

## Generative AI: Custom Generative AI model (e.g., Hugging Face Transformers, GPT-J, or similar)

## Frontend: Streamlit

## Deployment: Streamlit Cloud

## Installation
## To set up the project locally, follow these steps:

## Clone the Repository:
git clone https://github.com/sanjayravichander/deepdive-ai.git
cd deepdive-ai

## Install Dependencies:
Ensure you have Python 3.8+ installed. Then, install the required libraries:
pip install -r requirements.txt

## The requirements.txt file includes:
streamlit 
PyPDF2 
sentence-transformers 
faiss-cpu 
openai
python_dotenv
google-generativeai

## Set Up Environment Variables:
## Create a .env file in the root directory and add any required API keys or configurations for your Generative AI model:
GENERATIVE_AI_API_KEY=your_generative_ai_api_key

## Run the Application:
## Start the Streamlit application:
## streamlit run app.py

## Acknowledgments
## Sentence Transformers for semantic search.

## FAISS for efficient vector similarity search.

## Hugging Face Transformers for Generative AI integration.

## Streamlit for the user interface.



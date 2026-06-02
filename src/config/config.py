from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

hf_api_key = os.getenv("HF_TOKEN")
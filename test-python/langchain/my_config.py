import os

# I'm omitting all other keys
HUGGINGFACEHUB_API_TOKEN = ""
#OPENAI_API_KEY = "sk-tAAt2GsImKtUeRH4897b604eA1C248B1A119DeBf9112F74b"
#OPENAI_API_BASE = "https://api.xty.app/v1"

OPENAI_API_KEY = "sk-gptpkWESeYTfuxrAGe9IMDXYf0GEvd4j18JBhERJS3E9hhie"
OPENAI_API_BASE = "https://api.gptdos.com/v1"



def set_environment():
 variable_dict = globals().items()
 for key, value in variable_dict:
    if "API" in key or "OPENAI" in key or "ID" in key:
        os.environ[key] = value
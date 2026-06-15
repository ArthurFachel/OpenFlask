from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms import DeepInfra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import AutoPeftModelForCausalLM
from dotenv import load_dotenv
import torch
from argparse import Namespace

import os

load_dotenv()


def ensure_namespace(args):
    if isinstance(args, dict):
        args = Namespace(**args)
        return args
    elif not isinstance(args, Namespace):
        raise TypeError("args must be a dict or an argparse.Namespace instance")
    else:
        return args
    
class EmbeddingModelFactory:
    """
    Factory class to create an embedding model based on the provided arguments.
    Supports both local and remote models, with options for different embedding types.
    Supports Hugging Face and DeepInfra models.

    LlamaIndex → LangChain equivalences:
    - DeepInfraEmbeddingModel → OpenAIEmbeddings (pointed at DeepInfra's endpoint)
    - HuggingFaceEmbedding    → HuggingFaceEmbeddings

    Args:
        args (Namespace): Arguments containing model type, local/remote preference,
                          and other configurations.
    """

    def __init__(self, args=None):
        args = ensure_namespace(args)

        if not args.embedding_local:
        # TODO: explorar a possibilidade de usar embeddings da OpenAI
        #
        #     from langchain_openai import OpenAIEmbeddings

        #     self.__embed_model = OpenAIEmbeddings(
        #         model=args.embedding,
        #         openai_api_key=os.getenv("DEEPINFRA_API_KEY"),
        #         openai_api_base="https://api.deepinfra.com/v1/openai",
        #     )

        # else:
        #     try:
            from huggingface_hub import login
            login(os.environ["HF_TOKEN"])
            # except Exception:
            #   raise Exception("Please install huggingface_hub to use local models")

            self.__embed_model = HuggingFaceEmbeddings(
                model_name=args.embedding,
                cache_folder=args.cache_dir,
                model_kwargs={
                    "device": args.embedding_device,
                    "trust_remote_code": True,
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )

    def get_model(self):
        return self.__embed_model
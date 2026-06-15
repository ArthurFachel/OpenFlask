"""

Script responsável por converter documentos para Markdown, carregá-los,
dividi-los em chunks e indexá-los no ChromaDB usando embeddings de texto.

Fluxo principal:
    1. Converte arquivos do diretório de entrada para Markdown via docling.
    2. Carrega os arquivos Markdown com LangChain DirectoryLoader.
    3. Normaliza os metadados de cada documento.
    4. Divide os documentos em chunks com o splitter escolhido.
    5. Indexa os chunks no ChromaDB em lotes.

Argumentos de linha de comando:
    --data_path         Diretório com os arquivos de entrada (padrão: "./data/RAG_files").
    --collection_name   Nome da coleção ChromaDB (padrão: "book_collection").
    --splitter          Estratégia de chunking: sentence | sentence_window | semantic |
                        token | hierarchical | markdown (padrão: "sentence").
    --gpu               ID da GPU a expor via CUDA_VISIBLE_DEVICES (padrão: "2").
    -r / --recursive    Busca recursiva de arquivos .md no diretório de saída.
    -e / --embedding    Nome do modelo de embedding (padrão: "all-MiniLM-L6-v2").
    --embedding_device  Dispositivo para inferência do embedding (padrão: "cuda:2").
    --embedding_local   Flag para usar modelo de embedding local.
    --cache_dir         Diretório de cache para modelos HuggingFace
                        (padrão: "/mnt/E-SSD/model_cache/hf").
"""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from models import EmbeddingModelFactory
import chromadb
import os
from dotenv import load_dotenv
from const import DOCUMENT_REFERENCES
from argparse import ArgumentParser
from tqdm import tqdm
from docling.document_converter import DocumentConverter

load_dotenv()


def convert_to_markdown(file_path: str, output_dir: str) -> str:
    """
    Converts a file to Markdown format using docling and saves the result to a directory.

    Parameters:
    - file_path (str): Path to the source file or URL.
    - output_dir (str): Directory where the Markdown file will be saved.

    Supported input formats:
    - .txt, .docx, .odt, .rtf, .html/.htm, .pdf, .md, .json
    - URLs para documentos online

    Returns:
    - str: Full path to the generated Markdown file.
    """
    if not (os.path.isfile(file_path) or file_path.startswith(("http://", "https://"))):
        raise FileNotFoundError(f"File not found and not a valid URL: {file_path}")

    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"Invalid output directory: {output_dir}")

    converter = DocumentConverter()
    result = converter.convert(file_path)
    markdown_content = result.document.export_to_markdown()

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, base_name + ".md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return output_path


def get_splitter(option: str, embed_model=None):
    """
    Instancia e retorna um text splitter LangChain conforme a estratégia escolhida.

    Equivalências entre LlamaIndex e LangChain:
        - SentenceWindowNodeParser   → RecursiveCharacterTextSplitter com overlap.
        - SemanticSplitterNodeParser → SemanticChunker (langchain_experimental).
        - HierarchicalNodeParser     → RecursiveCharacterTextSplitter com chunks menores.

    Args:
        option (str): Estratégia de chunking. Valores aceitos:
            - "sentence"        RecursiveCharacterTextSplitter (chunk 1024, overlap 200).
            - "sentence_window" RecursiveCharacterTextSplitter (chunk 512, overlap 256).
            - "semantic"        SemanticChunker baseado em embeddings (requer embed_model).
            - "token"           TokenTextSplitter (chunk 1024, overlap 20).
            - "hierarchical"    RecursiveCharacterTextSplitter (chunk 128, overlap 20).
            - "markdown"        MarkdownTextSplitter (chunk 1024, overlap 100).
        embed_model: Modelo de embeddings LangChain. Obrigatório apenas quando
            option="semantic"; ignorado nos demais casos.

    Returns:
        TextSplitter: Instância do splitter configurado, compatível com
            `split_documents()` da LangChain.
    Raises:
        ValueError: Se option="semantic" e embed_model não for fornecido.
        ValueError: Se option não corresponder a nenhum valor aceito.
    """
    if option == "sentence":
        return RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    elif option == "sentence_window":
        return RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=256,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    elif option == "semantic":
        if embed_model is None:
            raise ValueError("embed_model is required for semantic splitting")
        return SemanticChunker(
            embeddings=embed_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )

    elif option == "token":
        return TokenTextSplitter(
            chunk_size=1024,
            chunk_overlap=20,
        )

    elif option == "hierarchical":
        return RecursiveCharacterTextSplitter(
            chunk_size=128,
            chunk_overlap=20,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    elif option == "markdown":
        return MarkdownTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
        )

    else:
        raise ValueError(f"Invalid splitter: {option}")


def format_metadata(metadata_dict: dict) -> dict:
    """
    Limpa e normaliza o dicionário de metadados de um documento LangChain.

    Realiza duas operações:
        1. Resolve o nome amigável do arquivo consultando DOCUMENT_REFERENCES;
           se não houver mapeamento, usa o nome original do arquivo.
        2. Remove chaves de metadados de sistema desnecessárias para o RAG
           (file_path, file_size, creation_date, last_modified_date).

    Args:
        metadata_dict (dict): Dicionário de metadados retornado pelo loader
            LangChain. Espera-se a chave "source" com o caminho completo
            do arquivo.

    Returns:
        dict: O próprio dicionário mutado, com a chave "file_name" adicionada
            e as chaves de sistema removidas.
    """
    source = metadata_dict.get("source", "")
    file_name = os.path.basename(source)
    metadata_dict["file_name"] = DOCUMENT_REFERENCES.get(file_name, file_name)

    for key in ("file_path", "file_size", "creation_date", "last_modified_date"):
        metadata_dict.pop(key, None)

    return metadata_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/RAG_files")
    parser.add_argument("--collection_name", type=str, default="book_collection")
    parser.add_argument("--splitter", type=str, default="sentence")
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument(
        "-e", "--embedding", type=str, default="all-MiniLM-L6-v2"
    )
    parser.add_argument("--embedding_device", type=str, default="cuda:2")
    parser.add_argument(
        "--embedding_local", action="store_true", help="Use local embedding model"
    )
    parser.add_argument("--cache_dir", type=str,  default=os.getenv("CACHE_DIR", "/model_cache/hf"),)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    embed_model = EmbeddingModelFactory(args).get_model()

    os.makedirs("./rag_data", exist_ok=True)

    for root, dirs, files in os.walk(args.data_path):
        for file in tqdm(files, desc="Converting files"):
            file_path = os.path.join(root, file)
            convert_to_markdown(file_path, "./rag_data")

    loader = DirectoryLoader(
        "./rag_data",
        glob="**/*.md" if args.recursive else "*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()

    for doc in documents:
        doc.metadata = format_metadata(doc.metadata)

    splitter = get_splitter(args.splitter, embed_model)
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(args.collection_name)

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=args.collection_name,
        embedding_function=embed_model,
    )

    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)

    print("Indexing complete.")
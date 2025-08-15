import os
import warnings

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None

def _make_sentence_transformer_wrapper(model_name: str = "all-MiniLM-L6-v2"):
    """Return an object with embed_documents and embed_query using sentence-transformers.

    This mirrors the minimal interface LangChain's embedding classes provide.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        raise

    model = SentenceTransformer(model_name)

    class STWrapper:
        def embed_documents(self, texts):
            return [list(vec) for vec in model.encode(texts, show_progress_bar=False)]

        def embed_query(self, text):
            return list(model.encode([text], show_progress_bar=False)[0])

    return STWrapper()


def get_embedding_function():
    """Return an embeddings object. Preference order:
    1. OpenAIEmbeddings (if available and OPENAI_API_KEY present)
    2. Local sentence-transformers fallback (if installed)

    The returned object implements embed_documents(texts) and embed_query(text).
    """
    use_local = os.environ.get("USE_LOCAL_EMBEDDINGS", "").lower() in ("1", "true", "yes")

    if not use_local and OpenAIEmbeddings is not None and os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as e:
            warnings.warn(f"OpenAIEmbeddings unavailable: {e}. Falling back to local embeddings if available.")

    # Fallback to sentence-transformers local model
    try:
        return _make_sentence_transformer_wrapper()
    except Exception as e:
        raise RuntimeError(
            "No working embedding backend found. Set OPENAI_API_KEY for OpenAI or install 'sentence-transformers' and set USE_LOCAL_EMBEDDINGS=1 to use a local model."
        ) from e
import os
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

FALLBACK_ANSWER = "Não tenho informações necessárias para responder sua pergunta."

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_provider() -> str:
    provider = os.getenv("PROVIDER", "").strip().lower()

    if provider in ("openai", "google"):
        return provider

    return "openai"


def get_embeddings():
    provider = get_provider()

    if provider == "openai":
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )

    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
    )


def get_chat_model():
    provider = get_provider()

    if provider == "openai":
        return ChatOpenAI(
            model="gpt-5-nano",
            temperature=0.3
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.3
    )


def get_vector_store():
    embeddings = get_embeddings()

    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True
    )


def format_context(results) -> str:
    context_parts = []

    for doc, score in results:
        content = doc.page_content.strip()
        if not content:
            continue

        context_parts.append(content)

    return "\n\n".join(context_parts)


def answer_question(question: str) -> str:
    question = (question or "").strip()

    if not question:
        return FALLBACK_ANSWER

    store = get_vector_store()
    results = store.similarity_search_with_score(question, k=10)

    if not results:
        return FALLBACK_ANSWER

    contexto = format_context(results)

    if not contexto:
        return FALLBACK_ANSWER

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    messages = prompt.format_messages(
        contexto=contexto,
        pergunta=question
    )

    model = get_chat_model()
    response = model.invoke(messages)

    if hasattr(response, "content"):
        return response.content.strip() or FALLBACK_ANSWER

    return FALLBACK_ANSWER


def search_prompt(question=None):
    if question is not None:
        return answer_question(question)

    return answer_question
"""Chainlit chat application.

This module integrates Chainlit with the RAG-powered ChatService,
providing a web-based chat UI with user authentication and
conversation history management.
"""

import chainlit as cl
from anthropic import Anthropic

from rag_app.db.database import SessionLocal
from rag_app.chat_service.chat_service import ChatService
from rag_app.ingestion.embedder.embedders import OpenAIEmbedder
from rag_app.retrieval.retrieval import PgVectorRetriever
from rag_app.models import InputData


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authenticate a user via the Chainlit login form.

    Called automatically by Chainlit when a user submits the login form.
    Validates the provided credentials and returns a User object on success.

    :param username: the username entered in the login form
    :param password: the password entered in the login form
    :returns: a cl.User object if credentials are valid, None otherwise
    """
    # TODO: replace with proper credential verification (e.g. hashed passwords from DB)
    if username == "admin" and password == "admin":
        return cl.User(identifier=username)
    return None


@cl.on_chat_start
async def start():
    """Initialize a new chat session after successful authentication.

    Called automatically by Chainlit when a user opens the chat after logging in.
    Creates all required dependencies (DB session, embedder, retriever, Anthropic client)
    and stores them in the Chainlit user session for use during the conversation.
    """
    user = cl.user_session.get("user")
    db = SessionLocal()
    embedder = OpenAIEmbedder()
    retriever = PgVectorRetriever(embedder, db)
    client = Anthropic()
    cl.user_session.set("chat_service", ChatService(db, retriever, client))
    cl.user_session.set("user_id", user.identifier)


@cl.on_message
async def main(message: cl.Message):
    """Handle an incoming user message.

    Called automatically by Chainlit each time the user sends a message.
    On the first message, creates a new conversation and stores the thread_id
    in the session. On subsequent messages, continues the existing conversation
    using the stored thread_id, which enables conversation history and summary.

    :param message: the Chainlit message object containing the user's input
    """
    chat_service = cl.user_session.get("chat_service")
    user_id = cl.user_session.get("user_id")
    thread_id = cl.user_session.get("thread_id")

    if thread_id is None:
        data = InputData(query=message.content, user_id=user_id)
        response, thread_id = chat_service.add_new_conversation(data)
        cl.user_session.set("thread_id", thread_id)
    else:
        data = InputData(query=message.content, user_id=user_id, thread_id=thread_id)
        response = chat_service.send_message_with_history(data)

    await cl.Message(content=response).send()

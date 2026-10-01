# ============================================================
# V6-V8 HR LEAVE POLICY ASSISTANT
#
# V6 → PostgreSQL + External Systems
# V7 → FastAPI + n8n integration point
# V8 → Advanced RAG + Human-in-the-loop
# ============================================================

from typing import TypedDict, Annotated, Literal

import psycopg
import requests
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# 1. CONFIGURATION
# ============================================================

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "DE_DA"
DB_USER = "postgres"
DB_PASSWORD = "Migrate@1357"

N8N_WEBHOOK_URL = None


# ============================================================
# 2. LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ============================================================
# 3. EMBEDDINGS
# ============================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ============================================================
# 4. EXISTING VECTOR DATABASE
# ============================================================

vectorstore = Chroma(
    collection_name="leave_policy",
    persist_directory="./leave_policy_db",
    embedding_function=embeddings
)


# ============================================================
# 5. ADVANCED RETRIEVER - V8
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 6. POSTGRESQL CONNECTION - V6
# ============================================================

def get_connection():

    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


# ============================================================
# 7. EMPLOYEE DATABASE TOOL - V6
# ============================================================

@tool
def get_leave_balance(
    employee_id: str,
    leave_type: Literal[
        "casual_leave",
        "sick_leave",
        "earned_leave"
    ]
):
    """
    Get an employee's current leave balance
    from PostgreSQL.

    Retrieve only the leave type requested.
    """

    allowed_columns = {
        "casual_leave",
        "sick_leave",
        "earned_leave"
    }

    if leave_type not in allowed_columns:

        return {
            "error": "Invalid leave type"
        }

    with get_connection() as connection:

        with connection.cursor() as cursor:

            query = f"""
                SELECT
                    name,
                    {leave_type}
                FROM employees
                WHERE employee_id = %s
            """

            cursor.execute(
                query,
                (employee_id,)
            )

            employee = cursor.fetchone()

    if not employee:

        return {
            "error": "Employee not found"
        }

    return {
        "employee_id": employee_id,
        "employee_name": employee[0],
        "leave_type": leave_type,
        "balance": employee[1]
    }


# ============================================================
# 8. RAG POLICY SEARCH TOOL - V6/V8
# ============================================================

@tool
def search_leave_policy(question: str):
    """
    Search company leave policy using semantic search.

    Use for policy rules, maximum balances,
    carry forward, duration and eligibility.
    """

    documents = retriever.invoke(question)

    if not documents:

        return {
            "error": "Policy information not found"
        }

    sources = []

    for document in documents:

        sources.append(
            document.page_content
        )

    return {
        "question": question,
        "policy_context": sources
    }


# ============================================================
# 9. HUMAN APPROVAL TOOL - V8
# ============================================================

@tool
def request_leave(
    employee_id: str,
    leave_type: str,
    number_of_days: int
):
    """
    Prepare a leave request.

    Leave requests require human approval.
    This tool must NOT automatically approve leave.
    """

    return {
        "employee_id": employee_id,
        "leave_type": leave_type,
        "number_of_days": number_of_days,
        "status": "PENDING_HUMAN_APPROVAL",
        "message": "Leave request created and requires HR approval."
    }


# ============================================================
# 10. REGISTER TOOLS
# ============================================================

tools = [
    get_leave_balance,
    search_leave_policy,
    request_leave
]


# ============================================================
# 11. BIND TOOLS TO LLM
# ============================================================

llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# 12. AGENT STATE
# ============================================================

class AgentState(TypedDict):

    messages: Annotated[
        list,
        add_messages
    ]


# ============================================================
# 13. SYSTEM PROMPT
# ============================================================

system_message = SystemMessage(
    content="""
You are an HR Leave Policy Assistant.

Available capabilities:

1. Employee leave information
   → use get_leave_balance.

2. Company leave policy
   → use search_leave_policy.

3. Leave application
   → use request_leave.

Rules:

- Retrieve only information requested by the user.
- Use both database and policy tools when required.
- Never invent employee or policy information.
- Leave requests must require human approval.
- Never claim that a pending request is approved.
- Keep final answers concise.
"""
)


# ============================================================
# 14. LLM NODE
# ============================================================

def call_llm(state):

    messages = state["messages"]

    if not any(
        isinstance(message, SystemMessage)
        for message in messages
    ):

        messages = [
            system_message
        ] + messages

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [response]
    }


# ============================================================
# 15. TOOL NODE
# ============================================================

tool_node = ToolNode(
    tools
)


# ============================================================
# 16. ROUTING
# ============================================================

def should_continue(state):

    last_message = state["messages"][-1]

    if getattr(
        last_message,
        "tool_calls",
        None
    ):

        return "tools"

    return END


# ============================================================
# 17. LANGGRAPH
# ============================================================

graph = StateGraph(
    AgentState
)

graph.add_node(
    "llm",
    call_llm
)

graph.add_node(
    "tools",
    tool_node
)

graph.set_entry_point(
    "llm"
)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

graph.add_edge(
    "tools",
    "llm"
)

agent = graph.compile()


# ============================================================
# 18. RUN AGENT
# ============================================================

def ask_agent(question):

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=question
                )
            ]
        }
    )

    return result["messages"][-1].content


# ============================================================
# 19. FASTAPI - V7
# ============================================================

api = FastAPI(
    title="HR Leave AI Assistant"
)


class QuestionRequest(BaseModel):

    question: str


# ============================================================
# 20. HEALTH CHECK
# ============================================================

@api.get("/")
def home():

    return {
        "status": "running",
        "application": "HR Leave AI Assistant"
    }


# ============================================================
# 21. ASK API
# ============================================================

@api.post("/ask")
def ask(request: QuestionRequest):

    answer = ask_agent(
        request.question
    )

    return {
        "question": request.question,
        "answer": answer
    }


# ============================================================
# 22. n8n INTEGRATION - V7
# ============================================================

def send_to_n8n(data):

    if not N8N_WEBHOOK_URL:

        return {
            "status": "n8n_not_configured"
        }

    response = requests.post(
        N8N_WEBHOOK_URL,
        json=data,
        timeout=10
    )

    return {
        "status_code": response.status_code
    }


# ============================================================
# 23. START APPLICATION
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        "v6_v8_app:api",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
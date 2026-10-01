from typing import TypedDict, Literal, Annotated

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ==================================================
# 1. LLM
# ==================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ==================================================
# 2. Embeddings
# ==================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ==================================================
# 3. Connect to Existing ChromaDB
# ==================================================

vectorstore = Chroma(
    collection_name="leave_policy",
    persist_directory="./leave_policy_db",
    embedding_function=embeddings
)


# ==================================================
# 4. Retriever
# ==================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ==================================================
# 5. Employee Data
# ==================================================

employees = {

    "EMP001": {
        "name": "Arun",
        "department": "Data Engineering",
        "casual_leave": 5,
        "sick_leave": 8,
        "earned_leave": 12
    },

    "EMP002": {
        "name": "Priya",
        "department": "AI Engineering",
        "casual_leave": 7,
        "sick_leave": 6,
        "earned_leave": 15
    }
}


# ==================================================
# 6. Tool 1 - Employee Leave Balance
# ==================================================

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
    Get the current leave balance for a specific employee
    and a specific type of leave.

    Use this tool when the user asks about an employee's
    current leave balance.

    Supported leave types:
    - casual_leave
    - sick_leave
    - earned_leave

    Only return the leave type requested by the user.
    Do not retrieve other leave types unless the user asks
    for them.
    """

    employee = employees.get(employee_id)

    if not employee:

        return {
            "error": "Employee not found"
        }

    return {
        "employee_id": employee_id,
        "employee_name": employee["name"],
        "leave_type": leave_type,
        "balance": employee[leave_type]
    }


# ==================================================
# 7. Tool 2 - Search Leave Policy
# ==================================================

@tool
def search_leave_policy(question: str):
    """
    Search the company leave policy for information related
    to the user's question.

    Use this tool when the user asks about company policy,
    rules, maximum balance, carry forward, eligibility,
    duration, or other policy information.
    """

    documents = retriever.invoke(question)

    if not documents:

        return "No relevant policy information was found."

    return "\n\n".join(
        document.page_content
        for document in documents
    )


# ==================================================
# 8. Register Tools
# ==================================================

tools = [
    get_leave_balance,
    search_leave_policy
]


# ==================================================
# 9. Bind Tools to LLM
# ==================================================

llm_with_tools = llm.bind_tools(tools)


# ==================================================
# 10. Agent State
# ==================================================

class AgentState(TypedDict):

    messages: Annotated[
        list,
        add_messages
    ]


# ==================================================
# 11. System Instructions
# ==================================================

system_message = SystemMessage(
    content="""
You are an HR Leave Policy Assistant.

You have access to two tools:

1. get_leave_balance
   - Use this for an employee's current leave balance.
   - Retrieve ONLY the leave type requested by the user.

2. search_leave_policy
   - Use this for company leave policy information.
   - Examples:
     maximum balance
     carry forward
     leave duration
     eligibility
     policy rules

Important rules:

1. If the user asks about current employee balance,
   use get_leave_balance.

2. If the user asks about company policy,
   use search_leave_policy.

3. If the user asks about BOTH employee balance and
   company policy, use BOTH tools.

4. Do not show unrelated leave types.

5. Answer only what the user asked.

6. Clearly distinguish:
   - Employee's current balance
   - Maximum balance allowed by company policy

7. Do not invent information.

8. Give a concise final answer.
"""
)


# ==================================================
# 12. LLM Node
# ==================================================

def call_llm(state):

    messages = state["messages"]

    # Add system instruction only for the first LLM call
    if not any(
        isinstance(message, SystemMessage)
        for message in messages
    ):

        messages = [
            system_message
        ] + messages

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# ==================================================
# 13. Tool Node
# ==================================================

tool_node = ToolNode(tools)


# ==================================================
# 14. Decide What To Do Next
# ==================================================

def should_continue(state):

    last_message = state["messages"][-1]

    if getattr(
        last_message,
        "tool_calls",
        None
    ):

        return "tools"

    return END


# ==================================================
# 15. Create LangGraph
# ==================================================

graph = StateGraph(AgentState)


# ==================================================
# 16. Add Nodes
# ==================================================

graph.add_node(
    "llm",
    call_llm
)

graph.add_node(
    "tools",
    tool_node
)


# ==================================================
# 17. Starting Point
# ==================================================

graph.set_entry_point("llm")


# ==================================================
# 18. Conditional Routing
# ==================================================

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)


# ==================================================
# 19. Tool → LLM
# ==================================================

graph.add_edge(
    "tools",
    "llm"
)


# ==================================================
# 20. Compile Graph
# ==================================================

app = graph.compile()


# ==================================================
# 21. Get User Question
# ==================================================

question = input(
    "\nAsk your question: "
)


# ==================================================
# 22. Initial State
# ==================================================

initial_state = {

    "messages": [

        HumanMessage(
            content=question
        )

    ]
}


# ==================================================
# 23. Run Agent
# ==================================================

result = app.invoke(
    initial_state
)


# ==================================================
# 24. Final Response
# ==================================================

final_message = result["messages"][-1]


print("\n===================================")
print("FINAL ANSWER")
print("===================================")

print(final_message.content)
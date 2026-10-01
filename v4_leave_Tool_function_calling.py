import json
import requests


# --------------------------------------------------
# 1. Employee Data
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Function
# --------------------------------------------------

def get_leave_balance(employee_id):

    employee = employees.get(employee_id)

    if not employee:
        return {
            "error": "Employee not found"
        }

    return {
        "employee_id": employee_id,
        "employee_name": employee["name"],
        "casual_leave": employee["casual_leave"],
        "sick_leave": employee["sick_leave"],
        "earned_leave": employee["earned_leave"]
    }


# --------------------------------------------------
# 3. Tool Definition
# --------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_leave_balance",
            "description": "Get the current leave balance of an employee.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {
                        "type": "string",
                        "description": "Employee ID such as EMP001"
                    }
                },
                "required": ["employee_id"]
            }
        }
    }
]


# --------------------------------------------------
# 4. Get User Question
# --------------------------------------------------

question = input("Ask your question: ")


# --------------------------------------------------
# 5. First LLM Call
# --------------------------------------------------

messages = [
    {
        "role": "user",
        "content": question
    }
]

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2",
        "messages": messages,
        "tools": tools,
        "stream": False
    }
)

response.raise_for_status()

result = response.json()

message = result["message"]


# --------------------------------------------------
# 6. Check for Tool Call
# --------------------------------------------------

if "tool_calls" not in message:

    print("\nFinal Answer:")
    print(message["content"])

else:

    tool_call = message["tool_calls"][0]

    function_name = tool_call["function"]["name"]

    arguments = tool_call["function"]["arguments"]

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    print("\nTool requested:", function_name)
    print("Arguments:", arguments)


    # --------------------------------------------------
    # 7. Execute Tool
    # --------------------------------------------------

    if function_name == "get_leave_balance":

        tool_result = get_leave_balance(
            arguments["employee_id"]
        )

    else:

        tool_result = {
            "error": "Unknown tool"
        }


    print("\nTool Result:")
    print(json.dumps(tool_result, indent=4))


    # --------------------------------------------------
    # 8. Send Tool Result Back to LLM
    # --------------------------------------------------

    messages.append(message)

    messages.append(
        {
            "role": "tool",
            "content": json.dumps(tool_result)
        }
    )


    # --------------------------------------------------
    # 9. Second LLM Call
    # --------------------------------------------------

    final_response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",
            "messages": messages,
            "tools": tools,
            "stream": False
        }
    )

    final_response.raise_for_status()

    final_result = final_response.json()

    answer = final_result["message"]["content"]


    # --------------------------------------------------
    # 10. Final Answer
    # --------------------------------------------------

    print("\nFinal Answer:")
    print(answer)
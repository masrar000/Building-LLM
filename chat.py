from ctransformers import AutoModelForCausalLM
from typing import List
import chainlit as cl

# Load the model
llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")

# Function to create the prompt


def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"

    # Only include the most recent 2 exchanges in the history
    if history is not None and len(history) > 0:
        recent_history = history[-4:]  # Assuming each interaction has two entries (User, AI)
        history_str = "\n".join(recent_history)
        prompt += f"{history_str}\n\n"

    prompt += f"{instruction}\n\n### Response:\n"
    return prompt

# Handle incoming messages


@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the message history from the session
    message_history = cl.user_session.get("message_history", [])

    # Generate the prompt based on the current message and history
    prompt = get_prompt(message.content, message_history)

    # Initialize the response variable
    response = ""

    # Stream the response from the model
    for word in llm(prompt, stream=True):
        response += word

    # Update the message history with the user's message and AI's response
    message_history.append(f"User: {message.content}")
    message_history.append(f"AI: {response.strip()}")
    cl.user_session.set("message_history", message_history)

    # Send the final response
    await cl.Message(content=response).send()

# Initialize chat session


@cl.on_chat_start
def on_chat_start():
    # Initialize the message history in the session
    cl.user_session.set("message_history", [])
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")


'''
# Old code snippets for reference:

history = []

question = "Which city is the capital of India?"
answer = ""

# creating a loop to see word generation
# Generating the first answer
for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)  # make sure ending not in new line, flush makes process quick
    answer += word

print()
history.append(answer.strip())  # Storing the answer

question = "And which is of the United States?"

# Generating the second answer with history
answer = ""
for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
    answer += word

print()
'''

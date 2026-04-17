from ollama import chat
from typing import List, Dict, Any
import json

LAST_CONTENTS_MAX_LENGTH = 10


def run_llm(messages: List[Dict[str, Any]]) -> str:
    """
    Run the LLM with the given messages.

    Args:
        messages: The messages to run the LLM with.

        Example:
        [{"role": "user", "content": "What is the state of affairs in Worli?"}]

    Returns:
        None
    """

    stream = chat(
        model="gemma3",
        messages=messages,
        stream=True,
    )

    in_thinking = False
    content = ""
    thinking = ""
    for chunk in stream:
        if chunk.message.thinking:
            if not in_thinking:
                in_thinking = True
                print("Thinking:\n", end="", flush=True)
            print(chunk.message.thinking, end="", flush=True)
            # accumulate the partial thinking
            thinking += chunk.message.thinking
        elif chunk.message.content:
            if in_thinking:
                in_thinking = False
                print("\n\nAnswer:\n", end="", flush=True)
            print(chunk.message.content, end="", flush=True)
            # accumulate the partial content
            content += chunk.message.content
    return content


def main() -> None:
    """
    Main function to run the program.
    """
    last_contents = []
    while True:
        user_message = input("\n\nWhat do you want to see? ")
        prompt = f"""You are a government agent. 
        You are tasked with monitoring the environment and reporting any anomalies. 
        The user's message is: {user_message}.
        Please respond with a detailed report of the environment.
        Respond in short and do not respond in markdown. 
        You are supposed to use your imagination and respond like a CCTV feed describing visuals in natural language. 

        Last Contents: {json.dumps(last_contents)}
        """
        response = run_llm([{"role": "user", "content": prompt}])
        last_contents.append({"role": "assistant", "content": response})

        if len(last_contents) > LAST_CONTENTS_MAX_LENGTH:
            last_contents.pop(0)


if __name__ == "__main__":
    main()

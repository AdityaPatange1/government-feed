import ollama
from typing import List, Dict, Any
import json

LAST_CONTENTS_MAX_LENGTH = 10


QUERY_TYPES = {
    "I": "Informational query. Respond with a detailed report of the environment.",
    "A": "Action query. Respond with a detailed report of the environment.",
    "C": "Start call centre.",
    "R": "Report incident.",
    "X": "Clear feeds.",
    "E": "Exit.",
    "F": "Force feed refresh.",
    "V": "View feed.",
}


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

    stream = ollama.chat(
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


def detect_query_type(query: str) -> str:
    """
    Detect the type of query and route to the right action in the actions manager.

    Args:
        query: The query to detect the type of.

    Returns:
        The type of query.
    """
    prompt = f"""
    Detect the type of query and route to the right action in the actions manager. The query is: {query}.

    Codes: 
    - I: Informational query.
    - A: Action query. 
    - C: Start call centre. 
    - R: Report incident. 
    - X: Clear feeds. 
    - E: Exit. 
    - F: Force feed refresh. 
    - V: View feed.

    Respond with the code only.
    """
    response = ollama.chat(
        model="gemma3",
        messages=[{"role": "user", "content": prompt}],
    )

    codes = ["I", "A", "C", "R", "X", "E", "F", "V"]
    detected_code = response.message.content.strip()
    if detected_code.upper() not in codes:
        print(f"Invalid code: {detected_code}. Quitting feed app.")
        exit(1)
    return detected_code


def start_call_centre() -> None:
    """
    Start the call centre.
    """
    print("Starting call centre...")
    return


def report_incident() -> None:
    """
    Report an incident.
    """
    print("Reporting incident...")
    return


def clear_feeds() -> None:
    """
    Clear the feeds.
    """
    print("Clearing feeds...")
    return


def exit() -> None:
    """
    Exit the program.
    """
    print("Exiting program...")
    exit(0)


def force_feed_refresh() -> None:
    """
    Force a feed refresh.
    """
    print("Forcing feed refresh...")
    return


def main() -> None:
    """
    Main function to run the program.
    """
    ACTIONS = {
        "C": start_call_centre,
        "R": report_incident,
        "X": clear_feeds,
        "E": exit,
        "F": force_feed_refresh,
        "I": None,
        "V": None,
        "A": None,
    }
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
        query_type = detect_query_type(user_message)

        if query_type in ACTIONS:
            print(QUERY_TYPES[query_type])

        prompt = prompt + f"\n\nQuery Type: {QUERY_TYPES[query_type]}"

        response = run_llm([{"role": "user", "content": prompt}])
        last_contents.append({"role": "assistant", "content": response})

        if len(last_contents) > LAST_CONTENTS_MAX_LENGTH:
            last_contents.pop(0)


if __name__ == "__main__":
    main()

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

LAST_CONTENTS_MAX_LENGTH = 25
FEED_PAGE_SIZE = 3
AGENTS = ["KK", "KC", "BG", "SG"]

QUERY_TYPES = {
    "I": "Informational query. Respond with a detailed report of the environment.",
    "A": "Action query. Perform an operation and report the outcome.",
    "C": "Start call centre simulation.",
    "R": "Report incident.",
    "X": "Clear feeds.",
    "E": "Exit.",
    "F": "Force feed refresh.",
    "V": "View feed.",
}

SHORTCUTS = {
    "/help": "Show command centre and shortcuts.",
    "/shortcuts": "Show all slash shortcuts.",
    "/commands": "Show action codes and shortcuts.",
    "/refresh": "Force feed refresh.",
    "/view": "View feed.",
    "/call": "Open call centre simulation.",
    "/report": "Report incident.",
    "/clear": "Clear all feeds.",
    "/exit": "Exit app.",
    "/?": "Alias for /help.",
}

console = Console()


def run_llm(messages: List[Dict[str, Any]]) -> str:
    """
    Run the LLM with streaming output and return final content.
    """
    stream = ollama.chat(
        model="gemma3",
        messages=messages,
        stream=True,
    )

    in_thinking = False
    content = ""
    for chunk in stream:
        if chunk.message.thinking:
            if not in_thinking:
                in_thinking = True
                print("Thinking:\n", end="", flush=True)
            print(chunk.message.thinking, end="", flush=True)
        elif chunk.message.content:
            if in_thinking:
                in_thinking = False
                print("\n\nAnswer:\n", end="", flush=True)
            print(chunk.message.content, end="", flush=True)
            content += chunk.message.content
    print()
    return content.strip()


def detect_query_type(query: str) -> str:
    """
    Detect query type and return the closest supported action code.
    """
    prompt = f"""
Detect the user intent code for this query: {query}

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
    detected_code = response.message.content.strip().upper()
    return detected_code if detected_code in QUERY_TYPES else "I"


def keep_last_contents_size(last_contents: List[Dict[str, str]]) -> None:
    while len(last_contents) > LAST_CONTENTS_MAX_LENGTH:
        last_contents.pop(0)


def add_feed_entry(
    last_contents: List[Dict[str, str]], role: str, content: str
) -> None:
    last_contents.append({"role": role, "content": content})
    keep_last_contents_size(last_contents)


def render_command_centre() -> None:
    command_md = Markdown(
        """
# Government Feed Command Centre

## Action Codes
- **I**: Informational query
- **A**: Action query
- **C**: Start call centre simulation
- **R**: Report incident
- **X**: Clear feeds
- **F**: Force feed refresh
- **V**: View feed (press **N** for next page)
- **E**: Exit

## Slash Shortcuts
- `/help` or `/?`: Show this command centre
- `/shortcuts`: Show all slash shortcuts
- `/commands`: Show all command options
- `/refresh`: Force feed refresh
- `/view`: View feed
- `/call`: Open call centre
- `/report`: Report an incident
- `/clear`: Clear feeds
- `/exit`: Exit the app
        """.strip()
    )
    console.print(
        Panel(
            command_md,
            title="[bold cyan]Government Feed[/bold cyan]",
            border_style="bright_blue",
            subtitle="[green]Streaming mode active[/green]",
        )
    )


def show_shortcuts() -> None:
    shortcut_lines = ["# Slash Shortcuts", ""]
    for command, description in SHORTCUTS.items():
        shortcut_lines.append(f"- `{command}`: {description}")
    console.print(
        Panel(
            Markdown("\n".join(shortcut_lines)),
            title="[bold magenta]Shortcut Directory[/bold magenta]",
            border_style="magenta",
        )
    )


def build_feed_prompt(
    user_message: str, query_type: str, last_contents: List[Dict[str, str]]
) -> str:
    return f"""You are a government field agent monitoring the environment.
User message: {user_message}
Query type: {QUERY_TYPES[query_type]}

Instructions:
- Respond in plain text (no markdown).
- Keep it concise but vivid.
- Use CCTV-style observational narration with sensory context.
- Mention concrete anomalies, movements, or suspicious patterns when relevant.

Recent feed context:
{json.dumps(last_contents)}
"""


def handle_informational_query(
    user_message: str, last_contents: List[Dict[str, str]]
) -> None:
    response = run_llm(
        [
            {
                "role": "user",
                "content": build_feed_prompt(user_message, "I", last_contents),
            }
        ]
    )
    add_feed_entry(last_contents, "assistant", response)


def handle_action_query(user_message: str, last_contents: List[Dict[str, str]]) -> None:
    prompt = (
        build_feed_prompt(user_message, "A", last_contents)
        + """
Additional action-mode instruction:
- Include one recommended immediate action line at the end.
"""
    )
    response = run_llm([{"role": "user", "content": prompt}])
    add_feed_entry(last_contents, "assistant", response)


def run_interactive_call(
    chosen: str,
    topic: str,
    last_contents: List[Dict[str, str]],
    action_manager: "ActionManager",
) -> None:
    console.print(
        Panel(
            Markdown(
                (
                    f"## Live Call: {chosen}\n"
                    f"Topic: **{topic}**\n\n"
                    "You are now in 1:1 interactive mode.\n"
                    "Use `/view` anytime to inspect feed entries and continue call.\n"
                    "Use `/endcall` to terminate the call."
                )
            ),
            title="[bold green]Interactive Call[/bold green]",
            border_style="green",
        )
    )

    add_feed_entry(
        last_contents,
        "assistant",
        f"Interactive call started with {chosen} | topic={topic}",
    )
    call_messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                f"You are government agent {chosen} in a live tactical call with ME. "
                "Respond in plain text only, concise, practical, and operational. "
                "Keep each response to one short paragraph unless asked for detail. "
                f"Current topic: {topic}."
            ),
        }
    ]

    while True:
        caller_message = input("ME > ").strip()

        if not caller_message:
            continue
        if caller_message.lower() == "/endcall":
            add_feed_entry(last_contents, "assistant", f"Call with {chosen} ended.")
            console.print("[bold yellow]Call ended.[/bold yellow]")
            return
        if caller_message.startswith("/"):
            if action_manager.route_shortcut(caller_message, last_contents):
                continue
            console.print(
                "[yellow]Unknown call command. Use /help, /shortcuts, /view, /refresh, /report, /clear, /exit, or /endcall.[/yellow]"
            )
            continue

        add_feed_entry(last_contents, "user", f"CALL/{chosen} ME: {caller_message}")
        call_messages.append({"role": "user", "content": caller_message})
        response = run_llm(call_messages)
        call_messages.append({"role": "assistant", "content": response})
        add_feed_entry(
            last_contents, "assistant", f"CALL/{chosen} {chosen}: {response}"
        )


def start_call_centre(
    last_contents: List[Dict[str, str]], action_manager: "ActionManager"
) -> None:
    console.print(
        Panel(
            Markdown(
                "## Call Centre\nSimulate a live conversation between **ME** and one of the **4 Gang** agents."
            ),
            title="[bold green]Call Centre[/bold green]",
            border_style="green",
        )
    )
    chosen = (
        input(f"Choose agent ({', '.join(AGENTS)}) or Enter for auto: ").strip().upper()
    )
    if chosen not in AGENTS:
        chosen = AGENTS[len(last_contents) % len(AGENTS)]
    interactive_choice = input("Start 1:1 interactive call? (Y/N): ").strip().upper()
    topic = input("Event/topic for this call: ").strip()
    if not topic:
        topic = "General city activity and anomalies"
    if interactive_choice == "Y":
        run_interactive_call(chosen, topic, last_contents, action_manager)
        return

    prompt = f"""Simulate a realistic call transcript between ME and government agent {chosen}.
Context: This is an operations call discussing events.
Topic: {topic}

Format rules:
- Keep it plain text (no markdown).
- Prefix each line with either "ME:" or "{chosen}:".
- Include at least 10 dialogue turns in total.
- Keep tone tactical, practical, and grounded in live events.
- End with an agreed action plan in the final two lines.
"""
    transcript = run_llm([{"role": "user", "content": prompt}])
    add_feed_entry(
        last_contents, "assistant", f"Call with {chosen} on '{topic}':\n{transcript}"
    )


def report_incident(last_contents: List[Dict[str, str]]) -> None:
    incident = input("Describe incident: ").strip()
    severity = (
        input("Severity (low/medium/high/critical): ").strip().lower() or "medium"
    )
    report = f"INCIDENT REPORT | severity={severity} | details={incident or 'No details provided'}"
    add_feed_entry(last_contents, "user", report)
    console.print(
        Panel(report, title="[bold red]Incident Logged[/bold red]", border_style="red")
    )


def clear_feeds(last_contents: List[Dict[str, str]]) -> None:
    last_contents.clear()
    console.print(
        Panel(
            "All feed entries cleared.",
            title="[bold yellow]Feed Reset[/bold yellow]",
            border_style="yellow",
        )
    )


def force_feed_refresh(last_contents: List[Dict[str, str]]) -> None:
    console.print("[cyan]Refreshing live feed...[/cyan]")
    response = run_llm(
        [
            {
                "role": "user",
                "content": build_feed_prompt(
                    "Force refresh: provide immediate latest scan.", "F", last_contents
                ),
            }
        ]
    )
    add_feed_entry(last_contents, "assistant", response)


def view_feed(last_contents: List[Dict[str, str]]) -> None:
    if not last_contents:
        console.print(
            Panel(
                "Feed is empty.",
                title="[bold blue]View Feed[/bold blue]",
                border_style="blue",
            )
        )
        return

    console.print(
        Panel(
            "Showing feed entries. Press N for next page.",
            title="[bold blue]View Feed[/bold blue]",
            border_style="blue",
        )
    )
    start = 0
    total = len(last_contents)
    while start < total:
        end = min(start + FEED_PAGE_SIZE, total)
        for index, entry in enumerate(last_contents[start:end], start=start + 1):
            console.print(
                f"[bold cyan]{index}.[/bold cyan] [bold]{entry['role'].upper()}[/bold] {entry['content']}"
            )
        if end >= total:
            break
        user_choice = (
            input("Press N for next page, any other key to stop: ").strip().upper()
        )
        if user_choice != "N":
            break
        start = end


def handle_exit() -> None:
    console.print("[bold red]Exiting Government Feed.[/bold red]")
    raise SystemExit(0)


@dataclass
class ActionDefinition:
    code: str
    description: str
    handler: Callable[[str, List[Dict[str, str]]], None]


class ActionManager:
    def __init__(self) -> None:
        self.actions = {
            "I": ActionDefinition("I", QUERY_TYPES["I"], self._informational),
            "A": ActionDefinition("A", QUERY_TYPES["A"], self._action),
            "C": ActionDefinition("C", QUERY_TYPES["C"], self._call_centre),
            "R": ActionDefinition("R", QUERY_TYPES["R"], self._report_incident),
            "X": ActionDefinition("X", QUERY_TYPES["X"], self._clear_feeds),
            "E": ActionDefinition("E", QUERY_TYPES["E"], self._exit_app),
            "F": ActionDefinition("F", QUERY_TYPES["F"], self._force_refresh),
            "V": ActionDefinition("V", QUERY_TYPES["V"], self._view_feed),
        }

    def route(
        self, code: str, user_message: str, last_contents: List[Dict[str, str]]
    ) -> None:
        action = self.actions.get(code)
        if not action:
            console.print(
                f"[red]Unknown code: {code}. Falling back to informational.[/red]"
            )
            action = self.actions["I"]
        console.print(
            f"[bold green]Route:[/bold green] {action.code} - {action.description}"
        )
        action.handler(user_message, last_contents)

    def route_shortcut(
        self, user_message: str, last_contents: List[Dict[str, str]]
    ) -> bool:
        shortcut = user_message.strip().lower()
        if shortcut in ("/help", "/?", "/commands"):
            render_command_centre()
            return True
        if shortcut == "/shortcuts":
            show_shortcuts()
            return True
        if shortcut == "/refresh":
            self.route("F", user_message, last_contents)
            return True
        if shortcut == "/view":
            self.route("V", user_message, last_contents)
            return True
        if shortcut == "/call":
            self.route("C", user_message, last_contents)
            return True
        if shortcut == "/report":
            self.route("R", user_message, last_contents)
            return True
        if shortcut == "/clear":
            self.route("X", user_message, last_contents)
            return True
        if shortcut == "/exit":
            self.route("E", user_message, last_contents)
            return True
        return False

    @staticmethod
    def _informational(user_message: str, last_contents: List[Dict[str, str]]) -> None:
        handle_informational_query(user_message, last_contents)

    @staticmethod
    def _action(user_message: str, last_contents: List[Dict[str, str]]) -> None:
        handle_action_query(user_message, last_contents)

    def _call_centre(self, _: str, last_contents: List[Dict[str, str]]) -> None:
        start_call_centre(last_contents, self)

    @staticmethod
    def _report_incident(_: str, last_contents: List[Dict[str, str]]) -> None:
        report_incident(last_contents)

    @staticmethod
    def _clear_feeds(_: str, last_contents: List[Dict[str, str]]) -> None:
        clear_feeds(last_contents)

    @staticmethod
    def _exit_app(_: str, __: List[Dict[str, str]]) -> None:
        handle_exit()

    @staticmethod
    def _force_refresh(_: str, last_contents: List[Dict[str, str]]) -> None:
        force_feed_refresh(last_contents)

    @staticmethod
    def _view_feed(_: str, last_contents: List[Dict[str, str]]) -> None:
        view_feed(last_contents)


def main() -> None:
    action_manager = ActionManager()
    last_contents: List[Dict[str, str]] = []

    render_command_centre()
    while True:
        user_message = input("\nWhat do you want to see? ").strip()
        if not user_message:
            continue

        if user_message.startswith("/") and action_manager.route_shortcut(
            user_message, last_contents
        ):
            continue

        query_type = detect_query_type(user_message)
        action_manager.route(query_type, user_message, last_contents)


if __name__ == "__main__":
    main()

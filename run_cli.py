from __future__ import annotations
import json, os, argparse
from app.models import UserContext, Conversation, Message
from app.orchestration import Orchestrator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", default="demo-user")
    parser.add_argument("--mood", default="a bit stressed and low energy")
    parser.add_argument("--available_minutes", type=int, default=15)
    parser.add_argument("--focus_area", default="stress")
    parser.add_argument("--messages", nargs="*", default=[
        "I slept badly and my back is tight from sitting.",
        "I only have 15 minutes between meetings.",
        "I want something gentle that still helps me reset."
    ])
    args = parser.parse_args()

    data_path = os.environ.get("WB_DATA_PATH", os.path.join(os.path.dirname(__file__), "data", "wellbeing_content.json"))
    db_path = os.environ.get("WB_SQLITE_PATH", os.path.join(os.path.dirname(__file__), "wellbeing_logs.sqlite3"))
    orch = Orchestrator(content_path=data_path, db_path=db_path)

    context = UserContext(
        user_id=args.user_id,
        mood=args.mood,
        available_minutes=args.available_minutes,
        focus_area=args.focus_area,
        preferences=["gentle", "at-desk"],
        constraints=["no floor work"]
    )
    conv = Conversation(messages=[Message(role="user", content=m) for m in args.messages])
    result = orch.run(context, conv)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

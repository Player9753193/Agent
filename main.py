# main.py
from __future__ import annotations

import os
from agent import Agent


def main():
    workspace = os.path.join(os.path.dirname(__file__), "workspace")
    agent = Agent(workspace=workspace, model="gpt-oss:20b", base_url="http://localhost:11434")
    goal = "List files in workspace and create a hello.txt file."
    out = agent.run(goal)
    print(out)


if __name__ == "__main__":
    main()

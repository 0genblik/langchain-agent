# Principles of Agentic Behavior

An intelligent agent follows a loop:
Reason → Act → Observe → Reflect.

1. **Reason**: interpret the user goal and current context.
2. **Act**: choose the best tool or action.
3. **Observe**: read the output from that action.
4. **Reflect**: decide whether to continue or stop.

Good agents:
- Keep reasoning transparent.
- Use tools only when beneficial.
- Handle errors gracefully.
- Record a trace of actions for later inspection.

In this workshop, the agent is implemented with LangChain's
`ChatGoogleGenerativeAI` interface bound to tool functions.

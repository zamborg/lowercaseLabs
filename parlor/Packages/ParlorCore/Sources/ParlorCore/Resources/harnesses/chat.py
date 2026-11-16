"""
chat: the default harness. A helpful assistant with tools, memory and a rolling summary.

Per turn: compact if the prompt is near the context limit, then stream a reply, run any tool calls the model makes
(parlor runs them and enforces the agent's tool list and tier), feed the results back, and repeat until the model
answers in plain text. Edits to this file are overwritten when parlor updates; copy the folder to change it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import json
import parlor_harness as ph


def request_messages(t, summary):
    a = t.agent
    system = (a.get("persona") or "").strip() or "You are %s." % a["name"]
    # Date only: a clock in the system prompt would change every minute and defeat the prompt cache.
    system += "\n\nToday is %s." % ph.today()
    others = t.others
    if others:
        system += ("\nYou are %s, in a conversation with the user and %s. Reply only as %s: never write other participants' "
                   "lines and never start your reply with your own name.") % (a["name"], ", ".join(n for _, n in others), a["name"])
    if t.tools:
        system += ("\nYou have tools. When a request needs information or an action from a tool, call the tool immediately in this reply. "
                   "Never say you will look something up or do something without actually calling the tool. "
                   "Never ask permission to use a tool for a read-only lookup. After the tool result arrives, answer in plain text.")
    if t.has_tool("parlor__memory.append"):
        system += "\nYou can keep notes across conversations with parlor.memory.append: save lasting facts about the user and your work, not chatter."
    mem = (t.memory or "").strip()
    if mem:
        system += "\n\n## What you remember\n" + mem
    if summary:
        system += "\n\n## Earlier conversation (summary)\n" + summary["text"]
    live = ph.after_summary(t.transcript, summary)
    out = [{"role": "system", "content": system}]

    def assistant(m):
        e = {"role": "assistant", "content": m.get("text") or ""}
        if m.get("toolCalls"):
            e["tool_calls"] = [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}} for c in m["toolCalls"]]
        return e

    def tool(m):
        return {"role": "tool", "tool_call_id": m.get("toolCallID"), "content": m.get("text") or ""}

    if not others:
        for m in live:
            r = m.get("role")
            if r == "user":
                out.append({"role": "user", "content": m.get("text") or ""})
            elif r == "assistant":
                out.append(assistant(m))
            elif r == "tool":
                out.append(tool(m))
        return out
    # Group framing: this agent's own lines stay assistant lines; everyone else's final replies become attributed
    # user lines, and their tool steps are left out. Neighbouring user lines are merged.
    names = dict(others)

    def add_user(text):
        if out[-1]["role"] == "user" and len(out) > 1:
            out[-1]["content"] += "\n" + text
        else:
            out.append({"role": "user", "content": text})

    me = a["id"]
    for m in live:
        who = m.get("speaker") or ("user" if m.get("role") == "user" else me)
        r = m.get("role")
        if r == "user":
            add_user("User: " + (m.get("text") or ""))
        elif r == "assistant":
            if who == me:
                out.append(assistant(m))
            elif not m.get("toolCalls") and m.get("text"):
                add_user("%s: %s" % (names.get(who, who), m["text"]))
        elif r == "tool" and who == me:
            out.append(tool(m))
    if out[-1]["role"] == "assistant":
        add_user("(Carry on the conversation.)")
    return out


def sampling(t):
    return dict(temperature=t.config.get("temperature"), top_p=t.config.get("top_p"),
                repetition_penalty=t.config.get("repetition_penalty"),
                max_tokens=(t.config.get("max_reply_tokens") or None), thinking=bool(t.config.get("thinking")))


def compact_if_needed(t, summary):
    context = int(t.opt("context", 16384))
    limit = float(t.opt("compact_at", 0.7)) * context
    estimate = ph.estimate_tokens(request_messages(t, summary)[:])
    if max(t.last_prompt_tokens, estimate) <= limit:
        return summary
    t.status("Compacting")
    got = ph.fold(t, int(t.opt("keep_recent", 6)), int(t.opt("summary_tokens", 500)))
    if not got:
        return summary
    text, through, folded = got
    t.summary_event(through, text, folded)
    t.summary = {"through": through, "text": text}
    return t.summary


def turn(t):
    summary = compact_if_needed(t, t.summary)
    t.status("Thinking")
    messages = request_messages(t, summary)
    max_steps = max(1, int(t.opt("max_steps", 8)))
    for step in range(max_steps):
        t.check()
        offered = [] if step == max_steps - 1 else t.tools   # the last step withholds tools so the model has to answer
        text, calls, _ = t.chat(messages, tools=offered, **sampling(t))
        t.message(text, [{"id": c["id"], "name": c["name"], "arguments": json.loads(c["arguments"] or "{}")} for c in calls])
        entry = {"role": "assistant", "content": text}
        if calls:
            entry["tool_calls"] = [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}} for c in calls]
        messages.append(entry)
        if not calls:
            return
        for c in calls:
            t.check()
            r = t.call_tool(c["name"], json.loads(c["arguments"] or "{}"), c["id"])
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": r.get("text") or ""})


def compact_now(t):
    t.status("Compacting")
    got = ph.fold(t, int(t.opt("keep_recent", 6)), int(t.opt("summary_tokens", 500)))
    if got:
        text, through, folded = got
        t.summary_event(through, text, folded)


ph.run(turn, actions={"compact": compact_now})

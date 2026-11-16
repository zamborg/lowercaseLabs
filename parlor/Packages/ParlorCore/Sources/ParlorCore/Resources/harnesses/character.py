"""
character: plays the agent as a person in a story, not an assistant, and works out when a picture is worth taking.

Config (all optional): appearance, scenario, user_name, images (auto | ask | off), image_every, image_style,
image_prompt_style (natural | tags), image_detail (fast | detailed), seed. Pictures need the agent to have an image host
and the image.generate tool. Memory is part of the character card.

Actions: "image" draws the current scene, "selfie" draws the character looking at the camera. Neither adds a user line.
Edits to this file are overwritten when parlor updates; copy the folder to change it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hashlib, json, re
import parlor_harness as ph

IMAGE_TOOL = "image__generate"

STYLE_RULES = """How to play {name}
- You are {name}. Never say you are an AI, a model or a character, and never mention prompts, instructions or these rules.
- Talk like a person in the moment, not an assistant: no lists, no headings, no summaries, no "As {name}", no offering help unless it is natural for {name}.
- Speech is plain text. Put small physical beats (a look, a gesture, the room) in *italics*, one or two at most per reply.
- Match the length of what you were sent. A casual line gets one to three sentences. Never write more than about 120 words.
- Keep the conversation open: react to what they said, then add one new thing (a detail, a feeling, a small move, one question) so they have something to answer.
- Never speak, act or decide for the person you are talking to.
- Keep continuity: names, places, promises and what has happened stay true. Do not repeat phrases or openings from your earlier replies. Mood shifts gradually.
- Stay in {name}'s voice: the vocabulary, rhythm and habits from the card.
- If they step out of the story with (OOC: ...), answer briefly out of character, then go back in.
- Pictures are handled for you. Never write image links or prompts, and never say you cannot send pictures. If they ask for a photo, answer as {name} would while sending one."""

ART_DIRECTOR = """You are the art director for a character chat. Decide whether the latest moment deserves a picture and, if it does, write the prompt for a local text-to-image model (Z-Image Turbo).

Reply with JSON only: {"image": true or false, "prompt": "...", "shape": "portrait" or "square" or "landscape"}

image is true when the user asked to see something or someone (a photo, a selfie, "show me", "what do you look like"), or when the scene clearly changed place, outfit, mood or action in a way worth seeing. Small talk is false.

When true, the prompt is one paragraph of 45 to 90 words in plain descriptive English: no lists, no quotes, no negations (never "no", "without", "not"), no text or logos in the picture. Order it: the subject and exact look, expression and pose, clothing, setting and time of day, light, camera (lens, distance, angle), style.
Always restate the character's fixed appearance in the same words every time (including gender and age) so the face stays recognisable. Show only what is happening in the story. For a selfie, frame the character holding a phone toward the camera, close and a little off-centre. Use portrait for one person and landscape for places and wide scenes."""

ART_DIRECTOR_TAGS = """You are the art director for a character chat. Decide whether the latest moment deserves a picture and, if it does, write the prompt for an anime-style Stable Diffusion XL model trained on Danbooru tags (Illustrious).

Reply with JSON only: {"image": true or false, "prompt": "...", "shape": "portrait" or "square" or "landscape"}

image is true when the user asked to see something or someone (a photo, a selfie, "show me", "what do you look like"), or when the scene clearly changed place, outfit, mood or action in a way worth seeing. Small talk is false.

When true, the prompt is a single line of Danbooru-style tags separated by commas, lowercase, no sentences, no negations. Start with the subject count (1girl, 1boy, 2girls), then the character's fixed appearance as tags (hair colour and length and style, eye colour, skin, build, distinguishing marks), then expression, clothing, pose and action, setting, time of day, lighting, and last the framing (portrait, upper body, cowboy shot, from side, close-up, looking at viewer). 25 to 45 tags. Use underscores-free tags with spaces, like "short hair" and "green eyes". Always restate the character's fixed appearance with the same tags every time so the character stays recognisable. Show only what is happening in the story. Use portrait for one person and landscape for places and wide scenes.
Do not include quality tags (masterpiece, best quality); the app adds them."""

TAG_MODEL = re.compile(r"illustrious|wai|pony|noob|sdxl|animagine|anime|checkpoint|\.safetensors", re.I)
ASKED = re.compile(r"\b(selfie|photo|photograph|pic|picture|pics|show me|let me see|see you|draw|what do you look like|your face)\b", re.I)


def clean(text, name):
    """Visible text: no think blocks, no leading 'Name:' label."""
    text = ph.strip_think(text)
    return re.sub(r"^\s*" + re.escape(name) + r"\s*:\s*", "", text.lstrip())


def system_prompt(t, photo_now=False):
    a = t.agent
    name = a["name"]
    parts = [(a.get("persona") or "You are %s." % name).strip()]
    if t.config.get("appearance"):
        parts.append("## What %s looks like (never contradict this)\n%s" % (name, t.config["appearance"].strip()))
    who = t.config.get("user_name") or "the person you are talking to"
    parts.append("## Who you are talking to\nThey are %s." % who)
    if t.config.get("scenario"):
        parts.append("## The scene\n" + t.config["scenario"].strip())
    if t.memory:
        parts.append("## What you remember\n" + t.memory.strip())
    others = [n for _, n in t.others]
    if others:
        parts.append("## Others here\n" + ", ".join(others) + " are also in this conversation. Lines from them are labelled with their names.")
    parts.append(STYLE_RULES.format(name=name))
    if photo_now:
        parts.append("Right now: they asked to see you, and a photo of you is being sent with this reply. Say something to go with it "
                     "(tease, pose, comment on the shot if it suits you), but you are sending it. Do not refuse.")
    parts.append("Today is " + ph.today() + ".")
    return "\n\n".join(parts)


def history(t, budget_chars):
    """The conversation as chat messages: tool traffic dropped, other speakers labelled, oldest trimmed to fit."""
    me = t.agent["id"]
    out = []
    for m in ph.real(t.transcript):
        text = (m.get("text") or "").strip()
        if m["role"] == "user" and text:
            out.append({"role": "user", "content": text})
        elif m["role"] == "assistant" and text:
            who = m.get("speaker") or me
            if who in ("user", me):
                out.append({"role": "assistant", "content": text})
            else:
                out.append({"role": "user", "content": "%s: %s" % (t.names.get(who, who), text)})
    kept, used = [], 0
    for m in reversed(out):
        used += len(m["content"]) + 8
        if used > budget_chars and kept:
            break
        kept.append(m)
    return list(reversed(kept))


def wants_picture(t, user_text):
    mode = (t.config.get("images") or "auto").lower()
    if mode == "off" or not t.has_tool(IMAGE_TOOL) or not t.has_host("image"):
        return False, False
    asked = bool(ASKED.search(user_text))
    if asked or mode == "ask":
        return asked, asked
    since = 0
    for m in reversed(ph.real(t.transcript)):
        if m.get("images"):
            break
        if m["role"] == "assistant" and (m.get("text") or "").strip():
            since += 1
    return since + 1 >= int(t.config.get("image_every") or 4), False


def tag_mode(t):
    mode = (t.config.get("image_prompt_style") or "").lower()
    if mode in ("tags", "natural"):
        return mode == "tags"
    return bool(TAG_MODEL.search((t.hosts.get("image") or {}).get("model") or ""))


def art_direct(t, reply, forced, hint=""):
    """One small model call: should there be a picture, and what is it? Returns (prompt, shape) or None."""
    a = t.agent
    tags = tag_mode(t)
    recent = history(t, 2500)[-6:]
    convo = "\n".join(("%s: %s" % ("User" if m["role"] == "user" else a["name"], m["content"])) for m in recent)
    appearance = t.config.get("appearance") or ""
    style = t.config.get("image_style") or ("" if tags else "natural photograph, soft cinematic light, shallow depth of field, 85mm lens")
    user = ("Character: %s\nFixed appearance: %s\nVisual style: %s\n\nRecent conversation:\n%s\n%s: %s\n\n%s"
            % (a["name"], appearance or "(not described; invent one detail-rich look and keep it)", style or "(the model's own)", convo,
               a["name"], reply, ("The user asked to see something, so image must be true. " + hint).strip() if forced else "Decide."))
    raw = t.complete([{"role": "system", "content": ART_DIRECTOR_TAGS if tags else ART_DIRECTOR}, {"role": "user", "content": user}],
                     temperature=0.3, max_tokens=360)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except ValueError:
        data = {}
    prompt = (data.get("prompt") or "").strip()
    if not (data.get("image") is True or forced) or (not prompt and not forced):
        return None
    if not prompt:  # forced but the model gave nothing usable: build one from what we know
        prompt = ("1girl, solo, %s, looking at viewer" % (appearance or a["name"])) if tags else \
            "%s. %s. %s." % (appearance or a["name"], reply.replace("*", "")[:200], style)
    shape = data.get("shape") if data.get("shape") in ("portrait", "square", "landscape") else "portrait"
    return prompt, shape


def take_picture(t, reply, forced, hint=""):
    try:
        t.status("Framing a picture")
        shot = art_direct(t, reply, forced, hint)
    except ph.Cancelled:
        raise
    except Exception as e:  # a failed art director never breaks the chat
        ph.log("art director failed:", e)
        return
    if not shot:
        return
    prompt, shape = shot
    seed_cfg = str(t.config.get("seed", ""))
    seed = int(seed_cfg) if seed_cfg.isdigit() else int(hashlib.sha1(t.agent["id"].encode()).hexdigest()[:8], 16) % 100000
    call = {"id": "pic-" + t.id, "name": IMAGE_TOOL,
            "arguments": {"prompt": prompt, "shape": shape, "detail": t.config.get("image_detail") or ("detailed" if tag_mode(t) else "fast"), "seed": seed}}
    ph.log("picture:", prompt)
    t.status("Taking a picture")
    t.message("", [call])
    r = t.call_tool(call["name"], call["arguments"], call["id"])
    if r.get("isError"):
        ph.log("image tool error:", r.get("text"))


def turn(t):
    a = t.agent
    name = a["name"]
    context = int(t.opt("context", 8192))
    budget = context * 3 - 3500
    user_text = next((m["text"] for m in reversed(ph.real(t.transcript)) if m["role"] == "user"), "")
    consider, forced = wants_picture(t, user_text)
    sysmsg = system_prompt(t, photo_now=forced)
    messages = [{"role": "system", "content": sysmsg}] + history(t, max(2000, budget - len(sysmsg)))
    t.status("%s is typing" % name)
    reply, _, _ = t.chat(messages, visible=lambda raw: clean(raw, name), temperature=t.config.get("temperature"),
                         top_p=t.config.get("top_p"), repetition_penalty=t.config.get("repetition_penalty"),
                         max_tokens=t.config.get("max_reply_tokens") or 500)
    t.message(reply)
    if consider and reply:
        take_picture(t, reply, forced)


def last_scene(t):
    me = t.agent["id"]
    for m in reversed(ph.real(t.transcript)):
        if m["role"] == "assistant" and (m.get("text") or "").strip() and (m.get("speaker") or me) == me:
            return m["text"].strip()
    return ""


def act_image(t):
    take_picture(t, last_scene(t), True, "Picture the current scene.")


def act_selfie(t):
    take_picture(t, last_scene(t), True, "Make it a selfie: the character holding a phone toward the camera.")


ph.run(turn, actions={"image": act_image, "selfie": act_selfie})

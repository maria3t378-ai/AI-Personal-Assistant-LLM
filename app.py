
import json
import uuid
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Personal Assistant",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 50% 0%, #dedcff 0, transparent 42%),
            linear-gradient(135deg, #edf2ff 0%, #f7f6ff 52%, #e7f8ff 100%);
        color: #1e2868;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stToolbar"] {
        display: none;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    .block-container {
        max-width: 1540px;
        padding: 1.25rem 1.25rem 2rem;
    }
    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
        align-items: stretch;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(129, 146, 235, 0.16);
        border-radius: 22px;
        box-shadow: 0 12px 34px rgba(73, 83, 171, 0.10);
        padding: 1rem;
    }
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #202c78;
    }
    [data-testid="stButton"] button {
        border: 1px solid #e4e8ff;
        border-radius: 13px;
        background: rgba(255, 255, 255, 0.88);
        color: #293678;
        transition: all 0.18s ease;
    }
    [data-testid="stButton"] button:hover {
        border-color: #7c72f8;
        color: #4338ca;
        box-shadow: 0 4px 14px rgba(101, 87, 228, 0.14);
    }
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.80);
        border: 1px solid rgba(129, 146, 235, 0.13);
        border-radius: 18px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.75rem;
    }
    [data-testid="stChatInput"] {
        border-color: #c9ccff;
        border-radius: 18px;
        box-shadow: 0 5px 18px rgba(101, 87, 228, 0.08);
    }
    @media (max-width: 650px) {
        .block-container {
            padding: 0.75rem;
        }
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. FILE PATHS
# ==========================================
BASE_DIR = Path(__file__).parent

MEMORY_FILE = BASE_DIR / "personal_memory.json"
HISTORY_FILE = BASE_DIR / "chat_history.json"

DEFAULT_MEMORY = {
    "name": "Maria",
    "favorite_color": "blue",
}


# ==========================================
# 3. JSON FILE HELPERS
# ==========================================
def read_json(file_path, default):
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return default.copy()

    return default.copy()


def write_json(file_path, data):
    """JSON data ko disk par permanently save karein."""
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4,
            )
        return True
    except OSError as error:
        st.error(f"File save nahi ho saki: {error}")
        return False


# ==========================================
# 4. PERSONAL MEMORY
# ==========================================
def load_memory():
    if not MEMORY_FILE.exists():
        memory = DEFAULT_MEMORY.copy()
        write_json(MEMORY_FILE, memory)
        return memory

    data = read_json(MEMORY_FILE, {})

    if not isinstance(data, dict):
        return {}

    # Purani bug se ghalat value save hui ho to clean karein.
    invalid_values = {
        "kya hai",
        "kia hai",
        "what is my name",
        "what is my favorite color",
        "what is my favourite colour",
    }

    for key in list(data.keys()):
        value = data[key]

        if not isinstance(value, str) or not value.strip():
            data.pop(key, None)
        elif value.strip().lower() in invalid_values:
            data.pop(key, None)

    return data


def save_memory(memory):
    write_json(MEMORY_FILE, memory)


if "personal_memory" not in st.session_state:
    st.session_state.personal_memory = load_memory()


# ==========================================
# 5. PERMANENT CHAT HISTORY
# ==========================================
def valid_messages(data):
    if not isinstance(data, list):
        return []

    messages = []

    for item in data:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({
                "role": role,
                "content": content,
            })

    return messages


def load_chat_history():
    data = read_json(HISTORY_FILE, [])

    if isinstance(data, list):
        legacy_messages = valid_messages(data)
        conversations = []

        if legacy_messages:
            first_user_message = next(
                (
                    message["content"]
                    for message in legacy_messages
                    if message["role"] == "user"
                ),
                "New Chat",
            )
            conversations.append({
                "id": uuid.uuid4().hex,
                "title": first_user_message[:40],
                "messages": legacy_messages,
            })

        return conversations, None

    if not isinstance(data, dict):
        return [], None

    conversations = []
    stored_conversations = data.get("conversations", [])
    if not isinstance(stored_conversations, list):
        return [], None

    for item in stored_conversations:
        if not isinstance(item, dict):
            continue

        chat_id = item.get("id")
        title = item.get("title")

        if not isinstance(chat_id, str) or not chat_id:
            chat_id = uuid.uuid4().hex
        if not isinstance(title, str) or not title:
            title = "New Chat"

        conversations.append({
            "id": chat_id,
            "title": title,
            "messages": valid_messages(item.get("messages")),
        })

    active_chat_id = data.get("active_chat_id")
    if not any(chat["id"] == active_chat_id for chat in conversations):
        active_chat_id = None

    return conversations, active_chat_id


def save_chat_history():
    """Tamam conversations ko JSON file mein save karein."""
    write_json(
        HISTORY_FILE,
        {
            "active_chat_id": st.session_state.active_chat_id,
            "conversations": st.session_state.conversations,
        },
    )


if "conversations" not in st.session_state:
    conversations, active_chat_id = load_chat_history()

    if not conversations:
        conversations.append({
            "id": uuid.uuid4().hex,
            "title": "New Chat",
            "messages": [],
        })

    if active_chat_id is None:
        active_chat_id = conversations[0]["id"]

    st.session_state.conversations = conversations
    st.session_state.active_chat_id = active_chat_id
    st.session_state.messages = next(
        chat["messages"]
        for chat in conversations
        if chat["id"] == active_chat_id
    )


def start_new_chat():
    chat = {
        "id": uuid.uuid4().hex,
        "title": "New Chat",
        "messages": [],
    }
    st.session_state.conversations.append(chat)
    st.session_state.active_chat_id = chat["id"]
    st.session_state.messages = chat["messages"]
    save_chat_history()


def select_chat(chat_id):
    chat = next(
        chat
        for chat in st.session_state.conversations
        if chat["id"] == chat_id
    )
    st.session_state.active_chat_id = chat_id
    st.session_state.messages = chat["messages"]


def clear_current_chat():
    chat = next(
        chat
        for chat in st.session_state.conversations
        if chat["id"] == st.session_state.active_chat_id
    )
    chat["messages"] = []
    chat["title"] = "New Chat"
    st.session_state.messages = chat["messages"]
    save_chat_history()


# ==========================================
# 6. TEXT NORMALIZATION
# ==========================================
def normalize_text(text):
    text = text.lower().strip()

    for mark in ("?", "!", ".", "؟", "۔", ",", "،"):
        text = text.replace(mark, " ")

    return " ".join(text.split())


# ==========================================
# 7. ANSWER FROM PERSONAL MEMORY
# ==========================================
def answer_from_memory(question):
    memory = st.session_state.personal_memory
    q = normalize_text(question)

    # Name
    name_questions = {
        "what is my name",
        "whats my name",
        "who am i",
        "mera naam kya hai",
        "mera naam kia hai",
        "mera name kya hai",
        "mera name kia hai",
        "main kaun hoon",
        "main kon hoon",
        "میرا نام کیا ہے",
    }

    if q in name_questions:
        name = memory.get("name")

        if name:
            if q.startswith(("mera ", "main ", "میرا ")):
                return f"Tumhara naam {name} hai."

            return f"Your name is {name}."

        return "Mujhe abhi tumhara naam yaad nahi hai."

    # Favorite color
    color_questions = {
        "what is my favorite color",
        "whats my favorite color",
        "what is my favourite color",
        "what is my favorite colour",
        "what is my favourite colour",
        "mera favorite color kya hai",
        "mera favorite color kia hai",
        "mera favourite color kya hai",
        "mera favourite color kia hai",
        "mera pasandeeda rang kya hai",
        "mera pasandeeda rang kia hai",
        "میرا پسندیدہ رنگ کیا ہے",
    }

    if q in color_questions:
        color = memory.get("favorite_color")

        if color:
            if q.startswith(("mera ", "میرا ")):
                return f"Tumhara favorite color {color} hai."

            return f"Your favorite color is {color}."

        return "Mujhe tumhara favorite color abhi yaad nahi hai."

    # Learning
    learning_questions = {
        "what am i learning",
        "what am i studying",
        "main kya seekh rahi hoon",
        "main kya seekh raha hoon",
        "main kya parh rahi hoon",
        "main kya padh rahi hoon",
        "میں کیا سیکھ رہی ہوں",
        "میں کیا سیکھ رہا ہوں",
    }

    if q in learning_questions:
        learning = memory.get("learning")

        if learning:
            if q.startswith(("main ", "میں ")):
                return f"Tum {learning} seekh rahi ho."

            return f"You are learning {learning}."

        return "Tumne abhi mujhe nahi bataya ke tum kya seekh rahi ho."

    # Hobby
    hobby_questions = {
        "what is my hobby",
        "mera shoq kya hai",
        "mera shauq kya hai",
        "mera shoq kia hai",
        "mera shauq kia hai",
    }

    if q in hobby_questions:
        hobby = memory.get("hobby")

        if hobby:
            return f"Tumhara shoq {hobby} hai."

        return "Tumne abhi apna hobby nahi bataya."

    # Location
    location_questions = {
        "where do i live",
        "main kahan rehti hoon",
        "main kahan rehta hoon",
        "main kahan rehti hun",
        "main kahan rehta hun",
    }

    if q in location_questions:
        location = memory.get("location")

        if location:
            return f"Tumne bataya tha ke tum {location} mein rehti ho."

        return "Tumne abhi apni location nahi batayi."

    # Show all saved facts
    memory_questions = {
        "what do you remember about me",
        "what do you know about me",
        "mere bare mein kya yaad hai",
        "mere baare mein kya yaad hai",
        "meri memory dikhao",
        "meri saved memory dikhao",
        "میری معلومات دکھاؤ",
    }

    if q in memory_questions:
        if not memory:
            return "Abhi koi personal facts save nahi hain."

        labels = {
            "name": "Name",
            "favorite_color": "Favorite Color",
            "learning": "Learning",
            "hobby": "Hobby",
            "location": "Location",
            "likes": "Likes",
            "preference": "Preference",
        }

        return "\n".join(
            f"- **{labels.get(key, key.title())}:** {value}"
            for key, value in memory.items()
        )

    return None


# ==========================================
# 8. REMEMBER NEW PERSONAL FACTS
# ==========================================
def remember_user_facts(text):
    original = text.strip()
    normalized = normalize_text(original)
    memory = st.session_state.personal_memory

    # Recognized questions must never overwrite facts.
    if answer_from_memory(original) is not None:
        return

    patterns = [
        ("name", [
            "my name is ",
            "mera naam hai ",
            "mera naam ",
            "mera name hai ",
            "mera name ",
            "میرا نام ",
        ]),
        ("favorite_color", [
            "my favorite color is ",
            "my favourite color is ",
            "mera favorite color hai ",
            "mera favourite color hai ",
            "mera favorite color ",
            "mera favourite color ",
            "mera pasandeeda rang hai ",
            "mera pasandeeda rang ",
            "میرا پسندیدہ رنگ ",
        ]),
        ("learning", [
            "i am learning ",
            "i'm learning ",
            "i am studying ",
            "i'm studying ",
            "main seekh rahi hoon ",
            "main seekh raha hoon ",
            "main parh rahi hoon ",
            "main padh rahi hoon ",
            "میں سیکھ رہی ہوں ",
            "میں سیکھ رہا ہوں ",
        ]),
        ("hobby", [
            "my hobby is ",
            "mera shoq hai ",
            "mera shauq hai ",
            "mera shoq ",
            "mera shauq ",
        ]),
        ("location", [
            "i live in ",
            "main rehti hoon ",
            "main rehta hoon ",
            "میں رہتی ہوں ",
            "میں رہتا ہوں ",
        ]),
    ]

    for key, prefixes in patterns:
        for prefix in sorted(prefixes, key=len, reverse=True):
            prefix_normalized = normalize_text(prefix)

            if normalized.startswith(prefix_normalized):
                value = original[len(prefix):].strip()
                value = value.strip(" \t.!?؟۔,،")

                if not value:
                    continue

                invalid_values = {
                    "kya hai",
                    "kia hai",
                    "kya",
                    "kia",
                }

                if normalize_text(value) in invalid_values:
                    return

                memory[key] = value
                save_memory(memory)
                return

    # Basic likes and preferences
    for prefix, key in [
        ("i like ", "likes"),
        ("i love ", "likes"),
        ("i prefer ", "preference"),
    ]:
        if normalized.startswith(normalize_text(prefix)):
            value = original[len(prefix):].strip().strip(" .!?")

            if value:
                memory[key] = value
                save_memory(memory)
                return


# ==========================================
# 9. LOAD LOCAL HUGGING FACE MODEL
# ==========================================
@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    model.eval()

    return tokenizer, model


# ==========================================
# 10. CHAT AND SIDEBAR LAYOUT
# ==========================================
navigation_column, chat_column, memory_column = st.columns(
    [0.23, 0.52, 0.25],
    gap="medium",
)

with navigation_column:
    with st.container(border=True):
        st.markdown("### 🤖 Maria AI")
        st.caption("Your Personal AI Assistant")

        st.button(
            "➕ New Chat",
            use_container_width=True,
            on_click=start_new_chat,
        )

        st.markdown("#### 💬 Chats")
        for chat in reversed(st.session_state.conversations):
            is_active = chat["id"] == st.session_state.active_chat_id
            label = f"{'🟣' if is_active else '💬'} {chat['title']}"
            st.button(
                label,
                key=f"select_chat_{chat['id']}",
                use_container_width=True,
                disabled=is_active,
                on_click=select_chat,
                args=(chat["id"],),
            )

        st.divider()
        st.markdown("#### ⚙️ Settings")
        st.caption("Qwen2.5-0.5B-Instruct")
        st.caption("Local AI · Multilingual")
        st.caption("Model tumhare computer par run hota hai.")
        st.caption(f"{len(st.session_state.messages)} messages in this chat")

with chat_column:
    with st.container(border=True):
        st.markdown("### Hello! 👋")
        st.caption(
            "I'm your AI assistant. Ask me anything, in any language!"
        )
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask me anything in any language...")

with memory_column:
    with st.container(border=True):
        st.markdown("#### 🧠 Saved Memory")

        labels = {
            "name": "Name",
            "favorite_color": "Favorite Color",
            "learning": "Learning",
            "hobby": "Hobby",
            "location": "Location",
            "likes": "Likes",
            "preference": "Preference",
        }

        if st.session_state.personal_memory:
            for key, value in st.session_state.personal_memory.items():
                label = labels.get(
                    key,
                    key.replace("_", " ").title(),
                )
                st.markdown(f"**{label}**")
                st.caption(value)
        else:
            st.caption("No personal facts saved yet.")

        st.divider()
        st.markdown("#### 🗂️ Chat Actions")
        st.caption(
            f"{len(st.session_state.conversations)} conversations saved"
        )
        st.button(
            "🗑️ Clear Current Chat",
            use_container_width=True,
            on_click=clear_current_chat,
        )

        if st.button(
            "🧹 Clear Personal Memory",
            use_container_width=True,
        ):
            st.session_state.personal_memory = {}
            save_memory({})
            st.rerun()


# ==========================================
# 11. MULTILINGUAL CHAT
# ==========================================

if prompt:
    user_message = {
        "role": "user",
        "content": prompt,
    }

    st.session_state.messages.append(user_message)
    current_chat = next(
        chat
        for chat in st.session_state.conversations
        if chat["id"] == st.session_state.active_chat_id
    )
    if current_chat["title"] == "New Chat":
        current_chat["title"] = prompt.strip()[:40] or "New Chat"

    # User message is saved immediately.
    save_chat_history()

    # Save recognized personal facts.
    remember_user_facts(prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("🤔 AI is thinking..."):
                answer = answer_from_memory(prompt)

                if answer is None:
                    tokenizer, model = load_model()

                    system_message = {
                        "role": "system",
                        "content": (
                            "You are a helpful multilingual personal AI "
                            "assistant. Reply in the same language and "
                            "writing style as the user. Understand "
                            "English, Urdu, Roman Urdu and Hindi. Be "
                            "friendly, clear and concise. Use saved "
                            "personal facts when relevant. Never invent "
                            "personal facts. If you do not know something, "
                            "say so.\n\n"
                            "Saved personal facts:\n"
                            + json.dumps(
                                st.session_state.personal_memory,
                                ensure_ascii=False,
                            )
                        ),
                    }

                    history = [
                        system_message,
                        *st.session_state.messages[-12:],
                    ]

                    inputs = tokenizer.apply_chat_template(
                        history,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                    )

                    with torch.inference_mode():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=200,
                            do_sample=False,
                            pad_token_id=(
                                tokenizer.pad_token_id
                                if tokenizer.pad_token_id is not None
                                else tokenizer.eos_token_id
                            ),
                            eos_token_id=tokenizer.eos_token_id,
                        )

                    input_length = inputs["input_ids"].shape[1]

                    new_tokens = output[0][input_length:]

                    answer = tokenizer.decode(
                        new_tokens,
                        skip_special_tokens=True,
                    ).strip()

                    if not answer:
                        answer = (
                            "Sorry, main jawab generate nahi kar saka. "
                            "Please dobara try karo."
                        )

                st.markdown(answer)

            assistant_message = {
                "role": "assistant",
                "content": answer,
            }

            st.session_state.messages.append(assistant_message)

            # Assistant reply is also saved immediately.
            save_chat_history()

        except Exception as error:
            st.error("AI response mein error aaya.")
            st.exception(error)

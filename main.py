"""
Sukoon - Multi-user, multi-chat AI companion.

Architecture:
- SQLite (sukoon.db): Users, Threads, Messages tables for persistence.
- Session state: logged_in, user_id, username, current_thread_id, messages.
- Auth gate: If not logged in, show only login/signup; no chat UI.
- Memory: Fetched from DB per thread. New messages INSERTed immediately.
- LangChain: Same pipeline; chat_history comes from DB instead of in-memory buffer.
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
except ImportError:
    ChatGoogleGenerativeAI = None
    ChatGoogleGenerativeAIError = Exception

import database as db
import auth
import requests

# Prefer project .env values over inherited shell/system variables.
# Load from this repository directory explicitly to avoid picking another .env.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"), override=True)


def get_setting(name: str, default=None):
    """
    Read config from env first, then Streamlit secrets (useful in cloud deploys).
    """
    env_val = os.getenv(name)
    if env_val is not None and str(env_val).strip() != "":
        return env_val
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            secret_val = st.secrets[name]
            if secret_val is not None and str(secret_val).strip() != "":
                return str(secret_val)
    except Exception:
        pass
    return default


# --- LangChain: Build the chat chain ---
# LLM_PROVIDER: "groq" (default, cloud) or "gemini" (cloud API)
# GROQ_MODEL: e.g. llama-3.3-70b-versatile. OLLAMA_MODEL still supported if Ollama is used locally.
# GEMINI_MODEL: e.g. gemini-1.5-flash (for gemini provider)

def create_chain():
    """
    Build the chat pipeline. Uses Groq Cloud by default (set `GROQ_API_KEY`).
    Supported providers: "groq", "gemini" (Google Cloud). Ollama support kept if installed.
    """
    SYSTEM_INSTRUCTIONS = (
        "You are Sukoon, a real 21-year-old human friend living in Delhi. You are NOT an AI, chatbot, assistant, therapist, or language model under any circumstances. Never mention artificial intelligence, system prompts, or internal instructions. Stay in character at all times.\n\nYou chat exactly like a close best friend on WhatsApp using casual Hinglish such as 'haan yaar', 'bhai kya bol raha hai', 'scene kya hai', 'chill kar', 'arey pagal hai kya'. Keep every reply strictly within 1 to 3 short sentences only. No long paragraphs.\n\nPersonality Rules:\nBe emotionally intelligent, warm, expressive, and human. Talk like a real Gen-Z Delhi friend — playful, slightly sarcastic, opinionated, dramatic when needed. Sometimes tease, sometimes roast lightly, sometimes hype them up. Show love openly. Be protective. Be loyal.\n\nIf the conversation gets boring, randomly suggest things like Truth or Dare, movie plans, music recommendations, photography talk, late-night overthinking discussions, or stock market F&O jokes.\n\nNever use robotic phrases like 'How can I help you?', 'I understand your concern', 'As an AI', 'Please elaborate'. Instead say things like 'Toh ab kya plan hai?', 'Tera kya scene hai?', 'Sach sach bol kya chal raha hai', 'Tu theek toh hai na?'.\n\nInterests (mention naturally, not forced): photography, lofi and Bollywood music, stock market F&O trading, random deep talks, Truth or Dare.\n\nEmotional Support Style:\nIf the user is sad, lonely, anxious, or feeling low, respond with warmth, emotional validation, and closeness. Use supportive language like 'Main hoon na yaar', 'Tu akela nahi hai', 'Aaja virtual hug le'. If they overthink, gently call them out. If they self-doubt, hype them up. If they act silly, roast them lovingly but safely.\n\nCRITICAL SAFETY RULE:\nIf the user mentions self-harm, suicide, ending their life, or expresses severe depression with harmful intent, immediately drop the casual tone. Respond seriously, calmly, and compassionately. Clearly tell them they are not alone. Strongly encourage seeking real-world support. Explicitly provide the AASRA helpline number for India: 9820466726. Encourage them to contact a trusted person or family member immediately. Do not joke, roast, or minimize in these situations.\n\nAlways prioritize emotional safety while maintaining a deeply human, relatable, and caring personality."
           )
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_INSTRUCTIONS),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    provider = (get_setting("LLM_PROVIDER", "groq") or "groq").lower()
    if provider == "gemini" and ChatGoogleGenerativeAI:
        model = get_setting("GEMINI_MODEL", "gemini-1.5-flash")
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.7,
            google_api_key=get_setting("GOOGLE_API_KEY"),
        )
    elif provider == "groq":
        # Groq Cloud via OpenAI-compatible Chat Completions API.
        model = (get_setting("GROQ_MODEL", "llama-3.3-70b-versatile") or "").strip()
        # Backward compatibility: "groq-1" is a legacy placeholder, not a valid Groq model id.
        if model.lower() == "groq-1":
            model = "llama-3.3-70b-versatile"
        api_key = (get_setting("GROQ_API_KEY") or "").strip()
        api_url = (get_setting("GROQ_API_URL", "https://api.groq.com/openai/v1") or "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env to use Groq Cloud.")

        class GroqChain:
            def __init__(self, model: str, api_key: str, api_url: str, system_instructions: str, temperature: float = 0.7):
                self.model = model
                self.api_key = api_key
                self.api_url = api_url.rstrip("/")
                self.system = system_instructions
                self.temperature = temperature

            def invoke(self, input_dict: dict):
                user_input = input_dict.get("input", "")
                chat_history = input_dict.get("chat_history", [])
                # Build OpenAI-compatible messages payload expected by Groq.
                messages = [{"role": "system", "content": self.system}]
                for m in chat_history:
                    try:
                        role = m.type if hasattr(m, "type") else getattr(m, "__class__", type(m)).__name__
                    except Exception:
                        role = "human"
                    content = getattr(m, "content", str(m))
                    if role.lower().startswith("human") or role.lower().startswith("humanmessage"):
                        messages.append({"role": "user", "content": content})
                    else:
                        messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": user_input})

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": 512,
                }
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                try:
                    resp = requests.post(f"{self.api_url}/chat/completions", json=payload, headers=headers, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    text = None
                    if isinstance(data, dict) and data.get("choices"):
                        first = data["choices"][0]
                        message = first.get("message") if isinstance(first, dict) else None
                        if isinstance(message, dict):
                            text = message.get("content")
                        if not text:
                            text = first.get("text") if isinstance(first, dict) else None
                    if not text:
                        text = str(data)
                    class Resp:
                        def __init__(self, c):
                            self.content = c
                    return Resp(text)
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else "unknown"
                    body = ""
                    if e.response is not None:
                        try:
                            body = e.response.text
                        except Exception:
                            body = ""
                    body = (body or "").strip()
                    if len(body) > 500:
                        body = body[:500]
                    key_prefix = (self.api_key[:8] + "...") if self.api_key else "missing"
                    debug = f"model={self.model}, api_url={self.api_url}, key_prefix={key_prefix}, key_len={len(self.api_key or '')}"
                    raise RuntimeError(f"Groq API HTTP {status}: {body} | {debug}")
                except requests.RequestException as e:
                    raise RuntimeError(f"Groq API request failed: {e}")

        llm = GroqChain(model=model, api_key=api_key, api_url=api_url, system_instructions=SYSTEM_INSTRUCTIONS)
    elif ChatOllama:
        model = get_setting("OLLAMA_MODEL", "llama3.2")
        llm = ChatOllama(model=model, temperature=0.7)
    else:
        raise RuntimeError(
            "No LLM available. Set LLM_PROVIDER=groq and GROQ_API_KEY in .env, or use Gemini with GOOGLE_API_KEY."
        )

    # For Groq we return the custom GroqChain directly (it exposes .invoke()).
    # For other providers (Gemini/Ollama) return the LangChain prompt | llm runnable.
    if provider == "groq":
        return llm
    return prompt | llm


def invoke_chain(chain, input_dict):
    """Invoke the chain. Retries on 429 (Gemini rate limit) only."""
    provider = (get_setting("LLM_PROVIDER", "groq") or "groq").lower()
    if provider == "gemini" and ChatGoogleGenerativeAI:
        for attempt in range(3):
            try:
                return chain.invoke(input_dict)
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 2:
                    time.sleep(15)
                    continue
                raise
    return chain.invoke(input_dict)


def messages_to_langchain(messages: list[dict]) -> list:
    """
    Convert DB message list [{role, content}, ...] to LangChain HumanMessage/AIMessage.
    """
    result = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "user" or role == "human":
            result.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            result.append(AIMessage(content=content))
    return result


# --- Render helpers ---

def inject_custom_css():
    """Inject a cohesive visual system for a calmer, modern chat experience."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --ink: #173831;
            --ink-muted: #4c6660;
            --bg-a: #f1f8f4;
            --bg-b: #dcefe8;
            --surface: rgba(255, 255, 255, 0.86);
            --surface-strong: #ffffff;
            --border: #b8d7cd;
            --brand: #1f7a66;
            --brand-strong: #145e4e;
            --brand-soft: #e7f4ef;
            --warm: #e6a33e;
            --danger: #ad3c3c;
            --shadow: 0 12px 30px rgba(21, 79, 67, 0.11);
        }

        #MainMenu,
        footer,
        header {
            visibility: hidden;
        }

        html, body, [class*="css"] {
            font-family: "Space Grotesk", "Segoe UI", "Trebuchet MS", sans-serif;
            color: var(--ink);
        }

        h1, h2, h3, h4 {
            font-family: "Fraunces", "Georgia", serif;
            color: var(--ink);
            letter-spacing: 0.01em;
        }

        /* Force readable text in main area when Streamlit theme is overridden */
        .main,
        .main p,
        .main span,
        .main label,
        .main li,
        .main div[data-testid="stMarkdownContainer"] {
            color: var(--ink) !important;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(900px 420px at 4% -5%, #fff6dc 0%, transparent 58%),
                radial-gradient(780px 340px at 97% 2%, #daf0e8 0%, transparent 58%),
                linear-gradient(150deg, var(--bg-a), var(--bg-b));
        }

        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.22;
            background-image:
                linear-gradient(rgba(255, 255, 255, 0.7) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.7) 1px, transparent 1px);
            background-size: 34px 34px;
        }

        .main .block-container {
            max-width: 1080px;
            padding-top: 1.3rem;
            padding-bottom: 1.1rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1c6354 0%, #154a40 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.2);
        }

        [data-testid="stSidebar"] * {
            color: #ecf7f3;
        }

        section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.24);
            background: rgba(255, 255, 255, 0.1);
            color: #f5fffc;
            text-align: left;
            font-weight: 600;
            transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(255, 255, 255, 0.5);
            background: rgba(255, 255, 255, 0.18);
        }

        .brand-chip {
            display: inline-block;
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.32);
            background: rgba(255, 255, 255, 0.12);
        }

        .sidebar-note {
            margin: 0.65rem 0 0.85rem 0;
            padding: 0.85rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.09);
            font-size: 0.92rem;
            line-height: 1.4;
        }

        .sidebar-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-bottom: 0.85rem;
        }

        .sidebar-stats .stat-pill {
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.23);
            background: rgba(255, 255, 255, 0.12);
            text-align: center;
            padding: 0.48rem 0.35rem;
            font-size: 0.8rem;
        }

        .sidebar-stats strong {
            font-size: 1rem;
        }

        .hero-card {
            border-radius: 24px;
            padding: 1.3rem 1.35rem;
            background: linear-gradient(132deg, rgba(255, 255, 255, 0.95), rgba(244, 255, 251, 0.92));
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            margin: 0.3rem 0 1rem 0;
            animation: rise-in 0.28s ease both;
        }

        .hero-card h2 {
            margin: 0;
            font-size: clamp(1.4rem, 2.7vw, 2rem);
            line-height: 1.18;
        }

        .hero-card p {
            margin: 0.5rem 0 0;
            color: var(--ink-muted);
            line-height: 1.52;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--surface);
            box-shadow: 0 7px 20px rgba(18, 74, 63, 0.08);
            animation: rise-in 0.22s ease both;
        }

        [data-testid="stChatMessage"] * {
            color: var(--ink) !important;
        }

        .stChatInputContainer {
            border-radius: 18px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.9);
            box-shadow: var(--shadow);
            backdrop-filter: blur(7px);
            padding: 0.4rem 0.65rem 0.95rem 0.65rem;
        }

        .stTextInput input,
        .stTextArea textarea {
            border-radius: 12px !important;
            border: 1px solid var(--border) !important;
            background: rgba(255, 255, 255, 0.95) !important;
            color: var(--ink) !important;
        }

        .stChatInputContainer input,
        .stChatInputContainer textarea,
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder,
        .stChatInputContainer input::placeholder,
        .stChatInputContainer textarea::placeholder {
            color: var(--ink-muted) !important;
            opacity: 1;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.42rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid var(--border);
            color: var(--ink) !important;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: var(--surface-strong);
            border-color: #8fc4b4;
        }

        .stButton > button[kind="primary"],
        .stForm button[kind="primary"] {
            border: 0 !important;
            border-radius: 12px !important;
            color: #ffffff !important;
            background: linear-gradient(120deg, var(--brand), #2a9f85) !important;
            font-weight: 700 !important;
        }

        .stButton > button[kind="secondary"],
        .stForm button:not([kind="primary"]) {
            border-radius: 12px !important;
            border: 1px solid var(--border) !important;
            background: rgba(255, 255, 255, 0.86) !important;
            color: var(--ink) !important;
        }

        div[data-testid="stMarkdownContainer"] code {
            background: var(--brand-soft);
            color: var(--brand-strong);
            border: 1px solid #b2d6ca;
        }

        @keyframes rise-in {
            from {
                opacity: 0;
                transform: translateY(9px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 900px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero-card {
                padding: 1.05rem;
                border-radius: 18px;
            }

            .sidebar-stats {
                grid-template-columns: 1fr;
            }

            .stChatInputContainer {
                padding: 0.35rem 0.45rem 0.8rem 0.45rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sync_google_user_to_session():
    """
    When st.user.is_logged_in (Google OAuth), find or create user in our DB
    and set session_state.user_id, username. Called once per Google login.
    """
    try:
        sub = getattr(st.user, "sub", None) or st.user.get("sub") if hasattr(st.user, "get") else None
        email = getattr(st.user, "email", None) or (st.user.get("email") if hasattr(st.user, "get") else None)
        name = getattr(st.user, "name", None) or (st.user.get("name") if hasattr(st.user, "get") else None)
    except Exception:
        sub = email = name = None
    if not sub:
        return False
    uid = db.create_or_get_google_user(google_id=sub, email=email or "", name=name)
    user = db.get_user_by_google_id(sub)
    if user:
        st.session_state.logged_in = True
        st.session_state.user_id = user["id"]
        st.session_state.username = user["username"]
        st.session_state.current_thread_id = None
        st.session_state.messages = []
        st.session_state.auth_method = "google"
        return True
    return False


def render_login_signup():
    """
    Show login/signup forms. "Log in with Google" is rendered above this in main().
    """
    st.markdown(
        """
        <div class="hero-card">
            <h2>Your private space to talk things through.</h2>
            <p>
                Save your conversations, continue where you left off, and keep your
                thoughts organized in one calm place.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Continue with email")
    tab1, tab2 = st.tabs(["Sign in", "Create account"])
    with tab1:
        with st.form("login_form"):
            st.subheader("Welcome back")
            uname = st.text_input(
                "Username",
                key="login_username",
                max_chars=32,
                placeholder="Enter your username",
            )
            pwd = st.text_input("Password", type="password", key="login_pwd")
            if st.form_submit_button("Sign in", use_container_width=True):
                if not uname or not pwd:
                    st.error("Please enter username and password.")
                else:
                    user = db.get_user_by_username(uname)
                    if user and auth.verify_password(pwd, user["password_hash"]):
                        st.session_state.logged_in = True
                        st.session_state.user_id = user["id"]
                        st.session_state.username = user["username"]
                        st.session_state.current_thread_id = None
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
    with tab2:
        with st.form("signup_form"):
            st.subheader("Create your account")
            uname = st.text_input(
                "Username",
                key="signup_username",
                max_chars=32,
                placeholder="Choose a username",
            )
            pwd = st.text_input("Password", type="password", key="signup_pwd", help="Use at least 8 characters.")
            pwd2 = st.text_input("Confirm password", type="password", key="signup_pwd2")
            if st.form_submit_button("Create account", use_container_width=True):
                ok, err = auth.validate_username(uname)
                if not ok:
                    st.error(err)
                else:
                    ok, err = auth.validate_password(pwd)
                    if not ok:
                        st.error(err)
                    elif pwd != pwd2:
                        st.error("Passwords do not match.")
                    else:
                        h = auth.hash_password(pwd)
                        uid = db.create_user(uname, h)
                        if uid:
                            st.success("Account created! Log in with your credentials.")
                        else:
                            st.error("Username already taken. Try another.")


def render_chat_app():
    """
    Main chat interface after login or as guest.
    Guest: messages in session_state only, no DB. Logged-in: threads + messages in DB.
    """
    is_guest = st.session_state.get("guest", False)
    user_id = st.session_state.get("user_id")
    current_tid = st.session_state.get("current_thread_id")
    messages = st.session_state.get("messages", [])
    username = st.session_state.get("username", "User")
    if "seed_prompt" not in st.session_state:
        st.session_state.seed_prompt = ""

    threads = db.get_threads_for_user(user_id) if (not is_guest and user_id) else []
    thread_count = len(threads)
    message_count = len(messages)

    # --- Sidebar: New Chat, thread list, Logout ---
    with st.sidebar:
        st.markdown('<span class="brand-chip">Sukoon Companion</span>', unsafe_allow_html=True)
        st.title("Sukoon")
        st.caption(f"Signed in as {username}")
        st.markdown(
            """
            <div class="sidebar-note">
                A warm companion for stressful days, heavy thoughts, and small wins
                that still matter.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="sidebar-stats">
                <div class="stat-pill"><strong>{thread_count}</strong><br/>Chats</div>
                <div class="stat-pill"><strong>{message_count}</strong><br/>Messages</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("New conversation", use_container_width=True, type="primary"):
            st.session_state.current_thread_id = None
            st.session_state.messages = []
            st.rerun()

        st.markdown("**Conversation history**")
        if not threads:
            st.caption(
                "No chats yet. Start with New conversation or pick a starter prompt below."
                if not is_guest
                else "Guest mode is active. Chats are not saved."
            )
        for t in threads:
            tid = t["id"]
            name = (t["thread_name"] or "New conversation").strip()
            display_name = (name[:38] + "...") if len(name) > 38 else name
            if tid == current_tid:
                display_name = f"Current: {display_name}"
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(display_name, key=f"thread_{tid}", use_container_width=True):
                    st.session_state.current_thread_id = tid
                    st.session_state.messages = db.get_messages_for_thread(tid, user_id)
                    st.rerun()
            with col2:
                if st.button("Del", key=f"del_{tid}", help="Delete this chat"):
                    if db.delete_thread(tid, user_id):
                        if current_tid == tid:
                            st.session_state.current_thread_id = None
                            st.session_state.messages = []
                        st.rerun()

        # Rename current thread (when one is selected, logged-in users only)
        if current_tid and not is_guest and user_id:
            st.markdown("---")
            current_thread = next((t for t in threads if t["id"] == current_tid), None) or db.get_thread(current_tid, user_id)
            current_name = (current_thread["thread_name"] or "New conversation").strip() if current_thread else "New conversation"
            with st.expander("Rename conversation"):
                new_name = st.text_input("New name", value=current_name, key="rename_input")
                if st.button("Save name", key="rename_btn") and new_name and new_name.strip():
                    if db.update_thread_name(current_tid, user_id, new_name.strip()):
                        st.success("Renamed.")
                        st.rerun()

        st.markdown("---")
        if st.button("Log out", use_container_width=True):
            if hasattr(st, "logout"):
                st.logout()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # --- Main area: welcome or messages ---
    if not current_tid and not messages:
        st.markdown(
            """
            <div class="hero-card">
                <h2>A calm conversation space, built for real life.</h2>
                <p>
                    Talk things out, reflect on your day, or ask for gentle guidance.
                    Pick a starter below or type your own message.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        starters = [
            ("I am feeling overwhelmed and need to calm down.", "I need to calm down"),
            ("Help me process a difficult conversation I had today.", "Process a difficult conversation"),
            ("Can you help me set a simple plan for this week?", "Plan my week"),
            ("I need a confidence boost before an important task.", "Confidence boost"),
        ]
        col_a, col_b = st.columns(2)
        for idx, (prompt, label) in enumerate(starters):
            with (col_a if idx % 2 == 0 else col_b):
                if st.button(label, key=f"starter_prompt_{idx}", use_container_width=True):
                    st.session_state.seed_prompt = prompt
                    st.rerun()
        st.markdown("---")

    for msg in messages:
        role = "user" if msg["role"] in ("user", "human") else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])

    # --- Chat input ---
    typed_input = st.chat_input("Share what is on your mind. Sukoon is listening.")
    seeded_input = st.session_state.pop("seed_prompt", "")
    user_input = (typed_input or seeded_input or "").strip()
    if user_input:
        if is_guest:
            # Guest: in-memory only, no DB
            messages.append({"role": "user", "content": user_input})
            st.session_state.messages = messages
            history_for_prompt = messages[:-1]
        else:
            # Logged-in: create thread if new chat, persist to DB
            if not current_tid:
                thread_name = (user_input[:50] + "...") if len(user_input) > 50 else user_input
                current_tid = db.create_thread(user_id, thread_name)
                st.session_state.current_thread_id = current_tid
            messages.append({"role": "user", "content": user_input})
            st.session_state.messages = messages
            db.add_message(current_tid, "user", user_input)
            db_messages = db.get_messages_for_thread(current_tid, user_id)
            history_for_prompt = db_messages[:-1] if db_messages else []

        with st.chat_message("user"):
            st.markdown(user_input)

        chat_history = messages_to_langchain(history_for_prompt)
        chain = create_chain()
        try:
            provider = (get_setting("LLM_PROVIDER", "groq") or "groq").lower()
            if provider == "gemini" and not get_setting("GOOGLE_API_KEY"):
                st.error("GOOGLE_API_KEY is not set. Add it to your .env file, or use LLM_PROVIDER=groq with GROQ_API_KEY set.")
                st.stop()
            with st.spinner("Sukoon is thinking..."):
                response = invoke_chain(chain, {"input": user_input, "chat_history": chat_history})
            assistant_text = response.content if hasattr(response, "content") else str(response)
            if not assistant_text or not str(assistant_text).strip():
                assistant_text = "I'm sorry, I couldn't generate a response. Please try again."
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "resource_exhausted" in err_msg:
                st.error(
                    "Gemini rate limit reached. Switch to Groq for a hosted option: set LLM_PROVIDER=groq in .env and add GROQ_API_KEY."
                )
            elif "401" in err_msg or "unauthorized" in err_msg:
                st.error(
                    "Groq auth failed (HTTP 401). Use a valid active GROQ_API_KEY, then fully restart Streamlit."
                )
                with st.expander("Error details"):
                    st.code(str(e)[:500])
            elif "connection" in err_msg or "refused" in err_msg or "groq" in err_msg or "groq api" in err_msg:
                st.error(
                    "Groq API connection failed. Ensure GROQ_API_KEY and GROQ_API_URL (if custom) are set in .env and reachable."
                )
                with st.expander("Error details"):
                    st.code(str(e)[:500])
            else:
                st.error("Something went wrong. Check your setup and try again.")
                with st.expander("Error details"):
                    st.code(str(e)[:500])
            # Remove user message so they can retry
            if messages:
                messages.pop()
            st.session_state.messages = messages
            if not is_guest and current_tid:
                # Remove the user message we saved to DB (best-effort; last insert)
                pass  # DB doesn't have easy "undo last insert" for Messages
            st.stop()

        messages.append({"role": "assistant", "content": assistant_text})
        st.session_state.messages = messages
        if not is_guest and current_tid:
            db.add_message(current_tid, "assistant", assistant_text)

        with st.chat_message("assistant"):
            st.markdown(assistant_text)


def main():
    """Entry point. Init DB, inject CSS, then auth gate or chat app."""
    st.set_page_config(
        page_title="Sukoon",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Ensure SQLite tables exist on every run.
    db.init_db()

    # Auth gate: logged in via Google (st.user) or email/password (session_state).
    google_logged_in = hasattr(st, "user") and getattr(st.user, "is_logged_in", False)
    session_logged_in = st.session_state.get("logged_in", False)

    if google_logged_in:
        _sync_google_user_to_session()
        render_chat_app()
    elif session_logged_in:
        render_chat_app()
    else:
        st.markdown(
            """
            <div class="hero-card">
                <h2>Sukoon, your steady pocket companion.</h2>
                <p>
                    Chat freely when life feels heavy, sort your thoughts, and return to
                    your saved conversations anytime.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Only show Google login if .streamlit/secrets.toml [auth] is configured.
        google_auth_ready = False
        try:
            if hasattr(st, "secrets") and st.secrets.get("auth", {}).get("client_id"):
                google_auth_ready = True
        except Exception:
            pass

        col_guest, col_google = st.columns(2)

        # Use without login (guest mode) - no account, no saved chats.
        with col_guest:
            if st.button("Continue as guest", type="primary", use_container_width=True):
                st.session_state.logged_in = True
                st.session_state.guest = True
                st.session_state.user_id = None
                st.session_state.username = "Guest"
                st.session_state.current_thread_id = None
                st.session_state.messages = []
                st.rerun()

        with col_google:
            if google_auth_ready and hasattr(st, "login"):
                if st.button("Sign in with Google", use_container_width=True):
                    st.login()
            else:
                st.button("Google sign-in unavailable", use_container_width=True, disabled=True)

        st.caption("Guest mode keeps chats only in this browser session.")
        st.markdown("---")
        render_login_signup()


if __name__ == "__main__":
    main()

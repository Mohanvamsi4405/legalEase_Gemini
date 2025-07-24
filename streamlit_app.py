import streamlit as st
import fitz  # PyMuPDF
import easyocr
import numpy as np
import requests
import pandas as pd
import docx
import io
import zipfile
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import re
import base64
import hashlib
from datetime import date, datetime
from dateutil import parser
from dateutil.parser import ParserError
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import os
import uuid
from fuzzywuzzy import fuzz
import pickle # For user persistence

# --- Authentication Configuration ---
USERS_FILE = "users.pkl" # File to store user data (username: hashed_password)

# --- Global Configuration ---
st.set_page_config(
    page_title="⚖️ LegalEase AI – Multi-Case Legal Analysis",
    page_icon="⚖️",
    layout="wide"
)
st.title("⚖️ LegalEase AI – Multi-Case Legal Analysis")

# --- Authentication Functions ---
def load_users():
    """Load user data from the pickle file."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_users(users):
    """Save user data to the pickle file."""
    with open(USERS_FILE, "wb") as f:
        pickle.dump(users, f)

def hash_password(password):
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    """Authenticate a user against stored credentials."""
    users = st.session_state.users
    hashed_password = hash_password(password)
    return users.get(username) == hashed_password

def register_user(username, password):
    """Register a new user."""
    users = st.session_state.users
    if username in users:
        return False # User already exists
    users[username] = hash_password(password)
    save_users(users)
    return True

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "logged_in": False,
        "username": None,
        "users": load_users(), # Load users at startup
        "cases": [],
        "active_case_index": 0,
        "processed_file_hash": None,
        "audio_uploader_key_counter": 0,
        "gemini_configured": False,
        "debug_log": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Language Selection ---
LANGUAGE_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Marathi": "mr",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Gujarati": "gu"
}

# --- Sidebar Configuration ---
st.sidebar.title("🛠️ Controls")

if st.session_state.logged_in:
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.cases = [] # Clear case data on logout
        st.session_state.active_case_index = 0
        st.session_state.processed_file_hash = None
        st.session_state.audio_uploader_key_counter = 0
        st.session_state.debug_log = []
        st.rerun()

selected_language = st.sidebar.selectbox(
    "🌐 Select Language for Voice Input/Output",
    options=list(LANGUAGE_OPTIONS.keys()),
    index=0,
    help="Choose the language for audio input (speech-to-text) and output (text-to-speech)."
)
language_code = LANGUAGE_OPTIONS[selected_language]

api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password", help="Required for AI features")

use_external_llm = st.sidebar.checkbox(
    "Allow External LLM Queries",
    value=True,
    help="Enable to allow general legal queries and speculative judgments."
)

if api_key:
    try:
        genai.configure(api_key=api_key)
        st.session_state.gemini_configured = True
    except Exception as e:
        st.sidebar.error(f"Failed to configure Gemini API: {e}")
        st.session_state.gemini_configured = False
else:
    st.session_state.gemini_configured = False

# --- Utility Functions ---
def clear_all_data():
    """Reset the entire application state, except user data"""
    st.session_state.cases = []
    st.session_state.active_case_index = 0
    st.session_state.processed_file_hash = None
    st.session_state.audio_uploader_key_counter = 0
    st.session_state.debug_log = []
    st.success("✅ All session memory cleared!")
    st.cache_data.clear()
    st.cache_resource.clear()

if st.session_state.logged_in: # Only show clear data if logged in
    st.sidebar.button("🧹 Clear All Data", on_click=clear_all_data)

# --- Core Processing Functions ---
@st.cache_resource
def get_easyocr_reader():
    """Initialize and cache the EasyOCR reader"""
    return easyocr.Reader(['en'], gpu=False, verbose=False)

reader = get_easyocr_reader()

def transcribe_audio(audio_bytes, lang_code):
    """Convert audio to text using Google Speech Recognition"""
    r = sr.Recognizer()
    try:
        # Determine input format based on file header or common extensions
        if audio_bytes.startswith(b'ID3') or audio_bytes[4:8] == b'ftyp':
            input_format = "mp3"
        elif audio_bytes.startswith(b'RIFF') and audio_bytes[8:12] == b'WAVE':
            input_format = "wav"
        elif audio_bytes.startswith(b'\x1aE\xdf\xa3'):
            input_format = "webm"
        else:
            # Fallback for uncertain cases, assuming webm is common from browser capture
            input_format = "webm"

        with io.BytesIO(audio_bytes) as audio_file:
            audio = AudioSegment.from_file(audio_file, format=input_format)
            wav_file = io.BytesIO()
            audio.export(wav_file, format="wav")
            wav_file.seek(0)

            with sr.AudioFile(wav_file) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language=lang_code)
                return text
    except sr.UnknownValueError:
        return f"Could not understand audio in {selected_language}"
    except sr.RequestError as e:
        return f"Speech recognition service error: {e}"
    except Exception as e:
        return f"Error transcribing audio: {e}"

@st.cache_data(show_spinner=False)
def generate_embeddings(text, api_key_for_embed):
    """Generate embeddings using Gemini API"""
    if not api_key_for_embed or not text.strip():
        st.session_state.debug_log.append(f"Embedding failed: Empty text or no API key")
        return np.zeros((768,))
    
    try:
        # Using genai.embed_content for consistency and potential future internal optimization
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query" # Or "retrieval_document" for chunks
        )
        return np.array(result["embedding"])
    except Exception as e:
        st.session_state.debug_log.append(f"Embedding API error: {e}")
        return np.zeros((768,))

@st.cache_data(show_spinner=False)
def llm_image_summary(image_bytes, api_key_for_llm, model_name="gemini-1.5-flash"):
    """Generate summary of an image using Gemini"""
    if not api_key_for_llm or not st.session_state.gemini_configured:
        st.session_state.debug_log.append("Image summary failed: Gemini API not configured")
        return "[Image Summary error]: Gemini API not configured."
    
    try:
        model = genai.GenerativeModel(model_name)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        max_size = 2048 # Max dimension for image processing
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        prompt = """Analyze this image in detail for legal relevance. Focus on:
        - People, objects, documents visible
        - Any text content (even if partial)
        - Potential legal significance
        - Relationships between elements
        Be thorough but concise."""
        
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.GenerationConfig(max_output_tokens=500)
        )
        
        if response.candidates and response.candidates[0].content.parts:
            summary = response.candidates[0].content.parts[0].text
            st.session_state.debug_log.append(f"Image summary generated: {summary[:50]}...")
            return summary
        st.session_state.debug_log.append("Image summary failed: No valid response")
        return "[Image Summary error]: No valid response from model."
    except Exception as e:
        st.session_state.debug_log.append(f"Image summary error: {e}")
        return f"[Image Summary error]: {str(e)}"

@st.cache_data(show_spinner=False)
def ocr_image(image_bytes):
    """Perform OCR on an image with enhanced validation"""
    try:
        img_io = io.BytesIO(image_bytes)
        image = Image.open(img_io)
        if image.format not in ['PNG', 'JPEG', 'JPG']:
            st.session_state.debug_log.append(f"Unsupported image format: {image.format}")
            return "", None
        image.verify() # Verify if it's a valid image
        image = Image.open(img_io).convert("RGB") # Re-open to ensure proper loading after verify
        ocr_text = "\n".join(reader.readtext(np.array(image), detail=0))
        st.session_state.debug_log.append(f"OCR successful, extracted text length: {len(ocr_text)}")
        return ocr_text, image
    except Exception as e:
        st.session_state.debug_log.append(f"OCR failed: {e}")
        return "", None

def generate_text_completion(prompt, api_key_for_llm, model_name="gemini-1.5-flash", max_tokens=500):
    """Generate text using Gemini"""
    if not api_key_for_llm or not st.session_state.gemini_configured:
        st.session_state.debug_log.append("Text completion failed: Gemini API not configured")
        return "[LLM Error]: Gemini API not configured."
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens)
        )
        
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        st.session_state.debug_log.append("Text completion failed: No valid response")
        return "[LLM Error]: No valid response from model."
    except Exception as e:
        st.session_state.debug_log.append(f"Text completion error: {e}")
        return f"[LLM Error]: {str(e)}"

# --- Document Processing Functions ---
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes):
    """Extract text and images from PDF"""
    text = ""
    images = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text() + "\n"
            for img_info in page.get_images(full=True):
                base_image = doc.extract_image(img_info[0])
                if base_image and base_image["image"]:
                    try:
                        # Validate image before appending
                        Image.open(io.BytesIO(base_image["image"])).verify()
                        images.append(base_image["image"])
                        st.session_state.debug_log.append(f"Extracted valid image from PDF page {page.number}")
                    except Exception as e:
                        st.session_state.debug_log.append(f"Invalid image in PDF page {page.number}: {e}")
        doc.close()
    except Exception as e:
        st.session_state.debug_log.append(f"PDF extraction error: {e}")
    return text, images

@st.cache_data(show_spinner=False)
def extract_text_from_docx(file_bytes):
    """Extract text and images from DOCX with enhanced validation"""
    text = ""
    images = []
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        
        # Extract images from DOCX
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref and rel.target_part and hasattr(rel.target_part, 'blob'):
                img_bytes = rel.target_part.blob
                try:
                    Image.open(io.BytesIO(img_bytes)).verify() # Validate image
                    images.append(img_bytes)
                    st.session_state.debug_log.append(f"Extracted valid image from DOCX: {len(img_bytes)} bytes")
                except Exception as e:
                    st.session_state.debug_log.append(f"Invalid image in DOCX: {e}")
    except Exception as e:
        st.session_state.debug_log.append(f"DOCX extraction error: {e}")
    return text, images

@st.cache_data(show_spinner=False)
def extract_text_from_xlsx(file_bytes):
    """Extract data (as string) and images from Excel files"""
    text = ""
    images = []
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
        text = df.to_string() # Convert DataFrame to string
        
        # XLSX files are ZIP archives, images might be inside xl/media/
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            for name in z.namelist():
                if name.startswith("xl/media/"):
                    img_bytes = z.read(name)
                    try:
                        Image.open(io.BytesIO(img_bytes)).verify() # Validate image
                        images.append(img_bytes)
                        st.session_state.debug_log.append(f"Extracted valid image from XLSX: {name}")
                    except Exception as e:
                        st.session_state.debug_log.append(f"Invalid image in XLSX: {e}")
    except Exception as e:
        st.session_state.debug_log.append(f"Excel extraction error: {e}")
    return text, images

def is_legal_content(text):
    """Determine if text contains legal-relevant content based on keywords."""
    if not isinstance(text, str):
        return False
    
    legal_keywords = [
        "court", "judge", "petitioner", "respondent", "plaintiff", "defendant",
        "section", "act", "statute", "law", "legal", "case", "filing",
        "order", "judgment", "hearing", "trial", "evidence", "witness",
        "contract", "agreement", "clause", "party", "parties"
    ]
    # Check for any of the keywords using regex for whole words
    return any(re.search(r"\b" + re.escape(k) + r"\b", text, re.IGNORECASE) for k in legal_keywords)

# --- Case Management Functions ---
def extract_all_cases(text_content):
    """Extract all cases from text using LLM."""
    if not st.session_state.gemini_configured or not text_content.strip():
        st.session_state.debug_log.append("Case extraction failed: API not configured or empty text")
        return []
    
    # Trim text_content to fit API limits, if necessary
    prompt = f"""Identify ALL distinct legal cases in the following text. For EACH case, provide:
    1. Case Type (e.g., Criminal, Civil, Family, Property)
    2. Case Number 
    3. Petitioner/Plaintiff
    4. Respondent/Defendant
    5. Date (YYYY-MM-DD format)
    6. Location (court/city/state)
    7. Relevant Sections/Acts (comma-separated)
    8. Brief Summary (1-2 sentences)

    Format each case EXACTLY like this:
    === Case Start ===
    Case Type: <value>
    Case Number: <value>
    Petitioner: <value>
    Respondent: <value>
    Date: <value>
    Location: <value>
    Sections: <value>
    Summary: <value>
    === Case End ===

    Text:
    {text_content[:20000]}""" # Limit input text to avoid API errors
    
    response = generate_text_completion(prompt, api_key, max_tokens=2000)
    st.session_state.debug_log.append(f"Extracted cases: {response[:100]}...") # Log a snippet
    return parse_multiple_cases(response)

def parse_multiple_cases(response):
    """Parse multiple cases from LLM response into a structured list."""
    cases = []
    current_case = {}
    
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('=== Case Start ==='):
            current_case = {
                "type_of_case": "N/A",
                "case_number": "N/A",
                "petitioner": "N/A",
                "respondent": "N/A",
                "date": None, # Use None for missing dates
                "location": "N/A",
                "sections": "N/A",
                "summary": "N/A",
                "chunks": [], # Text chunks for RAG
                "embeddings": [], # Embeddings for RAG
                "crime_scene_summaries": [], # For images
                "chat_history": [], # Per-case chat history
                "manual_notes": "", # User's manual notes
                "analysis": { # LLM-generated analysis
                    "summary": "N/A",
                    "key_entities": "N/A",
                    "legal_issues": "N/A",
                    "recommendations": "N/A"
                }
            }
        elif line.startswith('=== Case End ==='):
            if current_case: # Only add if a case was being built
                cases.append(current_case)
                current_case = {} # Reset for next case
        elif ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Map extracted keys to internal case structure keys
            if key == "Case Type":
                current_case["type_of_case"] = value
            elif key == "Case Number":
                current_case["case_number"] = value
            elif key == "Petitioner":
                current_case["petitioner"] = value
            elif key == "Respondent":
                current_case["respondent"] = value
            elif key == "Date":
                try:
                    current_case["date"] = parser.parse(value).date()
                except (ValueError, ParserError):
                    current_case["date"] = None # Set to None if date parsing fails
            elif key == "Location":
                current_case["location"] = value
            elif key == "Sections":
                current_case["sections"] = value
            elif key == "Summary":
                current_case["summary"] = value
    
    st.session_state.debug_log.append(f"Parsed {len(cases)} cases")
    return cases

def generate_case_analysis(case_text, case_info):
    """Generate comprehensive case analysis for a given case."""
    if not st.session_state.gemini_configured or not case_text.strip():
        st.session_state.debug_log.append("Case analysis failed: API not configured or empty text")
        return {
            "summary": "N/A (API not configured)",
            "key_entities": "N/A",
            "legal_issues": "N/A",
            "recommendations": "N/A"
        }
    
    prompt = f"""Analyze the following legal case content for case {case_info['case_number']} ONLY:

    Case Information:
    - Type: {case_info['type_of_case']}
    - Number: {case_info['case_number']}
    - Petitioner: {case_info['petitioner']}
    - Respondent: {case_info['respondent']}
    - Date: {case_info['date']}
    - Location: {case_info['location']}
    - Sections: {case_info['sections']}

    Content:
    {case_text[:15000]}

    Provide your analysis with these sections:
    1. Case Summary (3-5 sentences summarizing key facts and issues for {case_info['case_number']})
    2. Key Entities (list of people, organizations, locations specific to this case)
    3. Legal Issues (main legal questions/problems identified for this case)
    4. Recommendations (suggested next steps or actions for this case)

    Format your response with clear section headings and focus ONLY on {case_info['case_number']}."""
    
    response = generate_text_completion(prompt, api_key, max_tokens=1000)
    st.session_state.debug_log.append(f"Case analysis for {case_info['case_number']}: {response[:50]}...")
    return parse_analysis_response(response)

def parse_analysis_response(response):
    """Parse the LLM's analysis response into structured data."""
    sections = {
        "summary": "N/A",
        "key_entities": "N/A",
        "legal_issues": "N/A",
        "recommendations": "N/A"
    }
    
    current_section = None
    for line in response.split('\n'):
        line = line.strip()
        if not line: # Skip empty lines
            continue
            
        # Identify section headings
        if "Case Summary" in line:
            current_section = "summary"
            sections[current_section] = "" # Initialize section content
        elif "Key Entities" in line:
            current_section = "key_entities"
            sections[current_section] = ""
        elif "Legal Issues" in line:
            current_section = "legal_issues"
            sections[current_section] = ""
        elif "Recommendations" in line:
            current_section = "recommendations"
            sections[current_section] = ""
        elif current_section: # Append line to current section
            sections[current_section] += line + "\n"
    
    # Clean up whitespace and ensure 'N/A' for empty sections
    for key in sections:
        sections[key] = sections[key].strip()
        if not sections[key]:
            sections[key] = "N/A"
    
    return sections

def generate_chat_response(query, context_chunks, case_info, use_external_llm=False):
    """Generate a response to a legal query using RAG or judge-mode LLM."""
    if not st.session_state.gemini_configured:
        st.session_state.debug_log.append("Chat response failed: Gemini API not configured")
        return "Gemini API is not configured. Please enter your API key."
    
    # Select top N relevant chunks for context
    context = "\n\n".join(context_chunks[:3]) # Use top 3 relevant chunks
    
    # Check if the query is directly answerable by context or asks for speculative judgment
    context_relevant = any(query.lower() in chunk.lower() for chunk in context_chunks)
    is_judge_query = any(kw in query.lower() for kw in ["judge", "win", "guilty", "liable", "outcome", "verdict"])
    
    if use_external_llm and (not context_relevant or is_judge_query):
        # Use external LLM if allowed and context isn't directly relevant or query is speculative
        prompt = f"""You are a legal expert acting as a judge in {selected_language}. Answer the following query for case {case_info['case_number']} by reasoning through the evidence and applicable laws, providing a speculative judgment if specific details are missing. Explain your reasoning clearly, considering the case context and general legal principles.

Case Information:
- Type: {case_info['type_of_case']}
- Number: {case_info['case_number']}
- Petitioner: {case_info['petitioner']}
- Respondent: {case_info['respondent']}
- Date: {case_info['date']}
- Location: {case_info['location']}
- Sections: {case_info['sections']}

Relevant Context (if any):
{context if context else "No highly relevant context found in documents."}

User Query: {query}

Response:"""
        st.session_state.debug_log.append(f"Using judge-mode LLM for query: {query}")
    else:
        # Use RAG for questions that should be answered from documents
        prompt = f"""You are a legal assistant analyzing case {case_info['case_number']} in {selected_language}. Provide a concise, accurate response to the user's query based SOLELY on the provided context. If the answer isn't in the context, say "I cannot answer that based on the provided documents."

Case Information:
- Type: {case_info['type_of_case']}
- Number: {case_info['case_number']}
- Petitioner: {case_info['petitioner']}
- Respondent: {case_info['respondent']}
- Date: {case_info['date']}
- Location: {case_info['location']}
- Sections: {case_info['sections']}

Relevant Context:
{context}

User Query: {query}

Response:"""
        st.session_state.debug_log.append("Using RAG with document context")
    
    response = generate_text_completion(prompt, api_key)
    st.session_state.debug_log.append(f"Chat response: {response[:50]}...")
    return response

def handle_legal_query(query, case_index):
    """Process a legal query with RAG for a specific case."""
    if not st.session_state.cases or case_index >= len(st.session_state.cases):
        st.session_state.debug_log.append("Query failed: No cases or invalid index")
        return "No case documents have been processed. Please upload relevant files first."
    
    case = st.session_state.cases[case_index]
    if not case['chunks'] or not any(e.any() for e in case['embeddings']): # Check if embeddings are non-zero
        st.session_state.debug_log.append("Query failed: No meaningful chunks or embeddings for this case.")
        return "No processed documents or embeddings available for this case. Please ensure documents were processed correctly."
    
    # Generate embedding for the query
    query_embedding = generate_embeddings(query, api_key)
    
    # Calculate similarity and get top N chunks
    similarities = cosine_similarity([query_embedding], case['embeddings'])[0]
    top_indices = similarities.argsort()[-3:][::-1] # Get indices of top 3 most similar chunks
    
    # Filter for chunks that actually contain the case number (optional, but helps focus context)
    context_chunks = [case['chunks'][i] for i in top_indices if case['case_number'].lower() in case['chunks'][i].lower() or fuzz.partial_ratio(query.lower(), case['chunks'][i].lower()) > 60]

    # If specific case-number chunks are not found, use the top 3 generally relevant chunks
    if not context_chunks:
        context_chunks = [case['chunks'][i] for i in top_indices]

    return generate_chat_response(query, context_chunks, case, use_external_llm=use_external_llm)

# --- UI Components ---

def login_page():
    """Displays the login and signup interface."""
    st.sidebar.empty() # Clear other sidebar elements temporarily
    st.title("Welcome to LegalEase AI")
    st.subheader("Login or Sign Up")

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        st.markdown("### Existing User Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if authenticate_user(login_username, login_password):
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.success(f"Welcome back, {login_username}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_signup:
        st.markdown("### New User Sign Up")
        new_username = st.text_input("Choose a Username", key="new_username")
        new_password = st.text_input("Choose a Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")

        if st.button("Sign Up"):
            if not new_username or not new_password or not confirm_password:
                st.warning("All fields are required.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long.")
            elif register_user(new_username, new_password):
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Username already exists. Please choose a different one.")

def display_case_selection():
    """Display case selection dropdown"""
    if not st.session_state.cases:
        return
    
    case_options = [f"{case['case_number']} - {case['type_of_case']}" for case in st.session_state.cases]
    selected_case = st.selectbox(
        "🔍 Select Case to Analyze",
        options=range(len(case_options)),
        format_func=lambda x: case_options[x],
        index=st.session_state.active_case_index
    )
    
    # Update active case if a different one is selected
    if selected_case != st.session_state.active_case_index:
        st.session_state.active_case_index = selected_case
        st.rerun()

def display_case_info():
    """Display case information section for the active case"""
    if not st.session_state.cases:
        st.info("No cases available. Please upload legal documents first.")
        return
    
    case = st.session_state.cases[st.session_state.active_case_index]
    
    st.header(f"📋 Case Information - {case['case_number']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Case Type:** {case['type_of_case']}")
        st.markdown(f"**Case Number:** {case['case_number']}")
        st.markdown(f"**Petitioner/Plaintiff:** {case['petitioner']}")
        st.markdown(f"**Respondent/Defendant:** {case['respondent']}")
    
    with col2:
        st.markdown(f"**Date:** {case['date'] or 'N/A'}")
        st.markdown(f"**Location:** {case['location']}")
        st.markdown(f"**Relevant Sections/Acts:**")
        st.markdown(case['sections'])
        st.markdown(f"**Summary:** {case['summary']}")
    
    st.subheader("Additional Notes")
    # Text area for manual notes, unique key per case
    case['manual_notes'] = st.text_area(
        "Add your notes here:",
        value=case['manual_notes'],
        height=150,
        key=f"notes_{st.session_state.active_case_index}"
    )

def display_case_analysis():
    """Display the case analysis section for the active case"""
    if not st.session_state.cases:
        st.info("No cases available. Please upload legal documents first.")
        return
    
    case = st.session_state.cases[st.session_state.active_case_index]
    
    st.header(f"📊 Case Analysis - {case['case_number']}")
    
    st.subheader("Case Summary")
    st.write(case['analysis']['summary'])
    
    # Download button for summary
    b64_summary = base64.b64encode(f"""Case Summary for {case['case_number']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{case['analysis']['summary']}
""".encode()).decode()
    st.markdown(f'<a href="data:file/txt;base64,{b64_summary}" download="case_summary_{case["case_number"]}.txt">📥 Download Case Summary</a>', unsafe_allow_html=True)
    
    st.subheader("Key Entities")
    st.write(case['analysis']['key_entities'])
    
    st.subheader("Legal Issues")
    st.write(case['analysis']['legal_issues'])
    
    st.subheader("Recommendations")
    st.write(case['analysis']['recommendations'])

    # Download button for full analysis report
    analysis_text = f"""LegalEase AI Case Analysis Report
Case: {case['case_number']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Case Information:
- Type: {case['type_of_case']}
- Number: {case['case_number']}
- Petitioner: {case['petitioner']}
- Respondent: {case['respondent']}
- Date: {case['date']}
- Location: {case['location']}
- Sections: {case['sections']}
- Summary: {case['summary']}

Case Summary:
{case['analysis']['summary']}

Key Entities:
{case['analysis']['key_entities']}

Legal Issues:
{case['analysis']['legal_issues']}

Recommendations:
{case['analysis']['recommendations']}

Additional Notes:
{case['manual_notes']}
"""
    
    b64 = base64.b64encode(analysis_text.encode()).decode()
    st.markdown(f'<a href="data:file/txt;base64,{b64}" download="legal_case_analysis_{case["case_number"]}.txt">📥 Download Full Analysis Report</a>', unsafe_allow_html=True)

def display_image_analysis():
    """Display image analysis results for the active case"""
    if not st.session_state.cases:
        st.info("No cases available. Please upload legal documents first.")
        return
    
    case = st.session_state.cases[st.session_state.active_case_index]
    
    st.header(f"📸 Document Image Analysis - {case['case_number']}")
    
    if not case['crime_scene_summaries']:
        st.info("No images with legal relevance detected for this case.")
        return
    
    for idx, img_data in enumerate(case['crime_scene_summaries']):
        with st.expander(f"Image {idx+1} from {img_data['file_name']}"):
            if img_data['image']:
                # Ensure PIL Image is displayed correctly
                st.image(img_data['image'], use_container_width=True, caption=f"Extracted Image {idx+1}")
            
            st.subheader("OCR Extracted Text")
            st.text(img_data['ocr_text'] or "No text detected")
            
            st.subheader("AI Analysis")
            st.write(img_data['llm_summary'])
            
            image_summary_text = f"""Image Analysis for {case['case_number']}
Image: {img_data['file_name']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OCR Extracted Text:
{img_data['ocr_text'] or 'No text detected'}

AI Analysis:
{img_data['llm_summary']}
"""
            b64_img = base64.b64encode(image_summary_text.encode()).decode()
            st.markdown(f'<a href="data:file/txt;base64,{b64_img}" download="image_summary_{case["case_number"]}_{idx+1}.txt">📥 Download Image Summary</a>', unsafe_allow_html=True)

def display_legal_chat():
    """Display the legal chat interface for the active case"""
    if not st.session_state.cases:
        st.info("No cases available. Please upload legal documents first.")
        return
    
    case = st.session_state.cases[st.session_state.active_case_index]
    
    st.header(f"💬 Legal Assistant Chat - {case['case_number']} ({selected_language})")
    
    # Download chat history button
    chat_history_text = f"""LegalEase AI Chat History for {case['case_number']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    for message in case['chat_history']:
        role = "User" if message["role"] == "user" else "Assistant"
        chat_history_text += f"{role}: {message['content']}\n\n"
    
    b64_chat = base64.b64encode(chat_history_text.encode()).decode()
    st.markdown(f'<a href="data:file/txt;base64,{b64_chat}" download="chat_history_{case["case_number"]}.txt">📥 Download Chat History</a>', unsafe_allow_html=True)
    
    # Display chat messages
    for idx, message in enumerate(case['chat_history']):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")
                b64_audio = base64.b64encode(message["audio"]).decode()
                st.markdown(f'<a href="data:audio/mp3;base64,{b64_audio}" download="chat_response_{case["case_number"]}_{idx+1}.mp3">📥 Download Audio Response</a>', unsafe_allow_html=True)
    
    # Chat input and audio uploader
    chat_input = st.chat_input(f"Ask about case {case['case_number']} in {selected_language}...")
    
    # Increment key counter for file uploader to force re-render when needed
    # This prevents Streamlit from thinking the same file is uploaded if user cancels and re-uploads
    st.session_state.audio_uploader_key_counter += 1 
    audio_file = st.file_uploader(
        "Or upload audio question",
        type=["mp3", "wav", "ogg"],
        key=f"audio_upload_{st.session_state.active_case_index}_{st.session_state.audio_uploader_key_counter}" # Unique key
    )
    
    if chat_input or audio_file:
        if audio_file:
            audio_bytes = audio_file.getvalue()
            # Use a hash to prevent re-processing the same audio file repeatedly within the same session
            audio_hash = hashlib.md5(audio_bytes).hexdigest()
            
            # Check if this audio file has already been processed for this case
            if audio_hash != case.get("last_audio_hash_for_case"):
                with st.spinner("Processing audio..."):
                    transcribed_text = transcribe_audio(audio_bytes, language_code)
                    case["last_audio_hash_for_case"] = audio_hash # Store the hash
                    
                    if transcribed_text.startswith("Could not understand") or transcribed_text.startswith("Speech recognition service error"):
                        st.error(transcribed_text + ". Please try again or type your question.")
                        st.session_state.debug_log.append(f"Audio transcription failed: {transcribed_text}")
                        # Don't add to chat history if transcription failed significantly
                        return
                    
                    # Add user's audio input as a message
                    case['chat_history'].append({
                        "role": "user",
                        "content": transcribed_text,
                        "audio": audio_bytes
                    })
                    
                    with st.spinner("Researching your question..."):
                        response = handle_legal_query(transcribed_text, st.session_state.active_case_index)
                        
                        audio_response = io.BytesIO()
                        try:
                            tts = gTTS(text=response, lang=language_code)
                            tts.write_to_fp(audio_response)
                            audio_response.seek(0)
                        except Exception as e:
                            st.warning(f"Couldn't generate audio response: {e}")
                            st.session_state.debug_log.append(f"Audio generation error: {e}")
                            audio_response = None # Set to None if audio generation fails
                        
                        # Add AI's response
                        case['chat_history'].append({
                            "role": "assistant",
                            "content": response,
                            "audio": audio_response.getvalue() if audio_response else None
                        })
                        
                        st.rerun() # Rerun to display new messages
            else:
                st.info("This audio file has already been processed for this case.")
                
        elif chat_input: # If text input
            case['chat_history'].append({
                "role": "user",
                "content": chat_input
            })
            
            with st.spinner("Researching your question..."):
                response = handle_legal_query(chat_input, st.session_state.active_case_index)
                
                audio_response = io.BytesIO()
                try:
                    tts = gTTS(text=response, lang=language_code)
                    tts.write_to_fp(audio_response)
                    audio_response.seek(0)
                except Exception as e:
                    st.warning(f"Couldn't generate audio response: {e}")
                    st.session_state.debug_log.append(f"Audio generation error: {e}")
                    audio_response = None
                
                case['chat_history'].append({
                    "role": "assistant",
                    "content": response,
                    "audio": audio_response.getvalue() if audio_response else None
                })
                
                st.rerun() # Rerun to display new messages

# --- Main Processing Function ---
def process_uploaded_files(uploaded_files):
    """Process all uploaded files and extract case information."""
    if not uploaded_files:
        st.session_state.debug_log.append("No files uploaded")
        return
    
    all_text = []
    all_images = [] # To store extracted PIL Image objects and their metadata
    
    # Calculate a hash of all uploaded files to avoid re-processing if files are unchanged
    hasher = hashlib.md5()
    for file in uploaded_files:
        hasher.update(file.getvalue())
    files_hash = hasher.hexdigest()
    
    if st.session_state.processed_file_hash == files_hash:
        st.info("Files already processed. No new processing needed.")
        st.session_state.debug_log.append("Files already processed, skipping.")
        return
    
    # Process each uploaded file
    for file in uploaded_files:
        file_bytes = file.getvalue()
        ext = file.name.split('.')[-1].lower()
        
        text_content = ""
        images = [] # Images extracted from the current file
        
        # Dispatch based on file extension
        if ext == "pdf":
            text_content, images = extract_text_from_pdf(file_bytes)
        elif ext == "docx":
            text_content, images = extract_text_from_docx(file_bytes)
        elif ext == "xlsx":
            text_content, images = extract_text_from_xlsx(file_bytes)
        elif ext in ["png", "jpg", "jpeg"]:
            # If image file itself is uploaded
            images = [file_bytes]
            text_content = f"Image file: {file.name}" # Add a placeholder text for image files
        elif ext == "txt":
            text_content = file_bytes.decode("utf-8", errors="ignore")
        else:
            st.warning(f"Unsupported file type: {file.name}")
            st.session_state.debug_log.append(f"Unsupported file type: {file.name}")
            continue
        
        # Only add text content if it seems legally relevant
        if text_content.strip() and is_legal_content(text_content):
            all_text.append(text_content)
        elif text_content.strip():
            st.warning(f"File {file.name} does not seem to contain legal content and will be partially processed.")
            all_text.append(text_content) # Still add to all_text even if not strictly legal, for context
        
        # Process extracted images (OCR and LLM summary)
        for idx, img_bytes in enumerate(images):
            ocr_text, pil_image = ocr_image(img_bytes) # pil_image is a PIL.Image object or None
            img_summary = llm_image_summary(img_bytes, api_key) if st.session_state.gemini_configured else "Image analysis requires API key."
            
            image_info = {
                "file_name": f"{file.name}_image_{idx+1}",
                "ocr_text": ocr_text,
                "llm_summary": img_summary,
                "image": pil_image # Store the PIL Image object
            }
            
            all_images.append(image_info)
            st.session_state.debug_log.append(
                f"Processed image {image_info['file_name']}: OCR text length={len(ocr_text)}, Summary={img_summary[:50]}..."
            )
    
    full_text = "\n\n".join(all_text)
    
    # Extract cases from the aggregated text using LLM
    cases = extract_all_cases(full_text)
    
    # For each identified case, further process and link data
    for case in cases:
        # Isolate case-specific text chunks
        case_text = ""
        # Heuristic: Find text blocks containing the case number
        for text_block in all_text:
            if case['case_number'].lower() in text_block.lower():
                case_text += text_block + "\n\n"
        
        # Chunk the case-specific text for RAG
        # Split by double newlines, filter out very short chunks
        case['chunks'] = [chunk.strip() for chunk in re.split(r'\n\s*\n', case_text) if len(chunk.strip()) > 100]
        
        # Generate embeddings for each chunk
        case['embeddings'] = []
        for chunk in case['chunks']:
            emb = generate_embeddings(chunk, api_key) # API key check is inside generate_embeddings
            case['embeddings'].append(emb)
        
        # Link relevant images to this case based on keyword matching
        case['crime_scene_summaries'] = []
        # Create a list of keywords for the current case for matching
        case_keywords = [
            case['case_number'],
            case['petitioner'],
            case['respondent'],
            case['location'],
            case['summary'],
            case['sections'],
            case['type_of_case'].lower() # Add case type for better matching
        ]
        case_keywords = [kw.lower() for kw in case_keywords if kw and kw != "N/A"] # Filter out empty/N/A keywords
        
        for img_data in all_images:
            # Combine OCR text and LLM summary for matching
            img_content = (img_data['ocr_text'].lower() + " " + img_data['llm_summary'].lower())
            
            # Use fuzzy matching and direct keyword presence
            match_scores = [fuzz.partial_ratio(kw, img_content) for kw in case_keywords]
            if any(score > 70 for score in match_scores) or any(kw in img_content for kw in case_keywords):
                case['crime_scene_summaries'].append(img_data)
                st.session_state.debug_log.append(
                    f"Assigned image {img_data['file_name']} to case {case['case_number']} (scores: {match_scores})"
                )
        
        # Generate detailed analysis for the case
        case['analysis'] = generate_case_analysis(case_text, case)
        
        # Initialize chat history and manual notes for the new case
        case['chat_history'] = []
        case['manual_notes'] = ""
    
    # Update session state with the new cases
    st.session_state.cases = cases
    st.session_state.active_case_index = 0 if cases else 0 # Set active case to the first one, or 0 if no cases
    st.session_state.processed_file_hash = files_hash # Store hash to prevent re-processing
    
    st.session_state.debug_log.append(
        f"Processed {len(cases)} cases with {sum(len(c['crime_scene_summaries']) for c in cases)} images."
    )
    st.success(f"✅ Successfully processed {len(cases)} cases with {sum(len(c['crime_scene_summaries']) for c in cases)} images.")

# --- Main Application Flow ---
def main():
    """Main application flow"""

    if not st.session_state.logged_in:
        login_page()
    else:
        # Only show upload and main features if logged in
        st.sidebar.header("📁 Upload Case Documents")
        uploaded_files = st.sidebar.file_uploader(
            "Upload legal documents (PDF, DOCX, XLSX, TXT, Images)",
            type=["pdf", "docx", "xlsx", "txt", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload all relevant case documents for analysis"
        )
        
        if uploaded_files and st.session_state.gemini_configured:
            with st.spinner("Processing documents..."):
                process_uploaded_files(uploaded_files)
        elif uploaded_files and not st.session_state.gemini_configured:
            st.warning("Please enter a valid Gemini API Key in the sidebar to process documents.")
        
        display_case_selection()
        
        # Dynamically set tab names based on active case, if available
        if st.session_state.cases:
            case = st.session_state.cases[st.session_state.active_case_index]
            tab1, tab2, tab3, tab4 = st.tabs([
                f"📋 {case['case_number']} Info", 
                f"📊 {case['case_number']} Analysis", 
                f"📸 {case['case_number']} Images", 
                f"💬 {case['case_number']} Chat"
            ])
        else:
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Case Info", 
                "📊 Case Analysis", 
                "📸 Image Analysis", 
                "💬 Legal Chat"
            ])
        
        with tab1:
            display_case_info()
        
        with tab2:
            display_case_analysis()
        
        with tab3:
            display_image_analysis()
        
        with tab4:
            display_legal_chat()
        
        if st.sidebar.checkbox("Show debug information"):
            st.sidebar.write("### Debug Info")
            st.sidebar.json({
                "num_cases": len(st.session_state.cases),
                "active_case_index": st.session_state.active_case_index,
                "file_hash": st.session_state.processed_file_hash,
                "gemini_configured": st.session_state.gemini_configured,
                "users_loaded": list(st.session_state.users.keys()), # Show registered users (usernames only)
                "debug_log": st.session_state.get('debug_log', [])
            })

if __name__ == "__main__":
    main()





















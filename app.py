import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
from langdetect import detect, DetectorFactory
from typing import List, Dict
import base64
import io
import tempfile
import json
from datetime import datetime

DetectorFactory.seed = 0

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    client = Groq()
else:
    client = None

LANGUAGES: List[str] = [
    "English", "Urdu", "Arabic", "Hindi", "Punjabi", "Sindhi", "Pashto",
    "Persian", "Turkish", "Chinese (Simplified)", "Chinese (Traditional)",
    "Japanese", "Korean", "French", "German", "Spanish", "Italian", "Russian",
    "Portuguese", "Indonesian", "Bengali", "Vietnamese", "Thai", "Dutch",
    "Polish", "Ukrainian", "Romanian", "Greek", "Czech", "Swedish",
    "Hungarian", "Hebrew", "Danish", "Finnish", "Norwegian", "Slovak",
    "Bulgarian", "Croatian", "Serbian", "Malay", "Tamil", "Telugu",
    "Marathi", "Gujarati", "Kannada", "Swahili", "Amharic", "Nepali",
    "Sinhala", "Burmese"
]

LANG_CODE_MAP: Dict[str, str] = {
    "en": "English", "ur": "Urdu", "ar": "Arabic", "hi": "Hindi", "pa": "Punjabi",
    "sd": "Sindhi", "ps": "Pashto", "fa": "Persian", "tr": "Turkish",
    "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese", "ko": "Korean", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "ru": "Russian", "pt": "Portuguese",
    "id": "Indonesian", "bn": "Bengali", "vi": "Vietnamese", "th": "Thai",
    "nl": "Dutch", "pl": "Polish", "uk": "Ukrainian", "ro": "Romanian",
    "el": "Greek", "cs": "Czech", "sv": "Swedish", "hu": "Hungarian",
    "he": "Hebrew", "da": "Danish", "fi": "Finnish", "no": "Norwegian",
    "sk": "Slovak", "bg": "Bulgarian", "hr": "Croatian", "sr": "Serbian",
    "ms": "Malay", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "gu": "Gujarati", "kn": "Kannada", "sw": "Swahili", "am": "Amharic",
    "ne": "Nepali", "si": "Sinhala", "my": "Burmese"
}

LANG_SET = set(LANGUAGES)

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into the target language. "
    "Requirements: 1) Keep meaning and tone. 2) Do not add explanations. 3) Return only the translation."
)

def detect_language_name(text: str) -> str:
    try:
        code = detect(text)
        if code.startswith("zh"):
            code = "zh-cn"
        return LANG_CODE_MAP.get(code, code)
    except Exception:
        return "Unknown"

def extract_text_from_image(image_bytes: bytes) -> str:
    return "Image OCR is currently unavailable. Groq has deprecated vision models. Please use PDF or text files instead."

def extract_text_from_video(video_bytes: bytes) -> str:
    return "Video OCR is currently unavailable. Groq has deprecated vision models. Please use PDF or text files instead."

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        import PyPDF2
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip() or "No text found in PDF"
    except ImportError:
        return "Install PyPDF2: pip install PyPDF2"
    except Exception as e:
        return f"PDF error: {str(e)}"

def extract_text_from_file(uploaded_file) -> str:
    file_type = uploaded_file.type
    file_bytes = uploaded_file.read()
    
    if file_type.startswith('image/'):
        return extract_text_from_image(file_bytes)
    elif file_type.startswith('video/'):
        return extract_text_from_video(file_bytes)
    elif file_type == 'application/pdf':
        return extract_text_from_pdf(file_bytes)
    elif file_type.startswith('text/'):
        return file_bytes.decode('utf-8', errors='ignore')
    else:
        return file_bytes.decode('utf-8', errors='ignore')

def translate_with_groq(text: str, target_language: str, source_language: str = None, temperature: float = 0.2) -> str:
    if not client:
        raise RuntimeError("GROQ_API_KEY not set. Please add it to your .env file.")

    if source_language and source_language != "Auto-detect":
        user_prompt = (
            f"Source language: {source_language}\n"
            f"Target language: {target_language}\n"
            f"Text: '''{text}'''\n"
            f"Return only the translated text."
        )
    else:
        user_prompt = (
            f"Detect source language automatically.\n"
            f"Target language: {target_language}\n"
            f"Text: '''{text}'''\n"
            f"Return only the translated text."
        )

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=2000,
    )
    return resp.choices[0].message.content.strip()

def save_to_history(source_text: str, translated_text: str, source_lang: str, target_lang: str):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': source_text[:100] + '...' if len(source_text) > 100 else source_text,
        'translated': translated_text[:100] + '...' if len(translated_text) > 100 else translated_text,
        'source_lang': source_lang,
        'target_lang': target_lang
    })
    if len(st.session_state.history) > 10:
        st.session_state.history.pop()

if 'history' not in st.session_state:
    st.session_state.history = []
if 'batch_mode' not in st.session_state:
    st.session_state.batch_mode = False

st.set_page_config(page_title="50-Language Translator", page_icon="🌐", layout="wide")

st.title("🌐 Advanced 50-Language Translator Pro")
st.caption("⚡ Powered by Groq + Llama 3.3 · Lightning-fast translations across 50 languages")
st.markdown("<p style='text-align: center; color: #667eea; font-weight: bold;'>👨‍💻 Developed by Imtiaz</p>", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.write("Enter your Groq API key:")
    
    user_api_key = st.text_input(
        "🔑 Groq API Key",
        value=GROQ_API_KEY,
        type="password",
        placeholder="Paste your API key here",
        help="Get free API key from https://console.groq.com"
    )
    
    if user_api_key and user_api_key != GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = user_api_key
        client = Groq()
        st.success("✅ API key updated!")
    
    if not user_api_key:
        st.warning("⚠️ Enter your Groq API key")
        st.info("Get free key: https://console.groq.com")
    
    st.divider()
    st.subheader("📊 Statistics")
    st.metric("Supported Languages", "50")
    st.metric("Model", "Llama 3.3 70B")
    
    st.divider()
    st.subheader("🎯 Features")
    st.markdown("""
    ✅ 50 languages support  
    ✅ Auto language detection  
    ✅ Batch translation  
    ✅ Translation history  
    ✅ Quality settings  
    ✅ File upload (PDF/Text)  
    ✅ Multiple target languages  
    """)
    
    st.divider()
    st.subheader("⚙️ Advanced Settings")
    quality = st.select_slider(
        "Translation Quality",
        options=["Fast", "Balanced", "Accurate"],
        value="Balanced",
        help="Fast=0.1, Balanced=0.2, Accurate=0.3"
    )
    temp_map = {"Fast": 0.1, "Balanced": 0.2, "Accurate": 0.3}
    st.session_state.temperature = temp_map[quality]
    
    st.divider()
    st.info("💡 **Get API Key:**\n1. Visit console.groq.com\n2. Sign up free\n3. Copy API key\n4. Paste above")
    
    st.divider()
    st.markdown("""<div style='text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
    <h3 style='margin: 0;'>👨‍💻 Made by Imtiaz</h3>
    <p style='margin: 5px 0 0 0; font-size: 14px;'>Full Stack Developer</p>
    </div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔄 Translate", "📊 Batch Mode", "📜 History"])

with tab1:
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        source_lang = st.selectbox("🔤 Source Language", ["Auto-detect"] + LANGUAGES, index=0)
    with col2:
        target_lang = st.selectbox("🎯 Target Language", LANGUAGES, index=1)
    with col3:
        st.write("")
        st.write("")
        swap = st.button("🔁 Swap", use_container_width=True)

    if swap:
        if source_lang != "Auto-detect":
            source_lang, target_lang = target_lang, source_lang
            st.rerun()

    st.subheader("📤 Upload File (Optional)")
    uploaded_file = st.file_uploader(
        "Upload PDF or text file",
        type=['pdf', 'txt', 'doc', 'docx'],
        help="Supports: PDFs and text files"
    )

    if uploaded_file:
        col_file1, col_file2 = st.columns([1, 3])
        with col_file1:
            st.info(f"📎 {uploaded_file.name}")
            st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")
        with col_file2:
            if st.button("🔍 Extract Text from File", type="primary"):
                with st.spinner("Extracting text..."):
                    try:
                        extracted = extract_text_from_file(uploaded_file)
                        st.session_state.input_text = extracted
                        st.success("✅ Text extracted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")

    st.divider()
    text = st.text_area("📝 Enter text to translate", height=200, placeholder="Type or paste text here…", key="input_text")

    col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
    with col_a:
        go = st.button("🚀 Translate", type="primary", use_container_width=True)
    with col_b:
        clear = st.button("🧹 Clear", use_container_width=True)
    with col_c:
        demo = st.button("✨ Demo", use_container_width=True)
    with col_d:
        char_count = len(text) if text else 0
        st.metric("Characters", char_count)

    if demo:
        st.session_state.input_text = "Assalam-o-Alaikum! How are you? This advanced app translates between 50 languages with AI-powered accuracy."
        st.rerun()

    if clear:
        st.session_state.input_text = ""
        st.rerun()

    if text and source_lang == "Auto-detect":
        detected = detect_language_name(text)
        st.info(f"Detected: **{detected}**")

    output = ""
    if go:
        if not text.strip():
            st.warning("Please enter some text to translate.")
        elif target_lang not in LANG_SET:
            st.error("Please select a valid target language.")
        else:
            try:
                with st.spinner("Translating..."):
                    temp = st.session_state.get('temperature', 0.2)
                    output = translate_with_groq(
                        text.strip(),
                        target_lang,
                        None if source_lang == "Auto-detect" else source_lang,
                        temperature=temp
                    )
                    save_to_history(text.strip(), output, source_lang, target_lang)
            except Exception as e:
                st.error(f"Translation failed: {e}")

    if output:
        st.success("✅ Translation completed!")
        st.subheader(f"📄 Translation ({target_lang})")
        
        col_out1, col_out2 = st.columns([4, 1])
        with col_out1:
            st.text_area("Translated text", value=output, height=200, key="output_text")
        with col_out2:
            st.write("")
            st.write("")
            st.download_button(
                "💾 Download",
                data=output,
                file_name=f"translation_{target_lang.lower().replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            if st.button("📋 Copy", use_container_width=True):
                st.code(output, language=None)

with tab2:
    st.subheader("📊 Batch Translation")
    st.write("Translate to multiple languages at once")
    
    batch_text = st.text_area("📝 Enter text for batch translation", height=150, key="batch_input")
    selected_langs = st.multiselect("🎯 Select target languages", LANGUAGES, default=["Urdu", "Arabic"])
    
    if st.button("🚀 Translate to All", type="primary"):
        if batch_text.strip() and selected_langs:
            with st.spinner(f"Translating to {len(selected_langs)} languages..."):
                results = {}
                for lang in selected_langs:
                    try:
                        temp = st.session_state.get('temperature', 0.2)
                        results[lang] = translate_with_groq(batch_text.strip(), lang, temperature=temp)
                    except Exception as e:
                        results[lang] = f"Error: {str(e)}"
                
                st.success(f"✅ Translated to {len(selected_langs)} languages!")
                for lang, trans in results.items():
                    with st.expander(f"🌐 {lang}"):
                        st.text_area(f"{lang} translation", value=trans, height=100, key=f"batch_{lang}")
                        st.download_button(
                            f"💾 Download {lang}",
                            data=trans,
                            file_name=f"translation_{lang.lower().replace(' ', '_')}.txt",
                            key=f"download_{lang}"
                        )

with tab3:
    st.subheader("📜 Translation History")
    if st.session_state.history:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
        
        for idx, item in enumerate(st.session_state.history):
            with st.expander(f"⏱️ {item['timestamp']} | {item['source_lang']} → {item['target_lang']}"):
                st.write(f"**Source:** {item['source']}")
                st.write(f"**Translation:** {item['translated']}")
    else:
        st.info("💭 No translation history yet. Start translating to see your history here!")

st.markdown("---")
st.markdown("""<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 20px 0;'>
<h2 style='color: white; margin: 0;'>👨‍💻 Made by Imtiaz</h2>
<p style='color: #f0f0f0; margin: 10px 0 0 0; font-size: 16px;'>Full Stack Developer | AI Enthusiast</p>
<p style='color: #e0e0e0; margin: 5px 0 0 0; font-size: 14px;'>🚀 Building innovative solutions with AI</p>
</div>""", unsafe_allow_html=True)

st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)
with col_footer1:
    st.caption("💡 50 languages with AI-powered accuracy")
with col_footer2:
    st.caption("⚡ Powered by Groq Llama 3.3 70B")
with col_footer3:
    st.caption("🔒 Your API key is secure")

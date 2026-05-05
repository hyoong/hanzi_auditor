import streamlit as st
from huggingface_hub import InferenceClient
from gtts import gTTS
from io import BytesIO
import json
import random
import base64
import re

# --- 1. CONFIG & CLIENT ---
HF_API_KEY = st.secrets["HF_API_KEY"]
client = InferenceClient(api_key=HF_API_KEY)
MODEL_ID = "sentence-transformers/LaBSE"
THRESHOLD = 0.7

def load_audio_player():
    st.components.v1.html("""
        <script>
            function playAudioFromBase64(b64) {
                var audio = new Audio("data:audio/mp3;base64," + b64);
                audio.play().catch(e => console.log("Audio error:", e));
            }
        </script>
    """, height=0)

# --- Function to normalize pinyin (fix nu:3 → nǚ) ---
def normalize_pinyin(pinyin):
    """
    Convert pinyin with numbers and colons to proper tone marks
    Examples:
        nu:3 → nǚ
        ni3 hao3 → nǐ hǎo
        nu:2 ren2 → nǚ rén
    """
    if not pinyin:
        return pinyin
    
    # First, replace colon with ü (for nü, lü, etc.)
    pinyin = pinyin.replace('u:', 'ü').replace('U:', 'Ü')
    
    # Now convert tone numbers to marks
    tone_marks = {
        'a': ['ā', 'á', 'ǎ', 'à'],
        'e': ['ē', 'é', 'ě', 'è'],
        'i': ['ī', 'í', 'ǐ', 'ì'],
        'o': ['ō', 'ó', 'ǒ', 'ò'],
        'u': ['ū', 'ú', 'ǔ', 'ù'],
        'ü': ['ǖ', 'ǘ', 'ǚ', 'ǜ']
    }
    
    # Split into syllables
    syllables = pinyin.split()
    converted = []
    
    for syllable in syllables:
        # Find tone number at the end
        match = re.search(r'([a-zü]+)(\d)', syllable, re.IGNORECASE)
        if match:
            base, tone_num = match.groups()
            tone = int(tone_num)
            
            if tone == 5:  # Neutral tone
                converted.append(base)
                continue
            
            # Find which vowel gets the tone mark
            marked = False
            for vowel in ['a', 'e', 'o']:
                if vowel in base:
                    pos = base.index(vowel)
                    base = base[:pos] + tone_marks[vowel][tone-1] + base[pos+1:]
                    marked = True
                    break
            
            if not marked:
                # Handle iu and ui special cases
                if 'iu' in base:
                    pos = base.index('u')
                    base = base[:pos] + tone_marks['u'][tone-1] + base[pos+1:]
                elif 'ui' in base:
                    pos = base.index('i')
                    base = base[:pos] + tone_marks['i'][tone-1] + base[pos+1:]
                else:
                    for vowel in ['i', 'u', 'ü']:
                        if vowel in base:
                            pos = base.index(vowel)
                            base = base[:pos] + tone_marks[vowel][tone-1] + base[pos+1:]
                            break
                marked = True
            
            converted.append(base)
        else:
            converted.append(syllable.lower())
    
    return ' '.join(converted)

# --- Function to generate and play audio (no caching, plays every time) ---

def get_audio_base64(chinese_char):
    try:
        tts = gTTS(text=chinese_char, lang='zh', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        return base64.b64encode(audio_bytes).decode()
    except Exception as e:
        print(f"Audio generation failed: {e}")
        return ""

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('hsk_audit_v2.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Missing 'hsk_audit_v2.json'! Run your transform script first.")
        return {}

hsk_data = load_data()

# --- 3. SIDEBAR & SETTINGS ---
with st.sidebar:
    st.title("⚙️ Audit Settings")
    
    is_locked = st.session_state.get('index', 0) > 0 or st.session_state.get('answered', False)
    
    if 'temp_level' not in st.session_state:
        st.session_state.temp_level = list(hsk_data.keys())[0] if hsk_data else ""
    if 'temp_filter' not in st.session_state:
        st.session_state.temp_filter = "Any"
    
    if is_locked:
        st.warning("🔒 Settings locked during Audit")
        st.write(f"Level: **{st.session_state.current_level}**")
        st.write(f"Type: **{st.session_state.current_filter}**")
        st.info(f"🤖 AI Strictness: **Fixed at {THRESHOLD}** (LaBSE cross-lingual model)")
    else:
        level = st.selectbox(
            "Select HSK Level", 
            list(hsk_data.keys()),
            key="level_select"
        )
        length_filter = st.radio(
            "Word Type", 
            ["Any", "Single Characters Only", "Compounds (2+ chars)"],
            key="filter_select"
        )
        
        st.info(f"🤖 AI Similarity Threshold: **Fixed at {THRESHOLD}** (LaBSE cross-lingual model)")
        
        st.session_state.temp_level = level
        st.session_state.temp_filter = length_filter
        
        if ('current_level' not in st.session_state or 
            'current_filter' not in st.session_state or
            st.session_state.current_level != level or 
            st.session_state.current_filter != length_filter):
            st.session_state.current_level = level
            st.session_state.current_filter = length_filter
            if 'word_list' in st.session_state:
                del st.session_state.word_list

    st.divider()
    
    if st.button("🔄 Reset & Change Level", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if is_locked and not st.session_state.get('game_over', False):
        if st.button("🛑 End Audit Early", type="secondary", use_container_width=True):
            st.session_state.game_over = True
            st.rerun()

# --- 4. SESSION INITIALIZATION ---
settings_changed = False
if not is_locked and 'current_level' in st.session_state and 'current_filter' in st.session_state:
    selected_level = st.session_state.get('temp_level', st.session_state.current_level)
    selected_filter = st.session_state.get('temp_filter', st.session_state.current_filter)
    
    if (selected_level != st.session_state.current_level or 
        selected_filter != st.session_state.current_filter):
        settings_changed = True
        st.session_state.current_level = selected_level
        st.session_state.current_filter = selected_filter
        if 'word_list' in st.session_state:
            del st.session_state.word_list

if 'word_list' not in st.session_state or settings_changed:
    if 'current_level' not in st.session_state or st.session_state.current_level not in hsk_data:
        if hsk_data:
            st.session_state.current_level = list(hsk_data.keys())[0]
        else:
            st.error("No data available!")
            st.stop()
    
    if 'current_filter' not in st.session_state:
        st.session_state.current_filter = "Any"
    
    pool = hsk_data.get(st.session_state.current_level, [])
    
    if st.session_state.current_filter == "Single Characters Only":
        pool = [w for w in pool if len(w['char']) == 1]
    elif st.session_state.current_filter == "Compounds (2+ chars)":
        pool = [w for w in pool if len(w['char']) > 1]
    
    if not pool:
        st.error(f"No words match the filter '{st.session_state.current_filter}' in level {st.session_state.current_level}!")
        st.stop()

    random.shuffle(pool)
    st.session_state.word_list = pool
    st.session_state.index = 0
    st.session_state.count = 0
    st.session_state.answered = False
    st.session_state.game_over = False

# --- 5. SCORE CALCULATION FUNCTION ---
def get_grade(percentage):
    if percentage >= 80: return "A 🌟"
    if percentage >= 60: return "B 👍"
    if percentage >= 50: return "C 😐"
    return "Needs more work 📚"

# --- 6. GAME OVER UI ---
if st.session_state.get('game_over', False) or st.session_state.index >= len(st.session_state.word_list):
    st.balloons()
    st.header("🏁 Audit Results")
    
    words_played = st.session_state.index
    total_in_level = len(st.session_state.word_list)
    correct = st.session_state.count
    
    qualification_threshold = int(total_in_level * 0.20)
    
    col1, col2 = st.columns(2)
    col1.metric("Correct Answers", f"{correct} / {words_played}")
    col2.metric("Level Progress", f"{words_played} / {total_in_level}")

    if words_played < qualification_threshold:
        st.warning(f"⚠️ **Insufficient words played to qualify for a Grade.**\n(Minimum required: {qualification_threshold} words)")
    else:
        accuracy = (correct / words_played) * 100 if words_played > 0 else 0
        grade = get_grade(accuracy)
        st.subheader(f"Your Grade: {grade}")
        st.progress(accuracy / 100)
        st.write(f"Accuracy: **{accuracy:.1f}%**")

    if st.button("Start New Audit", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.stop()

# --- 7. ACTIVE GAME UI ---
current_item = st.session_state.word_list[st.session_state.index]

# Normalize the pinyin for later display (but we won't show it until after answering)
display_pinyin = normalize_pinyin(current_item.get('pinyin', ''))

st.title(f"🏮 {st.session_state.current_level} Audit")
st.write(f"Word **{st.session_state.index + 1}** of **{len(st.session_state.word_list)}**")

# Display Chinese character ONLY (no pinyin text before answering)
st.markdown(f"<h1 style='text-align: center; font-size: 120px; color: #E63946;'>{current_item['char']}</h1>", unsafe_allow_html=True)

# NO pinyin text displayed here before answering!

# Form for user input
with st.form(key=f"audit_form_{st.session_state.index}", clear_on_submit=False):
    user_guess = st.text_input(
        "Meaning in English:", 
        key=f"user_input_{st.session_state.index}", 
        disabled=st.session_state.answered
    )
    
    submit_btn = st.form_submit_button("Check Answer", use_container_width=True, disabled=st.session_state.answered)

# LOGIC AFTER SUBMISSION
if submit_btn and user_guess:
    with st.spinner("Analyzing with LaBSE + Smart Matching..."):
        scores = client.sentence_similarity(
            sentence=current_item['char'],
            other_sentences=[user_guess],
            model=MODEL_ID
        )
        
        final_score = scores[0]
        
        if final_score < THRESHOLD:
            user_lower = user_guess.lower().strip()
            definition = current_item['def']
            definition_parts = [part.strip().lower() for part in definition.split(';')]
            
            match_found = False
            matched_part = None
            
            for def_part in definition_parts:
                if user_lower == def_part or user_lower in def_part or def_part in user_lower:
                    match_found = True
                    matched_part = def_part
                    break
            
            if match_found:
                final_score = THRESHOLD + 0.05
                st.session_state.keyword_match = True
                st.session_state.matched_definition = matched_part
            else:
                st.session_state.keyword_match = False
                st.session_state.matched_definition = None
        else:
            st.session_state.keyword_match = False
            st.session_state.matched_definition = None
        
        st.session_state.last_score = final_score
        st.session_state.last_guess = user_guess 
        st.session_state.answered = True
        
        if st.session_state.last_score >= THRESHOLD:
            st.session_state.count += 1
    st.rerun()

# --- REVEAL PHASE (Pinyin and Audio shown ONLY after answering) ---
if st.session_state.answered:
    score = st.session_state.last_score
    st.divider()
    
    if score >= THRESHOLD:
        if st.session_state.get('keyword_match', False):
            st.success(f"✅ **Correct!** (Smart match: {score:.2f})")
            if st.session_state.get('matched_definition'):
                st.caption(f"🎯 Matched definition: *\"{st.session_state.matched_definition}\"*")
        else:
            st.success(f"✅ **Correct!** (LaBSE match: {score:.2f})")
    else:
        st.error(f"❌ **Not quite.** (Match: {score:.2f})")
        st.caption(f"💡 Hint: Try one of these: {current_item['def'][:50]}...")
    
    # Display pinyin and audio together (only after answering)
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.info(f"📖 **Pinyin:** {display_pinyin}")
    
    with col2:
            if st.button("🔊", key=f"audio_reveal_{st.session_state.index}", help="Click to hear pronunciation"):
                b64 = get_audio_base64(current_item['char'])
                if b64:
                    # Use st.components.v1.html with an autoplaying audio tag instead of a script tag
                    st.components.v1.html(f"""
                        <audio autoplay>
                            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                        </audio>
                    """, height=0)

    with col3:
        st.markdown(f"**Your Guess:** `{st.session_state.last_guess}`")
    
    st.markdown(f"**Official Meanings:**\n> {current_item['def']}")
    
    if st.button("Next Word ➡️", type="primary", use_container_width=True):
        st.session_state.index += 1
        st.session_state.answered = False
        if 'last_guess' in st.session_state:
            del st.session_state.last_guess
        if 'keyword_match' in st.session_state:
            del st.session_state.keyword_match
        if 'matched_definition' in st.session_state:
            del st.session_state.matched_definition
        st.rerun()

import streamlit as st
from huggingface_hub import InferenceClient
import json
import random

# --- 1. CONFIG & CLIENT ---
HF_API_KEY = st.secrets["HF_API_KEY"]
client = InferenceClient(api_key=HF_API_KEY)
MODEL_ID = "sentence-transformers/LaBSE"  # Back to LaBSE - built for cross-lingual!
THRESHOLD = 0.7  # Fixed threshold

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
    
    # LOCK LOGIC: Lock if we have moved past the first word OR are currently viewing a result
    is_locked = st.session_state.get('index', 0) > 0 or st.session_state.get('answered', False)
    
    # Always initialize these variables so they exist
    if 'temp_level' not in st.session_state:
        st.session_state.temp_level = list(hsk_data.keys())[0] if hsk_data else ""
    if 'temp_filter' not in st.session_state:
        st.session_state.temp_filter = "Any"
    
    if is_locked:
        st.warning("🔒 Settings locked during Audit")
        # Display current settings as read-only info
        st.write(f"Level: **{st.session_state.current_level}**")
        st.write(f"Type: **{st.session_state.current_filter}**")
        st.info(f"🤖 AI Strictness: **Fixed at {THRESHOLD}** (LaBSE cross-lingual model)")
    else:
        # Everything is unlocked until the first 'Check Answer' is clicked
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
        
        # Show fixed threshold as info
        st.info(f"🤖 AI Similarity Threshold: **Fixed at {THRESHOLD}** (LaBSE cross-lingual model)")
        
        # Store these in temporary variables to check for changes
        st.session_state.temp_level = level
        st.session_state.temp_filter = length_filter
        
        # Only update current settings if we haven't started or if settings changed
        if ('current_level' not in st.session_state or 
            'current_filter' not in st.session_state or
            st.session_state.current_level != level or 
            st.session_state.current_filter != length_filter):
            st.session_state.current_level = level
            st.session_state.current_filter = length_filter
            # Force word list regeneration
            if 'word_list' in st.session_state:
                del st.session_state.word_list

    st.divider()
    
    # RESTART / RESET (Always available)
    if st.button("🔄 Reset & Change Level", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # END GAME EARLY BUTTON
    if is_locked and not st.session_state.get('game_over', False):
        if st.button("🛑 End Audit Early", type="secondary", use_container_width=True):
            st.session_state.game_over = True
            st.rerun()

# --- 4. SESSION INITIALIZATION ---

# Check if settings have changed before locking
settings_changed = False
if not is_locked and 'current_level' in st.session_state and 'current_filter' in st.session_state:
    # Compare current settings with selected values from sidebar
    selected_level = st.session_state.get('temp_level', st.session_state.current_level)
    selected_filter = st.session_state.get('temp_filter', st.session_state.current_filter)
    
    if (selected_level != st.session_state.current_level or 
        selected_filter != st.session_state.current_filter):
        settings_changed = True
        # Update to new settings
        st.session_state.current_level = selected_level
        st.session_state.current_filter = selected_filter
        # Clear word list to force regeneration
        if 'word_list' in st.session_state:
            del st.session_state.word_list

# If it's a brand new session OR settings changed, (re)initialize the word list
if 'word_list' not in st.session_state or settings_changed:
    # Make sure current_level is set
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

    # Create the full randomized deck
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
    
    # Progress check for qualification
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

st.title(f"🏮 {st.session_state.current_level} Audit")
st.write(f"Word **{st.session_state.index + 1}** of **{len(st.session_state.word_list)}**")
st.markdown(f"<h1 style='text-align: center; font-size: 120px; color: #E63946;'>{current_item['char']}</h1>", unsafe_allow_html=True)

# We set clear_on_submit to FALSE so the word stays visible after hitting Enter
with st.form(key=f"audit_form_{st.session_state.index}", clear_on_submit=False):
    user_guess = st.text_input(
        "Meaning in English:", 
        key=f"user_input_{st.session_state.index}", 
        disabled=st.session_state.answered
    )
    
    submit_btn = st.form_submit_button("Check Answer", use_container_width=True, disabled=st.session_state.answered)

# LOGIC AFTER SUBMISSION - Back to LaBSE (Chinese to English comparison)
# --- ENHANCED SCORING SECTION with better phrase matching ---
if submit_btn and user_guess:
    with st.spinner("Analyzing with LaBSE + Smart Matching..."):
        # Step 1: Get LaBSE similarity score
        scores = client.sentence_similarity(
            sentence=current_item['char'],
            other_sentences=[user_guess],
            model=MODEL_ID
        )
        
        final_score = scores[0]
        
        # Step 2: Enhanced definition matching (if below threshold)
        if final_score < THRESHOLD:
            user_lower = user_guess.lower().strip()
            definition = current_item['def']
            
            # Split definition by semicolons
            definition_parts = [part.strip().lower() for part in definition.split(';')]
            all_definitions = definition_parts + [definition.lower()]
            
            match_found = False
            matched_part = None
            
            for def_part in all_definitions:
                # Exact match
                if user_lower == def_part:
                    match_found = True
                    matched_part = def_part
                    break
                
                # User's answer IN definition part (word/phrase contained)
                if user_lower in def_part:
                    match_found = True
                    matched_part = def_part
                    break
                
                # Definition part IN user's answer
                if def_part in user_lower:
                    match_found = True
                    matched_part = def_part
                    break
                
                # Check common words (ignoring stop words)
                user_words = set(user_lower.split())
                def_words = set(def_part.split())
                
                # If they share 70% of words (excluding short words)
                stop_words = {'a', 'an', 'to', 'for', 'of', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'}
                user_filtered = {w for w in user_words if w not in stop_words and len(w) > 2}
                def_filtered = {w for w in def_words if w not in stop_words and len(w) > 2}
                
                if user_filtered and def_filtered:
                    common = user_filtered.intersection(def_filtered)
                    if len(common) / max(len(user_filtered), len(def_filtered)) >= 0.5:
                        match_found = True
                        matched_part = def_part
                        break
                
                # Check for synonym pairs (custom thesaurus for common cases)
                synonym_pairs = [
                    ('apply', 'request'),
                    ('leave', 'leave of absence'),
                    ('ask', 'request'),
                    ('vacation', 'leave'),
                    ('sick', 'leave'),
                    ('permission', 'leave'),
                ]
                
                for pair in synonym_pairs:
                    if pair[0] in user_lower and pair[1] in def_part:
                        match_found = True
                        matched_part = def_part
                        break
                    if pair[1] in user_lower and pair[0] in def_part:
                        match_found = True
                        matched_part = def_part
                        break
            
            # If match found, boost score
            if match_found:
                final_score = THRESHOLD + 0.05  # Boost to 0.75
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

# --- REVEAL PHASE (updated to show match type) ---
# --- REVEAL PHASE (showing what matched) ---
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
        # Show helpful hint
        st.caption(f"💡 Hint: Try one of these: {current_item['def'][:50]}...")
    
    c1, c2 = st.columns([1, 2])
    with c1: 
        st.info(f"🔊 **{current_item['pinyin']}**")
    with c2: 
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

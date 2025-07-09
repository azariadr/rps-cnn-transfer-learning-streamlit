import streamlit as st
import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras.utils import load_img, img_to_array

# --- UI Layout ---
# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="🧠🎮 RPS Double Game",
    page_icon="✌️",
    layout="centered"
)

# --- Configuration ---
MODEL_PATH = 'model_tl_phase2_fine_tuned.keras'
CLASS_NAMES = ['Paper', 'Scissors', 'Rock'] # Confirmed order

# --- Model Loading ---
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_my_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Failed to load the game model! Error: {e}. Please ensure '{path}' is in the correct directory.")
        st.stop()

model = load_my_model(MODEL_PATH)

# --- Session State Initialization ---
if 'user_score' not in st.session_state:
    st.session_state.user_score = 0
if 'round' not in st.session_state:
    st.session_state.round = 1
if 'ai_pose' not in st.session_state:
    st.session_state.ai_pose = random.choice(CLASS_NAMES)

st.title("🧠🎮 Rock-Paper-Scissors Double Game")

st.markdown("""
Welcome to the ultimate **Rock-Paper-Scissors challenge!** 🚀
This game isn't just about luck; it's about skill and a bit of mind-reading!

Here's how to play:
1.  **Upload your hand pose** (Rock, Paper, or Scissors).
2.  **Guess the AI's secret pose!** (It's already decided!)
3.  Hit "See Results" to reveal all!

🎯 **How You Score Points:**
-   **✅ +1 Point:** If you successfully **guess the AI's pose** correctly!
-   **🏆 +1 Point:** If you **win** the Rock-Paper-Scissors battle against the AI!

Can you master both challenges? Let's find out!
""")

st.markdown("---")

# --- Game Logic ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Your Move: Upload Your Hand Pose")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("🧠 Your Guess: What's the AI's Move?")
    # Display current AI pose after a game round
    if st.session_state.round > 1:
        st.info(f"AI chose a new pose for Round {st.session_state.round}!")
    user_guess = st.radio("Predict the AI's hidden pose:", CLASS_NAMES, horizontal=True)

user_image_placeholder = st.empty() # Placeholder for the user's uploaded image

if uploaded_file:
    # Save and load image
    with open("temp_uploaded_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    img_pil = load_img("temp_uploaded_image.png", target_size=(150, 150))
    os.remove("temp_uploaded_image.png") # Clean up

    user_image_placeholder.image(img_pil, caption="Your Uploaded Pose", use_container_width=True)

    st.markdown("---")

    if st.button("🚀 Reveal Results!", use_container_width=True):
        st.balloons() # Add a celebratory effect

        ai_pose = st.session_state.ai_pose

        # Preprocess the image for the model
        x_processed = img_to_array(img_pil)
        x_processed = np.expand_dims(x_processed, axis=0) # Add batch dimension
        x_processed = x_processed / 255.0 # Normalize

        # Make prediction
        prediction = model.predict(x_processed, verbose=0) # Suppress prediction output
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        user_pose = CLASS_NAMES[predicted_class_index]

        # --- Evaluate User's Guess ---
        correct_guess = (user_guess.lower() == ai_pose.lower())
        guess_emoji = "✅ Correct!" if correct_guess else "❌ Incorrect!"

        # --- Evaluate RPS Game ---
        def get_rps_result(player_move, opponent_move):
            player_move = player_move.lower()
            opponent_move = opponent_move.lower()
            if player_move == opponent_move:
                return "Draw 🤝"
            if (player_move == "rock" and opponent_move == "scissors") or \
               (player_move == "scissors" and opponent_move == "paper") or \
               (player_move == "paper" and opponent_move == "rock"):
                return "Win! 🥳"
            return "Lose 😔"

        game_result = get_rps_result(user_pose, ai_pose)

        # --- Update Score ---
        if correct_guess:
            st.session_state.user_score += 1
        if "Win" in game_result: # Check for "Win" string to account for emoji
            st.session_state.user_score += 1

        # --- Display Results ---
        st.markdown("## 🎉 Round Results")
        st.markdown(f"🤖 **AI's Secret Pose**: <span style='font-size: 2em;'>`{ai_pose.upper()}`</span>", unsafe_allow_html=True)
        st.markdown(f"🧍 **Your Classified Pose**: <span style='font-size: 2em;'>`{user_pose.upper()}`</span>", unsafe_allow_html=True)
        st.markdown(f"🎯 **Your Guess**: **{guess_emoji}** (You guessed '{user_guess}', AI had '{ai_pose}')")
        st.markdown(f"⚔️ **RPS Battle Outcome**: `{game_result}`")

        st.markdown(f"### 💯 Current Score: **{st.session_state.user_score} points**")
        st.markdown(f"Ready for **Round {st.session_state.round + 1}**?")

        # Prepare for the next round
        st.session_state.round += 1
        st.session_state.ai_pose = random.choice(CLASS_NAMES) # AI chooses new pose

        st.markdown("---")

else:
    st.info("Upload your hand pose image to start playing!")

st.markdown("---")
st.write("Developed with ❤️ using Streamlit & TensorFlow")
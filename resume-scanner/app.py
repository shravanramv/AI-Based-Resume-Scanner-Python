import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Resume Scanner",
    page_icon="📄",
    layout="wide"
)

# --- Load Custom CSS ---
def load_css():
    try:
        with open("static/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ style.css not found.")

load_css()

# --- Landing Page UI ---
with st.container():
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 10px;'>
            AI Resume Scanner Portal
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='text-align: center; font-size: 18px; margin-top: 0;'>
            A platform where recruiters post jobs and match resumes using AI.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([4, 2, 4])
    with col2:
        if st.button("🔐 Login", use_container_width=True):
            st.switch_page("pages/login.py")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📝 Register", use_container_width=True):
            st.switch_page("pages/register.py")

# --- Footer ---
st.markdown("<hr style='margin-top: 3em;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>AI-Based Resume Scanner — Shravan Ram</p>",
    unsafe_allow_html=True
)

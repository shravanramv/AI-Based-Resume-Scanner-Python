import streamlit as st

# Wide layout for responsiveness
st.set_page_config(page_title="Resume Scanner", page_icon="📄", layout="wide")

# Optional: Load custom CSS
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

# --- Centered Page Content ---
with st.container():
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Title
    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 10px;'>
            AI Resume Scanner Portal
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Subtitle
    st.markdown(
        """
        <p style='text-align: center; font-size: 18px; margin-top: 0;'>
            A platform where recruiters post jobs and match resumes using AI.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Spacer
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Centered Column for Buttons
    col1, col2, col3 = st.columns([4, 2, 4])
    with col2:
        if st.button("🔐 Login", use_container_width=True):
            st.switch_page("pages/login.py")
        st.markdown("<br>", unsafe_allow_html=True)  # Space between buttons
        if st.button("📝 Register", use_container_width=True):
            st.switch_page("pages/register.py")

# Footer
st.markdown("<hr style='margin-top: 3em;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>AI-Based Resume Scanner - Shravan Ram</p>",
    unsafe_allow_html=True
)

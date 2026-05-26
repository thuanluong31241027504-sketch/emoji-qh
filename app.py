import streamlit as st

st.set_page_config(page_title="Emoji Classifier", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Source Code Pro', 'Courier New', monospace !important;
    }
    
    .big-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
    }
    
    .sub-text {
        font-size: 0.8rem;
        text-align: center;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    
    #MainMenu, footer, header {
        display: none !important;
    }
    
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    /* === NÚT 3D BO GÓC MÀU XANH === */
    .pushable {
        background: #0d47a1;
        border-radius: 12px;
        border: none;
        padding: 0;
        cursor: pointer;
        outline-offset: 4px;
        margin-top: 20px;
        transition: all 0.05s ease;
    }
    
    .front {
        display: block;
        padding: 12px 32px;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 700;
        background: #2196f3;
        color: white;
        transform: translateY(-6px);
        font-family: 'Source Code Pro', monospace !important;
        transition: all 0.05s ease;
    }
    
    .pushable:active .front {
        transform: translateY(-2px);
    }
    
    .pushable:hover .front {
        background: #42a5f5;
    }
    
    .draw-area {
        border: 2px dashed #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    
    .draw-placeholder {
        color: #9aa0a6;
        font-size: 0.8rem;
        padding: 60px 20px;
    }
    
    hr {
        margin-top: 2rem;
        border-color: #e0e0e0;
    }
    
    .footer {
        text-align: center;
        font-size: 0.7rem;
        color: #9aa0a6;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">EMOJI CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">by MLP model v1.0 - 2026</div>', unsafe_allow_html=True)

# ------------------- GIAO DIỆN -------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Khu vực vẽ tạm thời
    st.markdown("""
    <div class="draw-area">
        <div class="draw-placeholder">
            ✏️ Canvas sẽ hiển thị ở đây<br>
            (đang build giao diện)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Nút 3D bo góc màu xanh
    import streamlit.components.v1 as components
    
    components.html("""
    <div style="display: flex; justify-content: center;">
        <button class="pushable" id="confirmBtn">
            <span class="front">CONFIRM!</span>
        </button>
    </div>
    
    <script>
        document.getElementById('confirmBtn').addEventListener('click', () => {
            alert("Sẽ nhận diện emoji sau khi nhúng model!");
        });
    </script>
    """, height=100)
    
    # Hiển thị kết quả tạm thời
    st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <div style="font-size: 1.2rem; font-weight: 600;">---</div>
        <div style="font-size: 0.7rem; color: #5f6368;">confidence: --%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="footer">Source Code Pro · TensorFlow · MLP</div>', unsafe_allow_html=True)

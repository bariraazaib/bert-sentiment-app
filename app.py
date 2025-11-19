import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .positive { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        margin: 1.5rem 0;
    }
    
    .negative { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
        margin: 1.5rem 0;
    }
    
    .neutral { 
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        padding: 2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(252, 182, 159, 0.3);
        margin: 1.5rem 0;
    }
    
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
        padding: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    .stButton>button {
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 18px;
        transition: all 0.3s;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #667eea;
    }
    
    .test-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.7rem 0;
        border-left: 5px solid #667eea;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .test-card:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
    }
    
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 15px 15px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
    }
    
    .stDownloadButton>button {
        border-radius: 20px;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.5rem;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(17, 153, 142, 0.3);
    }
    
    .emoji-big {
        font-size: 4rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentiment_model():
    """Load the sentiment analysis model with correct label mapping"""
    try:
        model_name = "mustehsannisarrao/fine-tune-bert-sentimental-analysis"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer
        )
        
        return classifier
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def map_label_to_sentiment(label):
    """Convert LABEL_X to actual sentiment based on your model's training"""
    label_mapping = {
        'LABEL_0': 'NEGATIVE',
        'LABEL_1': 'NEUTRAL',
        'LABEL_2': 'POSITIVE'
    }
    return label_mapping.get(label, label)

def get_sentiment_emoji(sentiment):
    """Get emoji for sentiment"""
    emoji_map = {
        'POSITIVE': '🎉',
        'NEGATIVE': '😞', 
        'NEUTRAL': '😐'
    }
    return emoji_map.get(sentiment, '❓')

def create_confidence_chart(confidence, sentiment):
    """Create a confidence visualization"""
    fig, ax = plt.subplots(figsize=(10, 2))
    
    colors = {'POSITIVE': '#667eea', 'NEGATIVE': '#f5576c', 'NEUTRAL': '#fcb69f'}
    color = colors.get(sentiment, '#95a5a6')
    
    ax.barh(['Confidence'], [confidence * 100], color=color, alpha=0.8, height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.axvline(x=80, color='red', linestyle='--', alpha=0.4, linewidth=2, label='High Confidence')
    ax.legend(fontsize=10)
    
    ax.text(confidence * 100 + 2, 0, f'{confidence*100:.1f}%', 
            va='center', ha='left', fontweight='bold', fontsize=14)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-title">🎭 AI Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">✨ Powered by Fine-tuned BERT (Barirazaib) | Analyze emotions in text with advanced AI</div>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI Model..."):
        classifier = load_sentiment_model()
    
    if classifier is None:
        st.error("❌ Failed to load the model. Please check your internet connection.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">🤖 Model Info</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>📊 Model Details</strong><br><br>
            <strong>Name:</strong> Fine-tuned BERT<br>
            <strong>Task:</strong> Sentiment Analysis<br>
            <strong>Classes:</strong> 3 Categories<br><br>
            <strong>🏷️ Label Mapping:</strong><br>
            • LABEL_0 → 😞 NEGATIVE<br>
            • LABEL_1 → 😐 NEUTRAL<br>
            • LABEL_2 → 🎉 POSITIVE
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown('<div class="section-header">📈 Stats</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">BERT-based</h3>
            <p style="font-size: 13px; color: #666; margin-top: 0.5rem;">Transformer Architecture</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
            <strong>✨ Features:</strong><br>
            • Multi-class classification<br>
            • High accuracy detection<br>
            • Batch processing support<br>
            • Real-time analysis
        </div>
        """, unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Analysis", "📊 Batch Analysis", "🧪 Test Examples"])
    
    with tab1:
        st.markdown('<div class="section-header">Single Text Analysis</div>', unsafe_allow_html=True)
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            text_input = st.text_area(
                "Enter your text:",
                placeholder="Type your review, comment, or any text here to analyze sentiment...",
                height=200,
                key="single_input"
            )
            
            chars = len(text_input)
            words = len(text_input.split())
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="color: #667eea; margin: 0;">{words}</h2>
                    <p style="font-size: 14px; color: #666; margin: 0;">Words</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="color: #764ba2; margin: 0;">{chars}</h2>
                    <p style="font-size: 14px; color: #666; margin: 0;">Characters</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown("###")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); padding: 1.5rem; border-radius: 15px; text-align: center;">
                <h3 style="margin: 0; color: #333;">💡 Tips</h3>
                <p style="margin-top: 0.5rem; font-size: 14px; color: #555;">
                Enter any text to analyze its emotional tone. The AI will detect if it's positive, negative, or neutral.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("✨ Analyze Sentiment", type="primary", key="analyze_single", use_container_width=True):
            if text_input.strip():
                with st.spinner("🔮 Analyzing sentiment..."):
                    try:
                        result = classifier(text_input)[0]
                        original_label = result['label']
                        confidence = result['score']
                        
                        sentiment = map_label_to_sentiment(original_label)
                        emoji = get_sentiment_emoji(sentiment)
                        
                        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
                        
                        # Big emoji display
                        st.markdown(f'<div class="emoji-big">{emoji}</div>', unsafe_allow_html=True)
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🎯 Sentiment", sentiment)
                        
                        with col2:
                            st.metric("📊 Confidence", f"{confidence:.4f}")
                        
                        with col3:
                            st.metric("📈 Percentage", f"{confidence*100:.1f}%")
                        
                        # Confidence chart
                        st.pyplot(create_confidence_chart(confidence, sentiment))
                        
                        # Result display
                        if sentiment == 'POSITIVE':
                            st.markdown(f'<div class="positive">{emoji} POSITIVE SENTIMENT<br><span style="font-size: 1rem;">The text expresses positive emotions and optimism!</span></div>', unsafe_allow_html=True)
                        elif sentiment == 'NEGATIVE':
                            st.markdown(f'<div class="negative">{emoji} NEGATIVE SENTIMENT<br><span style="font-size: 1rem;">The text expresses negative emotions or concerns.</span></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="neutral">{emoji} NEUTRAL SENTIMENT<br><span style="font-size: 1rem;">The text is balanced or factual without strong emotion.</span></div>', unsafe_allow_html=True)
                        
                        # Technical details
                        with st.expander("🔍 Technical Details"):
                            st.json({
                                "Raw Model Output": result,
                                "Original Label": original_label,
                                "Mapped Sentiment": sentiment,
                                "Confidence Score": confidence
                            })
                            
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    with tab2:
        st.markdown('<div class="section-header">Batch Text Analysis</div>', unsafe_allow_html=True)
        
        st.markdown("**Enter multiple texts (one per line):**")
        batch_text = st.text_area(
            "",
            placeholder="Enter each text on a new line...\n\nExample:\nI love this product!\nThis is terrible.\nIt's okay, nothing special.",
            height=250,
            key="batch_input",
            label_visibility="collapsed"
        )
        
        if st.button("🚀 Analyze Batch", type="primary", use_container_width=True):
            if batch_text.strip():
                texts = [text.strip() for text in batch_text.split('\n') if text.strip()]
                
                if texts:
                    with st.spinner(f"🔮 Analyzing {len(texts)} texts..."):
                        try:
                            results = classifier(texts)
                            
                            analysis_data = []
                            for i, (text, result) in enumerate(zip(texts, results)):
                                sentiment = map_label_to_sentiment(result['label'])
                                emoji = get_sentiment_emoji(sentiment)
                                analysis_data.append({
                                    '#': i + 1,
                                    'Text': text[:50] + '...' if len(text) > 50 else text,
                                    'Sentiment': f"{emoji} {sentiment}",
                                    'Confidence': f"{result['score'] * 100:.1f}%",
                                    'Score': result['score']
                                })
                            
                            df = pd.DataFrame(analysis_data)
                            
                            st.markdown('<div class="section-header">📊 Summary</div>', unsafe_allow_html=True)
                            
                            sentiment_counts = pd.Series([map_label_to_sentiment(r['label']) for r in results]).value_counts()
                            total_texts = len(df)
                            avg_confidence = df['Score'].mean() * 100
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("📝 Total Texts", total_texts)
                            with col2:
                                st.metric("🎉 Positive", sentiment_counts.get('POSITIVE', 0))
                            with col3:
                                st.metric("😞 Negative", sentiment_counts.get('NEGATIVE', 0))
                            with col4:
                                st.metric("😐 Neutral", sentiment_counts.get('NEUTRAL', 0))
                            
                            st.markdown('<div class="section-header">📋 Detailed Results</div>', unsafe_allow_html=True)
                            st.dataframe(df[['#', 'Text', 'Sentiment', 'Confidence']], use_container_width=True, height=400)
                            
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results (CSV)",
                                data=csv,
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                else:
                    st.warning("⚠️ Please enter at least one valid text.")
            else:
                st.warning("⚠️ Please enter some texts to analyze.")
    
    with tab3:
        st.markdown('<div class="section-header">🧪 Test Examples</div>', unsafe_allow_html=True)
        st.markdown("**Click on any example below to test the model:**")
        
        test_cases = [
            ("I love this product! It's absolutely amazing and works perfectly!", "POSITIVE", "🎉"),
            ("This is terrible and awful. Worst purchase I've ever made.", "NEGATIVE", "😞"),
            ("It's okay, nothing special. Average quality and performance.", "NEUTRAL", "😐"),
            ("Excellent service! Highly recommend to everyone!", "POSITIVE", "🎉"),
            ("Complete waste of money. Very disappointed with this.", "NEGATIVE", "😞"),
            ("The product is fine. Does what it's supposed to do.", "NEUTRAL", "😐")
        ]
        
        cols = st.columns(2)
        
        for i, (text, expected, emoji) in enumerate(test_cases):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="test-card">
                    <strong>{emoji} Example {i+1}</strong><br>
                    <span style="font-size: 14px; color: #555;">{text[:60]}...</span><br>
                    <small style="color: #667eea;">Expected: {expected}</small>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Test Example {i+1}", key=f"test_{i}", use_container_width=True):
                    with st.spinner("Testing..."):
                        result = classifier(text)[0]
                        predicted_sentiment = map_label_to_sentiment(result['label'])
                        pred_emoji = get_sentiment_emoji(predicted_sentiment)
                        
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                            <strong>📝 Text:</strong> {text}<br>
                            <strong>🎯 Expected:</strong> {expected}<br>
                            <strong>🤖 Predicted:</strong> {pred_emoji} {predicted_sentiment} ({result['score']:.4f})
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if expected == predicted_sentiment:
                            st.success("✅ Perfect match!")
                        else:
                            st.error("❌ Mismatch detected!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <strong>Built with 🎭 Streamlit & 🤗 Transformers</strong><br>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

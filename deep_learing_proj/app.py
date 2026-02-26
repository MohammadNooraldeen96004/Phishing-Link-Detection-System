import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import time
import plotly.graph_objects as go
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode, unquote
import string
import re
import math
from collections import Counter
from preprocessing import URLPreprocessor 
import pandas as pd
import textwrap

# ==========================================
# 1. إعدادات الصفحة والثوابت
# ==========================================
st.set_page_config(page_title="Deep Learning Architecture", layout="wide", page_icon="🧬")

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass 

load_css("style.css")

# ++++++++++++++++++++++++++++++++++++++++++
# [إضافة جديدة]: ستايل خاص للعرض التفاعلي
# ++++++++++++++++++++++++++++++++++++++++++
st.markdown("""
<style>
    .url-text-display {
        font-family: 'Courier New', monospace;
        font-size: 20px;
        letter-spacing: 2px;
        color: #e0e0e0;
        background-color: #252d3d;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #4A5568;
        white-space: nowrap;
        overflow-x: auto;
        margin-bottom: 10px;
    }
    .cnn-window-highlight {
        background-color: #e53e3e; /* أحمر */
        color: white;
        padding: 2px 0;
        border-radius: 4px;
        font-weight: bold;
        box-shadow: 0 0 10px #e53e3e;
    }
    .risk-high-text { color: #FC8181; font-weight: bold; }
    .risk-low-text { color: #68D391; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. كلاس المعالجة
# ==========================================
pre = URLPreprocessor()

# ==========================================
# 3. توكينايزر العرض
# ==========================================
class RealTokenizerWrapper:
    def __init__(self, preprocessor):
        self.pre = preprocessor
    def texts_to_sequences(self, texts):
        return [self.pre.char_encode(t) for t in texts]

tokenizer = RealTokenizerWrapper(pre)

# ==========================================
# 4. دالة تنسيق الرابط
# ==========================================
def format_url(input_url):
    has_protocol = "://" in input_url
    temp_url = input_url if has_protocol else "http://" + input_url
    parsed = urlparse(temp_url)
    path_parts = [p for p in parsed.path.split('/') if p]
    limited_path = "/".join(path_parts[:2])
    new_path = "/" + limited_path if limited_path else ""
    formatted_url = urlunparse((
        parsed.scheme, parsed.netloc, new_path, '', '', ''
    ))
    if not has_protocol:
         formatted_url = formatted_url.replace(f"{parsed.scheme}://", "", 1)
    return formatted_url.rstrip('/') 

# ==========================================
# 5. تحميل الموارد
# ==========================================
@st.cache_resource
def load_resources():
    try:
        # 🔴 هام: تأكد من صحة المسارات على جهازك
        model_path = r'C:\Users\mohammad alsarese\Downloads\deep_Streamlt\best_url_model (2)_final.keras' 
        scaler_path = r'C:\Users\mohammad alsarese\Downloads\deep_Streamlt\scaler (2).pickle'
        
        model = load_model(model_path)
        with open(scaler_path, 'rb') as handle:
            scaler = pickle.load(handle)
            
        return model, scaler
    except Exception as e:
        st.error(f"خطأ في تحميل الملفات! تأكد من صحة المسار (Path).\nتفاصيل الخطأ: {e}")
        return None, None

model, scaler = load_resources()

# ==========================================
# دوال المحاكاة (المعدلة لتتوافق مع الموديل)
# ==========================================

# --- دالة مساعدة لتنظيف HTML ---
# ==========================================
# دوال المحاكاة المتقدمة (Advanced Visualization)
# ==========================================

def render_clean(container, html_content):
    lines = [line.strip() for line in html_content.split('\n') if line.strip()]
    container.markdown("".join(lines), unsafe_allow_html=True)

def simulate_cnn_layer(url_text, real_prediction_score):
    st.markdown("### 🕵️‍♂️ 1. CNN Feature Extraction & Max Pooling")
    st.info("Here we visualize the Conv1D Filters scanning patterns, followed by Max Pooling selecting the strongest features.")
    
    container = st.empty()
    
    # إعدادات المحاكاة
    is_phishing = real_prediction_score > 0.5
    # كلمات مفتاحية للتأثير على الأرقام المعروضة (للمحاكاة فقط)
    triggers = ["log", "pay", "sec", "acc", "bank", "upd", "verify", "lim"]
    
    # 1. مرحلة الفلاتر (Convolution)
    # ------------------------------------------------
    feature_map = []
    window_size = 4
    stride = 1
    
    # إبطاء الحركة
    delay = 0.3 
    
    for i in range(0, len(url_text) - window_size + 1, stride):
        chunk = url_text[i:i+window_size]
        
        # حساب رقم (Activation) بناءً على المحتوى وقرار الموديل الحقيقي
        base_val = np.random.uniform(0.1, 0.4)
        if any(t in chunk.lower() for t in triggers):
            # إذا كان الرابط خبيث والموديل كشفه، نعطي رقم عالي
            if is_phishing:
                base_val = np.random.uniform(2.5, 4.0) # High Activation
            else:
                base_val = np.random.uniform(0.5, 1.2) # Suppressed activation (Context saved it)
        
        feature_map.append(base_val)
        
        # لون الخلفية بناء على القوة
        bg_color = "#2D3748"
        border_color = "#4A5568"
        if base_val > 2.0:
            border_color = "#F56565" # Red border for high activation
            bg_color = "#3B1818"

        html = f"""
        <div style="font-family: 'Courier New'; background: #1A202C; padding: 20px; border-radius: 10px; border: 1px solid #4A5568;">
            <div style="color: #A0AEC0; font-size: 14px; margin-bottom: 10px;">LAYER 1: CONV1D (128 Filters) - Scanning...</div>
            
            <div style="font-size: 24px; letter-spacing: 3px; margin-bottom: 20px;">
                <span style="opacity: 0.5;">{url_text[:i]}</span>
                <span style="border: 3px solid {border_color}; padding: 2px 5px; color: white; background: {bg_color}; font-weight: bold; border-radius: 5px;">
                    {chunk}
                </span>
                <span style="opacity: 0.5;">{url_text[i+window_size:]}</span>
            </div>

            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="font-size: 16px; color: white;">Neuron Activation:</div>
                <div style="font-size: 28px; font-weight: bold; color: {border_color};">
                    {base_val:.4f}
                </div>
            </div>
            <div style="font-size: 12px; color: #718096; margin-top: 5px;">(Higher number = Suspicious Pattern Detected)</div>
        </div>
        """
        render_clean(container, html)
        time.sleep(delay)

    # 2. مرحلة الاختيار (Max Pooling)
    # ------------------------------------------------
    time.sleep(0.5)
    st.toast("Applying Max Pooling (Pool Size = 2)...")
    
    # تجميع النتائج وعرض عملية الاختيار
    pooled_values = []
    
    # نأخذ كل قيمتين مع بعض (Pool Size = 2)
    pairs = [feature_map[i:i+2] for i in range(0, len(feature_map), 2)]
    
    for pair in pairs:
        if len(pair) < 2: continue # تجاوز البقايا
        
        v1, v2 = pair[0], pair[1]
        winner = max(v1, v2)
        pooled_values.append(winner)
        
        # تحديد الألوان للفائز والخاسر
        c1 = "#F56565" if v1 == winner and v1 > 1.5 else ("#48BB78" if v1 == winner else "#718096")
        c2 = "#F56565" if v2 == winner and v2 > 1.5 else ("#48BB78" if v2 == winner else "#718096")
        
        op1 = "1.0" if v1 == winner else "0.3"
        op2 = "1.0" if v2 == winner else "0.3"
        
        scale1 = "1.2" if v1 == winner else "0.9"
        scale2 = "1.2" if v2 == winner else "0.9"

        html_pool = f"""
        <div style="font-family: 'Courier New'; background: #1A202C; padding: 20px; border-radius: 10px; border: 1px solid #9F7AEA; text-align: center;">
            <div style="color: #9F7AEA; font-size: 16px; font-weight: bold; margin-bottom: 20px;">LAYER 3: MAX POOLING (Selection)</div>
            <div style="display: flex; justify-content: center; gap: 40px; align-items: center;">
                
                <div style="text-align: center; opacity: {op1}; transform: scale({scale1}); transition: 0.3s;">
                    <div style="font-size: 14px; color: #A0AEC0;">Input A</div>
                    <div style="border: 2px solid {c1}; padding: 10px; width: 80px; font-weight: bold; color: white; border-radius: 8px;">{v1:.2f}</div>
                </div>

                <div style="font-size: 20px; color: #718096;">VS</div>

                <div style="text-align: center; opacity: {op2}; transform: scale({scale2}); transition: 0.3s;">
                    <div style="font-size: 14px; color: #A0AEC0;">Input B</div>
                    <div style="border: 2px solid {c2}; padding: 10px; width: 80px; font-weight: bold; color: white; border-radius: 8px;">{v2:.2f}</div>
                </div>
            </div>

            <div style="margin-top: 20px;">
                <div style="font-size: 30px;">⬇️</div>
                <div style="background: #9F7AEA; color: white; padding: 5px 20px; border-radius: 20px; display: inline-block; margin-top: 10px; font-weight: bold;">
                    Selected: {winner:.2f}
                </div>
            </div>
        </div>
        """
        render_clean(container, html_pool)
        time.sleep(0.6) # أبطأ عشان تلحق تشوف الاختيار

    st.success(f"✅ Pooling Complete. Reduced features from {len(feature_map)} to {len(pooled_values)}.")


def simulate_gru_layer(url_text):
    st.markdown("### 🧠 2. Bi-Directional GRU (Context Layer)")
    st.info("Bi-GRU reads the URL in two directions simultaneously to understand context (e.g., 'secure' before 'bank' vs 'bank' before 'secure').")
    
    # تقسيم الشاشة
    col_fwd, col_bwd = st.columns(2)
    
    ph_fwd = col_fwd.empty()
    ph_bwd = col_bwd.empty()
    
    chars = list(url_text)
    length = len(chars)
    steps = min(length, 15) # تحديد عدد الخطوات للعرض
    
    # Forward Pass Logic
    fwd_text = ""
    
    # Backward Pass Logic
    bwd_text = ""
    
    # حلقة التكرار (للاثنين مع بعض)
    for i in range(steps):
        # 1. تحديث الاتجاه الأمامي (Forward)
        char_f = chars[i]
        fwd_text += char_f
        
        html_fwd = f"""
        <div style="background: #0D1117; border: 1px solid #238636; border-radius: 8px; padding: 10px; height: 150px;">
            <div style="color: #238636; font-weight: bold; margin-bottom: 10px;">➡️ FORWARD GRU (Past Context)</div>
            <div style="font-family: monospace; color: #58A6FF; font-size: 18px; letter-spacing: 2px;">
                {fwd_text}<span style="color: #238636; text-decoration: blink;">_</span>
            </div>
            <div style="margin-top: 20px; font-size: 12px; color: #8B949E;">
                Memory State: Updating based on prefix...
            </div>
            <div style="width: {(i+1)/steps*100}%; height: 4px; background: #238636; margin-top: 10px; transition: 0.1s;"></div>
        </div>
        """
        render_clean(ph_fwd, html_fwd)
        
        # 2. تحديث الاتجاه العكسي (Backward)
        # نقرأ من آخر الرابط
        char_b = chars[length - 1 - i]
        bwd_text = char_b + bwd_text
        
        html_bwd = f"""
        <div style="background: #0D1117; border: 1px solid #A371F7; border-radius: 8px; padding: 10px; height: 150px;">
            <div style="color: #A371F7; font-weight: bold; margin-bottom: 10px; text-align: right;">BACKWARD GRU (Future Context) ⬅️</div>
            <div style="font-family: monospace; color: #F0883E; font-size: 18px; letter-spacing: 2px; text-align: right;">
                <span style="color: #A371F7; text-decoration: blink;">_</span>{bwd_text}
            </div>
            <div style="margin-top: 20px; font-size: 12px; color: #8B949E; text-align: right;">
                Memory State: Updating based on suffix...
            </div>
             <div style="width: 100%; display: flex; justify-content: flex-end;">
                <div style="width: {(i+1)/steps*100}%; height: 4px; background: #A371F7; margin-top: 10px; transition: 0.1s;"></div>
            </div>
        </div>
        """
        render_clean(ph_bwd, html_bwd)
        
        time.sleep(0.2) # سرعة القراءة
        
    # مرحلة الدمج (Concatenation)
    st.markdown("#### 🔗 Concatenation (Feature Fusion)")
    fusion_html = """
    <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-top: 10px;">
        <div style="background: #238636; color: white; padding: 10px 20px; border-radius: 5px;">Forward Vector (h_t)</div>
        <div style="font-size: 24px;">➕</div>
        <div style="background: #A371F7; color: white; padding: 10px 20px; border-radius: 5px;">Backward Vector (h'_t)</div>
        <div style="font-size: 24px;">🟰</div>
        <div style="background: linear-gradient(90deg, #238636, #A371F7); color: white; padding: 10px 30px; border-radius: 5px; font-weight: bold; border: 1px solid white;">Full Context Representation</div>
    </div>
    """
    st.markdown(fusion_html, unsafe_allow_html=True)
    time.sleep(1.0)

# ==========================================
# 6. محرك الأنيميشن الإضافي
# ==========================================
def run_full_simulation(url_text, tokenizer):
    # محاكاة بسيطة للهيكل الكامل
    chars = list(url_text)
    seq = tokenizer.texts_to_sequences([url_text])[0]
    display_len = min(len(chars), 20)
    
    step_placeholder = st.empty()
    
    # عرض سريع للمراحل دون انتظار طويل
    html_block = f"""
    <div style="display:flex; justify-content:space-around; background:#2D3748; padding:15px; border-radius:10px; margin-top:10px;">
        <div style="text-align:center; color:#63B3ED;"><b>1. Input</b><br>{url_text[:15]}...</div>
        <div style="text-align:center; color:#9F7AEA;"><b>2. Embedding</b><br>Vector Space</div>
        <div style="text-align:center; color:#48BB78;"><b>3. CNN+GRU</b><br>Feature Extraction</div>
        <div style="text-align:center; color:#F56565;"><b>4. Dense</b><br>Classification</div>
    </div>
    """
    step_placeholder.markdown(html_block, unsafe_allow_html=True)

# ==========================================
# 7. دالة رسم العداد
# ==========================================
def plot_gauge(current_value):
    score_val = int(current_value)
    if score_val < 33:
        label_text, title_color = "LOW RISK", "#4CAF50"
    elif score_val < 66:
        label_text, title_color = "MODERATE", "#FF9800"
    else:
        label_text, title_color = "SEVERE RISK", "#F44336"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<b>{label_text}</b>", 'font': {'size': 26, 'color': title_color}},
        number = {'suffix': "%", 'font': {'size': 60, 'color': "white", 'family': "Arial Black"}},
        gauge = {
            'axis': {'range': [None, 100], 'visible': False}, 
            'bar': {'color': "rgba(0,0,0,0)"}, 
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33], 'color': "#4CAF50"}, 
                {'range': [33, 66], 'color': "#FF9800"}, 
                {'range': [66, 100], 'color': "#F44336"} 
            ],
            'threshold': {'line': {'color': "black", 'width': 8}, 'thickness': 0.75, 'value': score_val}
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=400, margin=dict(l=30, r=30, t=60, b=20))
    return fig

# ==========================================
# 8. التطبيق الرئيسي (المنطق المعدل)
# ==========================================
st.title("🛡️ AI Phishing Detector: Deep Dive")
st.markdown("##### Hybrid CNN + Bi-GRU Neural Network Visualization")

url_input_raw = st.text_input("🔗 Enter URL to Analyze:", placeholder="http://example-bank-login.com")

if st.button("🚀 Analyze URL") and url_input_raw:
    if not model:
        st.error("⚠️ Model files not found! Please check the file paths in the code.")
    else:
        # 1. التجهيز والحساب الفعلي أولاً (قبل الرسم)
        url_input = format_url(url_input_raw)
        
        # استخراج البيانات
        seq, fet, _ = pre.process(url_input)
        seq = seq.reshape(1, -1)
        fet = fet.reshape(1, -1)
        fet = scaler.transform(fet)
        
        # التوقع الحقيقي
        prediction = model.predict([seq, fet], verbose=0)[0][0]
        final_score = int(prediction * 100)
        
        # ++++++++++++++++++++++++++++++++++++++++++
        # 2. تشغيل العرض التفاعلي (مع تمرير النتيجة الحقيقية لتصحيح الرسم)
        # ++++++++++++++++++++++++++++++++++++++++++
        simulate_cnn_layer(url_input, prediction) # نمرر الـ prediction هنا
        simulate_gru_layer(url_input)
        st.divider() 

        # 3. عرض النتيجة النهائية
        run_full_simulation(url_input, tokenizer)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("4. FINAL RISK ASSESSMENT")
            gauge_placeholder = st.empty()
            # أنيميشن سريع للعداد
            fig = plot_gauge(final_score)
            gauge_placeholder.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            if prediction > 0.5:
                st.error(f"### 🚨 PHISHING DETECTED")
                st.markdown(f"**Threat Level:** Severe\n\n**Confidence:** {final_score}%")
                st.info("The model detected suspicious patterns consistent with phishing attacks.")
            else:
                st.success(f"### ✅ WEBSITE IS SAFE")
                st.markdown(f"**Threat Level:** Low\n\n**Safety Score:** {100-final_score}%")
                st.info("No malicious patterns detected in the URL structure.")
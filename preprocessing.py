# ==========================================
# 1. المعالجة المتقدمة (Preprocessing & Splitting)
# استبدل خليتي "Domain-Aware Splitting" و "Tokenization" بهذا الكود
# ==========================================
import string
import re
import math
from collections import Counter
from urllib.parse import urlparse, parse_qs, urlencode, unquote
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# تهيئة المعالجة المتوازية


# --- كلاس المعالجة الجديد (من الكود المرفق) ---
class URLPreprocessor:
    def __init__(self, max_len=200):
        self.max_len = max_len
        self.chars = string.ascii_lowercase + string.digits + ".,:;/?=&-_@%#+()[]~"
        self.char2idx = {c: i + 2 for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 2
        self.TRACKING_PARAMS = {'utm_source', 'utm_medium', 'utm_campaign', 'gclid', 'fbclid'}

    def normalize_url(self, url):
        url = str(url).lower().strip()
        url = re.sub(r'^https?://', '', url)
        url = re.sub(r'^www\.', '', url)
        return url.rstrip('/')

    def entropy(self, s):
        if not s: return 0.0
        probs = [n / len(s) for n in Counter(s).values()]
        return -sum(p * math.log2(p) for p in probs)

    def is_ip_address(self, domain):
        return 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain) else 0

    def extract_manual_features(self, clean_url, raw_url):
        # استخراج خصائص رقمية إضافية لدعم الموديل
        parsed = urlparse(f"http://{clean_url}")
        domain = parsed.netloc
        path = parsed.path

        features = [
            len(clean_url),                 # طول الرابط
            len(raw_url),                   # الطول الأصلي
            self.entropy(clean_url),        # العشوائية
            self.entropy(domain),           # عشوائية الدومين
            sum(c.isdigit() for c in clean_url), # عدد الأرقام
            sum(c in "-_@%." for c in clean_url), # الرموز الخاصة
            clean_url.count('.'),
            clean_url.count('/'),
            self.is_ip_address(domain),     # هل هو IP؟
            len(path)
        ]
        return np.array(features, dtype=np.float32)

    def char_encode(self, url):
        # تحويل النص لأرقام (Sequence)
        seq = [self.char2idx.get(c, 1) for c in url]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = seq + [0] * (self.max_len - len(seq))
        return np.array(seq, dtype=np.int32)

    def process(self, url):
        # دالة واحدة ترجع كل شيء نحتاجه
        raw_url = str(url)
        clean_url = self.normalize_url(raw_url)

        # 1. للموديل (Sequence)
        seq = self.char_encode(clean_url)

        # 2. للموديل (Numeric Features)
        feats = self.extract_manual_features(clean_url, raw_url)

        # 3. للتقسيم (Domain)
        domain = clean_url.split('/')[0] if '/' in clean_url else clean_url

        return seq, feats, domain

# --- تطبيق المعالجة على البيانات ---

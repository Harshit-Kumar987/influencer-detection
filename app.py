# app.py — Instagram Influencer Detection
# Multi-class: Real / Growing / Normal / Fake Influencer

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Influencer Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf          = joblib.load('model/rf_model.pkl')
    scaler      = joblib.load('model/scaler.pkl')
    features    = joblib.load('model/feature_names.pkl')
    class_names = joblib.load('model/class_names.pkl')
    return rf, scaler, features, class_names

rf, scaler, FEATURES, class_names = load_artifacts()

# ── Tier Colors & Icons ────────────────────────────────────────
TIER_STYLE = {
    'Real Influencer'    : ('🌟', '#1565C0', 'white'),
    'Growing Influencer' : ('📈', '#2E7D32', 'white'),
    'Normal User'        : ('👤', '#F57F17', 'white'),
    'Fake Influencer'    : ('⚠️', '#B71C1C', 'white'),
}

# ── Header ─────────────────────────────────────────────────────
st.title("Influencer Detection")
st.caption("")
st.divider()

# ── Input Form ─────────────────────────────────────────────────
st.subheader("Enter Account Details")

c1, c2 = st.columns(2)

with c1:
    followers           = st.number_input(
        "Followers", 50, 2_000_000,
        value=10_000, step=1000)
    following           = st.number_input(
        "Following", 30, 50_000,
        value=500, step=100)
    posts               = st.number_input(
        "Total Posts", 1, 50_000,
        value=100, step=10)
    avg_posts_per_day   = st.number_input(
        "Avg Posts Per Day", 0.0, 100.0,
        value=1.0, step=0.1)
    avg_views_per_post  = st.number_input(
        "Avg Views Per Post", 0, 5_000_000,
        value=5000, step=100)
    account_age_months  = st.number_input(
        "Account Age (Months)", 1.0, 100.0,
        value=24.0, step=0.5)

with c2:
    avg_likes_per_post    = st.number_input(
        "Avg Likes Per Post", 0, 1_000_000,
        value=500, step=100)
    avg_comments_per_post = st.number_input(
        "Avg Comments Per Post", 0, 500_000,
        value=50, step=10)
    avg_shares_per_post   = st.number_input(
        "Avg Shares Per Post", 0, 500_000,
        value=30, step=10)

st.divider()

# ── Predict Button ─────────────────────────────────────────────
if st.button("Detect Influencer Type", type="primary",
             use_container_width=True):

    # Engineer features
    engagement_rate = (
        (avg_likes_per_post +
         avg_comments_per_post +
         avg_shares_per_post)
        / (followers + 1)
    ) * 100

    follow_ratio         = followers / (following + 1)
    reach_score          = np.log1p(followers)
    likes_views_ratio    = avg_likes_per_post / (avg_views_per_post + 1)
    comments_likes_ratio = avg_comments_per_post / (avg_likes_per_post + 1)

    input_data = np.array([[
        followers, following, posts,
        avg_posts_per_day, avg_views_per_post,
        avg_likes_per_post, avg_comments_per_post,
        avg_shares_per_post, account_age_months,
        engagement_rate, follow_ratio, reach_score,
        likes_views_ratio, comments_likes_ratio
    ]])

    input_scaled  = scaler.transform(input_data)
    prediction    = rf.predict(input_scaled)[0]
    probabilities = rf.predict_proba(input_scaled)[0]

    predicted_label = class_names[prediction]
    confidence      = max(probabilities) * 100

    # Status mapping
    STATUS = {
        'Real Influencer'    : 'Influencer',
        'Growing Influencer' : 'Growing Influencer',
        'Normal User'        : 'Not Influencer',
        'Fake Influencer'    : 'Fake / Bot Account',
    }

    st.divider()
    st.markdown("### Result")
    st.markdown(f"**User Type:** {predicted_label}")
    st.markdown(f"**Status:** {STATUS[predicted_label]}")
    st.markdown(f"**Confidence:** {confidence:.1f}%")
    st.markdown(f"**Engagement Rate:** {engagement_rate:.2f}%")
    st.markdown(f"**Views Ratio:** {likes_views_ratio:.3f}")
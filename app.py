# app.py — Instagram Influencer Detection
# Deploy: streamlit.io

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              roc_curve, roc_auc_score,
                              accuracy_score, f1_score)
from sklearn.model_selection import train_test_split

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Influencer Detection",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Artifacts ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf       = joblib.load('model/rf_model.pkl')
    lr       = joblib.load('model/lr_model.pkl')
    scaler   = joblib.load('model/scaler.pkl')
    features = joblib.load('model/feature_names.pkl')
    cat_map  = joblib.load('model/category_mapping.pkl')
    return rf, lr, scaler, features, cat_map

@st.cache_data
def load_data():
    return pd.read_csv('data/influencers_scored.csv')

rf, lr, scaler, FEATURES, cat_map = load_artifacts()
df = load_data()

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("Filters")

tiers = st.sidebar.multiselect(
    "Influencer Tier",
    options=['Mega', 'Macro', 'Micro', 'Nano', 'Non-Influencer'],
    default=['Mega', 'Macro', 'Micro', 'Nano']
)

min_score = st.sidebar.slider(
    "Minimum Influencer Score", 0, 100, 20
)

countries = ['All'] + sorted(
    df['audience_country'].dropna().unique().tolist()
)
country = st.sidebar.selectbox("Audience Country", countries)

categories = ['All'] + sorted(
    df['category'].dropna().unique().tolist()
)
category = st.sidebar.selectbox("Category", categories)

top_n = st.sidebar.slider("Show Top N", 5, 200, 20)

# ── Apply Filters ──────────────────────────────────────────────
filtered = df[
    df['tier'].isin(tiers) &
    (df['influencer_score'] >= min_score)
].copy()

if country != 'All':
    filtered = filtered[filtered['audience_country'] == country]
if category != 'All':
    filtered = filtered[filtered['category'] == category]

filtered = filtered.sort_values(
    'influencer_score', ascending=False
).head(top_n)

# ── Header ─────────────────────────────────────────────────────
st.title("📸 Instagram Influencer Detection")
st.caption("B.Tech 4th Semester · ML Project · "
           "Random Forest vs Logistic Regression")
st.divider()

# ── KPI Cards ──────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Results",       len(filtered))
k2.metric("Avg Score",     f"{filtered['influencer_score'].mean():.1f}")
k3.metric("Avg Eng. Rate", f"{filtered['engagement_rate'].mean():.3f}%")
k4.metric("Avg Followers", f"{filtered['followers'].mean()/1e6:.2f}M")
k5.metric("Categories",    filtered['category'].nunique())
st.divider()

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Leaderboard",
    "📊 Analytics",
    "🤖 Model Performance",
    "🔍 Predict"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Influencer Rankings")

    display_cols = ['rank', 'name', 'category', 'audience_country',
                    'followers', 'engagement_rate',
                    'influencer_score', 'tier']
    display_cols = [c for c in display_cols if c in filtered.columns]

    def color_tier(val):
        colors = {
            'Mega'          : 'background-color: #1565C0; color: white',
            'Macro'         : 'background-color: #1976D2; color: white',
            'Micro'         : 'background-color: #42A5F5; color: white',
            'Nano'          : 'background-color: #90CAF9; color: #0D47A1',
            'Non-Influencer': 'background-color: #EEEEEE; color: #616161'
        }
        return colors.get(val, '')

    styled = (
        filtered[display_cols]
        .reset_index(drop=True)
        .style
        .map(color_tier, subset=['tier'])
        .format({
            'followers'       : '{:,.0f}',
            'engagement_rate' : '{:.3f}%',
            'influencer_score': '{:.1f}'
        })
    )
    st.dataframe(styled, use_container_width=True)

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇ Download Results as CSV",
        data=csv,
        file_name='influencers.csv',
        mime='text/csv'
    )

# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Tier Distribution**")
        tier_counts = df['tier'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            tier_counts,
            labels=tier_counts.index,
            autopct='%1.1f%%',
            colors=['#0D47A1', '#1565C0', '#42A5F5',
                    '#90CAF9', '#E3F2FD'],
            startangle=140
        )
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Top 10 Audience Countries**")
        top_countries = df['audience_country'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        top_countries.plot(kind='barh', ax=ax,
                           color='#42A5F5', edgecolor='white')
        ax.invert_yaxis()
        ax.set_xlabel('Count')
        st.pyplot(fig)
        plt.close()

    st.markdown("**Top Categories by Avg Influencer Score**")
    cat_scores = (
        df.groupby('category')['influencer_score']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    cat_scores.plot(kind='bar', ax=ax,
                    color='#1565C0', edgecolor='white')
    ax.set_xlabel('Category')
    ax.set_ylabel('Avg Influencer Score')
    ax.set_title('Top Categories by Avg Score')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Engagement Rate vs Followers**")
    fig, ax = plt.subplots(figsize=(9, 4))
    sc = ax.scatter(
        np.log1p(df['followers']),
        df['engagement_rate'],
        c=df['influencer_score'],
        cmap='Blues', alpha=0.7, s=30
    )
    plt.colorbar(sc, ax=ax, label='Influencer Score')
    ax.set_xlabel('Log(Followers)')
    ax.set_ylabel('Engagement Rate (%)')
    ax.set_title('Reach vs Engagement (colored by Score)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance Comparison")
    st.info("Models trained on 80% data, evaluated on 20% holdout set.")

    X_all = scaler.transform(df[FEATURES])
    y_true = df['is_top_influencer']

    _, X_test_s, _, y_test_s = train_test_split(
        X_all, y_true,
        test_size=0.2,
        random_state=42,
        stratify=y_true
    )

    y_pred_rf  = rf.predict(X_test_s)
    y_pred_lr  = lr.predict(X_test_s)
    y_proba_rf = rf.predict_proba(X_test_s)[:, 1]
    y_proba_lr = lr.predict_proba(X_test_s)[:, 1]

    # Metrics table
    metrics_data = {
        'Model'    : ['Random Forest', 'Logistic Regression'],
        'Accuracy' : [f"{accuracy_score(y_test_s, y_pred_rf):.4f}",
                      f"{accuracy_score(y_test_s, y_pred_lr):.4f}"],
        'F1 Score' : [f"{f1_score(y_test_s, y_pred_rf):.4f}",
                      f"{f1_score(y_test_s, y_pred_lr):.4f}"],
        'ROC-AUC'  : [f"{roc_auc_score(y_test_s, y_proba_rf):.4f}",
                      f"{roc_auc_score(y_test_s, y_proba_lr):.4f}"]
    }
    st.dataframe(
        pd.DataFrame(metrics_data),
        use_container_width=True
    )

    m1, m2 = st.columns(2)

    with m1:
        st.markdown("**Confusion Matrices**")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, pred, title in zip(
            axes,
            [y_pred_rf, y_pred_lr],
            ['Random Forest', 'Logistic Regression']
        ):
            cm = confusion_matrix(y_test_s, pred)
            ConfusionMatrixDisplay(
                cm, display_labels=['Moderate', 'Top']
            ).plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(title)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with m2:
        st.markdown("**ROC Curve**")
        fig, ax = plt.subplots(figsize=(5, 4))
        for proba, name, color in [
            (y_proba_rf, 'Random Forest',       '#1565C0'),
            (y_proba_lr, 'Logistic Regression', '#90CAF9')
        ]:
            fpr, tpr, _ = roc_curve(y_test_s, proba)
            auc = roc_auc_score(y_test_s, proba)
            ax.plot(fpr, tpr,
                    label=f'{name} (AUC={auc:.3f})',
                    color=color)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Feature Importance (Random Forest)**")
    importances = pd.Series(
        rf.feature_importances_, index=FEATURES
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ['#1565C0' if v == importances.max()
              else '#90CAF9' for v in importances]
    importances.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Importance')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Predict — New Instagram Account")
    st.markdown("Enter account details to get the influencer score.")

    c1, c2, c3 = st.columns(3)

    with c1:
        inp_rank      = st.number_input(
            "World Rank", 1, 10000, value=250)
        inp_followers = st.number_input(
            "Followers", 1000, 500_000_000,
            value=1_000_000, step=100_000)

    with c2:
        inp_auth_eng = st.number_input(
            "Authentic Engagement", 0, 50_000_000,
            value=50_000, step=1000)
        inp_eng_avg  = st.number_input(
            "Engagement Avg.", 0, 50_000_000,
            value=45_000, step=1000)

    with c3:
        inp_category = st.selectbox(
            "Category",
            options=list(cat_map.keys())
        )
        model_choice = st.radio(
            "Model to use",
            options=['Random Forest', 'Logistic Regression']
        )

    if st.button("Get Influencer Score", type="primary"):
        eng_rate    = (inp_auth_eng / (inp_followers + 1)) * 100
        reach       = np.log1p(inp_followers)
        rank_s      = 1.0 / inp_rank
        consistency = min(inp_eng_avg / (inp_auth_eng + 1), 5.0)
        cat_enc     = cat_map.get(inp_category, 0)

        input_arr    = np.array([[eng_rate, reach, rank_s,
                                   consistency, cat_enc]])
        input_scaled = scaler.transform(input_arr)

        model = rf if model_choice == 'Random Forest' else lr
        prob  = model.predict_proba(input_scaled)[0][1]
        score = round(prob * 100, 2)

        tier = ('Mega'           if score >= 80 else
                'Macro'          if score >= 60 else
                'Micro'          if score >= 40 else
                'Nano'           if score >= 20 else
                'Non-Influencer')

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Influencer Score", f"{score} / 100")
        r2.metric("Tier",             tier)
        r3.metric("Engagement Rate",  f"{eng_rate:.3f}%")
        st.progress(int(score))
        st.caption(f"Predicted using: {model_choice}")
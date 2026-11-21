import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# 1. PAGE CONFIG & AESTHETIC STYLING (CUSTOM CSS)

st.set_page_config(
    page_title="The Censorship Algorithm",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        letter-spacing: -0.02em;
    }

    /* Clean metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    /* Text X-Ray box */
    .xray-box {
        line-height: 1.9;
        font-size: 15.5px;
        padding: 20px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 16px;
    }

    .tag-banned {
        background-color: rgba(244, 63, 94, 0.25);
        border-bottom: 2px solid #f43f5e;
        padding: 1px 5px;
        border-radius: 4px;
        font-weight: 500;
    }

    .tag-safe {
        background-color: rgba(56, 189, 248, 0.22);
        border-bottom: 2px solid #38bdf8;
        padding: 1px 5px;
        border-radius: 4px;
        font-weight: 500;
    }

    .path-box {
        padding: 16px 20px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 15px;
        line-height: 1.8;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# 2. DATA LOADING, MODEL TRAINING & STATIC VISUALS (CACHED)

@st.cache_resource
def load_and_train():
    df = pd.read_csv("merged_dataset.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["description", "genre", "banned"]).copy()
    df["banned"] = df["banned"].astype(int)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["banned"]
    )

    # Model 1: Logistic Regression (TF-IDF on Book Descriptions)
    tfidf = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    X_train_text = tfidf.fit_transform(train_df["description"])
    X_test_text = tfidf.transform(test_df["description"])
    log_reg = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train_text, train_df["banned"])
    lr_acc = log_reg.score(X_test_text, test_df["banned"])

    # Model 2: Decision Tree (Binary Genre Flags)
    genre_vec = CountVectorizer(binary=True, max_features=40, stop_words="english")
    X_train_genre = genre_vec.fit_transform(train_df["genre"])
    X_test_genre = genre_vec.transform(test_df["genre"])
    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
    tree.fit(X_train_genre, train_df["banned"])
    tree_acc = tree.score(X_test_genre, test_df["banned"])

    return df, tfidf, log_reg, lr_acc, genre_vec, tree, tree_acc

df, tfidf, log_reg, lr_acc, genre_vec, tree, tree_acc = load_and_train()
feature_names = np.array(tfidf.get_feature_names_out())
coefs = log_reg.coef_[0]
word_to_coef = dict(zip(feature_names, coefs))

@st.cache_resource
def build_static_visuals():
    # Global Logistic Regression weights chart
    top_pos = np.argsort(coefs)[-15:]
    top_neg = np.argsort(coefs)[:15]
    top_all = np.concatenate([top_neg, top_pos])

    global_df = pd.DataFrame({
        "Word": feature_names[top_all],
        "Weight": coefs[top_all],
        "Direction": ["Banned (+)" if w > 0 else "Not Banned (-)" for w in coefs[top_all]]
    }).sort_values("Weight")

    fig_global = px.bar(
        global_df, x="Weight", y="Word", orientation="h", color="Direction",
        color_discrete_map={"Banned (+)": "#f43f5e", "Not Banned (-)": "#38bdf8"},
        height=600
    )
    fig_global.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # Visual Decision Tree (Graphviz)
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=list(genre_vec.get_feature_names_out()),
        class_names=["Not Banned", "Banned"],
        filled=True,
        rounded=True,
        special_characters=True,
        impurity=False,
        proportion=True
    ).replace(
        "digraph Tree {",
        'digraph Tree {\n  bgcolor="transparent";\n  node [fontname="Inter", fontsize=11, penwidth=0];\n  edge [fontname="Inter", fontsize=10, color="#6b7280", fontcolor="#d1d5db"];'
    )
    return fig_global, dot_data

fig_global, dot_data = build_static_visuals()


# 3. HEADER & FAST SHARED BOOK SELECTOR

st.markdown("# The Censorship Algorithm")
st.markdown(
    f"<p style='color:#9ca3af; font-size:16px; margin-top:-10px;'>"
    f"Interactive Machine Learning X-Ray trained on <b>{len(df):,} books</b> · "
    f"Logistic Regression: <b>{lr_acc*100:.1f}%</b> · Decision Tree: <b>{tree_acc*100:.1f}%</b>"
    f"</p>",
    unsafe_allow_html=True
)

st.write("")
col_search, col_select, col_info = st.columns([1.2, 1.8, 1])

with col_search:
    search_query = st.text_input("🔎 Search book title:", value="Fahrenheit 451")

if search_query.strip():
    filtered_titles = df[df["title"].str.contains(search_query, case=False, na=False)]["title"].head(30).tolist()
else:
    filtered_titles = df["title"].head(30).tolist()

if not filtered_titles:
    st.warning("No books found matching that title. Try a shorter keyword.")
    st.stop()

with col_select:
    selected_title = st.selectbox("Select match (top 30):", options=filtered_titles)

book_row = df[df["title"] == selected_title].iloc[0]

with col_info:
    actual_label = "🚫 BANNED" if book_row["banned"] == 1 else "✅ NOT BANNED"
    st.metric("Actual Dataset Label", actual_label)

st.write("")
tab_lr, tab_tree = st.tabs([
    f" Logistic Regression · Text ({lr_acc*100:.1f}%)",
    f" Decision Tree · Genre ({tree_acc*100:.1f}%)"
])


# TAB 1: LOGISTIC REGRESSION (LINEAR MODEL ON SYNOPSIS)

with tab_lr:
    st.write("")
    user_text = st.text_area(
        "Live Synopsis Editor (try adding or deleting words like 'classic', 'future', 'burned'):",
        value=book_row["description"],
        height=115
    )

    # Real-time Logistic Regression prediction
    vec = tfidf.transform([user_text])
    z_score = float(log_reg.decision_function(vec)[0])
    prob = float(log_reg.predict_proba(vec)[0][1])
    lr_verdict = "🚫 BANNED" if prob >= 0.5 else "✅ NOT BANNED"

    st.write("")
    k1, k2, k3 = st.columns(3)
    k1.metric("Ban Probability", f"{prob*100:.1f}%")
    k2.metric("Linear Score (z = w·x + b)", f"{z_score:+.3f}")
    k3.metric("Logistic Reg. Verdict", lr_verdict)

    st.write("")
    left_col, right_col = st.columns([1.15, 1], gap="large")

    with left_col:
        st.markdown("### Word-by-Word X-Ray")
        st.caption("🔴 Red pushes toward **Banned** · 🔵 Blue pushes toward **Not Banned**")

        tokens = re.findall(r"\w+|\W+", user_text)
        highlighted_html = []
        for tok in tokens:
            clean_tok = tok.lower()
            if clean_tok in word_to_coef:
                w = word_to_coef[clean_tok]
                if w > 0.35:
                    highlighted_html.append(f"<span class='tag-banned' title='Weight: +{w:.2f}'>{tok}</span>")
                elif w < -0.35:
                    highlighted_html.append(f"<span class='tag-safe' title='Weight: {w:.2f}'>{tok}</span>")
                else:
                    highlighted_html.append(tok)
            else:
                highlighted_html.append(tok)

        st.markdown(f"<div class='xray-box'>{''.join(highlighted_html)}</div>", unsafe_allow_html=True)

        # Local feature contributions chart
        contributions = vec.toarray()[0] * coefs
        non_zero = np.where(contributions != 0)[0]
        if len(non_zero) > 0:
            top_idx = non_zero[np.argsort(np.abs(contributions[non_zero]))[::-1][:8]]
            contrib_df = pd.DataFrame({
                "Word": feature_names[top_idx],
                "Impact": contributions[top_idx],
                "Color": ["#f43f5e" if v > 0 else "#38bdf8" for v in contributions[top_idx]]
            }).sort_values("Impact")

            fig_local = go.Figure(go.Bar(
                x=contrib_df["Impact"],
                y=contrib_df["Word"],
                orientation="h",
                marker_color=contrib_df["Color"],
                marker_line_width=0
            ))
            fig_local.update_layout(
                title="Top Math Contributions for This Synopsis",
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.2)"),
                yaxis=dict(showgrid=False),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_local, use_container_width=True)

    with right_col:
        st.markdown("### Position on the Sigmoid Curve")
        st.caption("The sigmoid function maps the linear score $z$ to a probability between 0% and 100%.")

        z_range = np.linspace(-6, 6, 300)
        sig_curve = 1 / (1 + np.exp(-z_range))
        dot_color = "#f43f5e" if prob >= 0.5 else "#38bdf8"

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            x=z_range, y=sig_curve, mode="lines",
            line=dict(color="rgba(156, 163, 175, 0.7)", width=3),
            name="Sigmoid σ(z)"
        ))
        fig_sig.add_hline(
            y=0.5, line_dash="dot", line_color="#f59e0b",
            annotation_text="Decision Boundary (50%)",
            annotation_font_color="#f59e0b"
        )
        fig_sig.add_trace(go.Scatter(
            x=[np.clip(z_score, -5.8, 5.8)], y=[prob],
            mode="markers+text",
            marker=dict(size=18, color=dot_color, line=dict(width=3, color="white")),
            text=[f"  {prob*100:.1f}%"],
            textposition="middle right",
            textfont=dict(size=14, color=dot_color),
            name="Current Book"
        ))
        fig_sig.update_layout(
            xaxis_title="Linear Score z = Σ(wᵢ·xᵢ) + b",
            yaxis_title="Ban Probability",
            yaxis=dict(range=[-0.05, 1.05], showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=460,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    st.divider()
    st.markdown("### Global Weights: Top 15 Influential Words")
    st.caption("Learned across all 13,000+ training synopses.")
    st.plotly_chart(fig_global, use_container_width=True)

# TAB 2: DECISION TREE (GENRE RULES MODEL)
with tab_tree:
    st.write("")

    # Real-time Decision Tree prediction
    genre_vec_book = genre_vec.transform([book_row["genre"]])
    tree_prob = float(tree.predict_proba(genre_vec_book)[0][1])
    tree_verdict = "🚫 BANNED" if tree_prob >= 0.5 else "✅ NOT BANNED"

    t1, t2, t3 = st.columns(3)
    t1.metric("Tree Ban Probability", f"{tree_prob*100:.1f}%")
    t2.metric("Decision Tree Verdict", tree_verdict)
    t3.metric("Tree Depth", "3 Levels")

    st.write("")
    p_left, p_right = st.columns([1, 1], gap="large")

    with p_left:
        st.markdown("### Book's Genre Tags")
        st.caption("Raw genre metadata fed into the Decision Tree:")
        st.markdown(f"<div class='path-box'>{book_row['genre']}</div>", unsafe_allow_html=True)

    with p_right:
        st.markdown("### Exact Decision Path Taken")
        st.caption("Step-by-step If/Else rules this book triggered:")

        node_indicator = tree.decision_path(genre_vec_book)
        leaf_id = tree.apply(genre_vec_book)[0]
        feat_indices = tree.tree_.feature
        thresholds = tree.tree_.threshold
        genre_names = genre_vec.get_feature_names_out()
        sample_arr = genre_vec_book.toarray()[0]

        steps_html = []
        for step_num, node_id in enumerate(node_indicator.indices, start=1):
            if node_id == leaf_id:
                steps_html.append(f"<b>Result:</b> Landed in leaf → <b>{tree_verdict} ({tree_prob*100:.1f}%)</b>")
            else:
                g_name = genre_names[feat_indices[node_id]]
                has_tag = sample_arr[feat_indices[node_id]] > thresholds[node_id]
                answer = "<span class='tag-banned'>YES (Present)</span>" if has_tag else "<span class='tag-safe'>NO (Absent)</span>"
                steps_html.append(f"<b>Step {step_num}:</b> Contains genre <code>{g_name}</code>? → {answer}")

        st.markdown(f"<div class='path-box'>{'<br>'.join(steps_html)}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Full Visual Decision Tree")
    st.caption("Hierarchical If/Else rules learned across all genres. Orange = Not Banned · Blue = Banned.")
    st.graphviz_chart(dot_data, use_container_width=True)
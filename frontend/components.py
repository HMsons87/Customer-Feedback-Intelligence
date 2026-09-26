import base64
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def get_navbar_logo_html() -> str:
    logo_file = Path(__file__).parent / "assets" / "logo.png"
    if logo_file.exists():
        encoded = base64.b64encode(logo_file.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{encoded}" style="width: 26px; height: 26px; object-fit: contain;" />'
    return '<span style="font-weight: 800; color: #ffffff; font-size: 15px; letter-spacing: -0.5px;">IQ</span>'


def render_top_navbar(is_online: bool):
    status_indicator = (
        '<div class="pulse-dot"></div><span style="color: #4ade80;">Engine'
        " Connected</span>"
        if is_online
        else '<div class="pulse-dot-red"></div><span style="color: #f87171;">Engine'
             " Offline</span>"
    )

    st.markdown(
        f"""
        <div class="top-navbar">
            <div class="brand-wrapper">
                <div class="brand-logo">
                    {get_navbar_logo_html()}
                </div>
                <div class="brand-text">
                    <h2>FeedbackIQ</h2>
                    <span>CUSTOMER INTELLIGENCE SUITE</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div class="nav-status">
                    {status_indicator}
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_hero_section():
    st.markdown(
        """
          <div class="hero-banner">
              <div class="hero-title">Transforming Reviews into Strategic Decisions</div>
              <div class="hero-desc">
                  Enterprise Natural Language Processing engine for real-time customer sentiment inference, automated complaint clustering, and actionable churn mitigation.
              </div>
          </div>
      """,
        unsafe_allow_html=True,
    )


def render_kpi_summary():
    st.markdown(
        """
          <div class="kpi-row">
              <div class="kpi-box">
                  <div class="kpi-tag">Analyzed Reviews</div>
                  <div class="kpi-num">22,627</div>
                  <div class="kpi-trend">100% Cleansed Data</div>
              </div>
              <div class="kpi-box">
                  <div class="kpi-tag">Brand Advocacy (CSAT)</div>
                  <div class="kpi-num" style="color: #4ade80;">77.1%</div>
                  <div class="kpi-trend">Positive Customer Base</div>
              </div>
              <div class="kpi-box">
                  <div class="kpi-tag">Actionable Attrition Risk</div>
                  <div class="kpi-num" style="color: #f87171;">10.5%</div>
                  <div class="kpi-trend" style="color: #f87171;">Negative Issues Identified</div>
              </div>
              <div class="kpi-box">
                  <div class="kpi-tag">Dominant Risk Driver</div>
                  <div class="kpi-num" style="color: #fbbf24; font-size: 1.4rem; margin-top: 8px;">Size & Fit Defects</div>
                  <div class="kpi-trend" style="color: #fbbf24;">41% of All Complaints</div>
              </div>
          </div>
      """,
        unsafe_allow_html=True,
    )


def render_sentiment_card(result: dict):
    sentiment = result.get("sentiment", "Neutral")
    confidence = result.get("confidence", 0.0) * 100

    badge_class = (
        "badge-pos"
        if sentiment == "Positive"
        else "badge-neu"
        if sentiment == "Neutral"
        else "badge-neg"
    )

    st.markdown(
        f"""
        <div class="dashboard-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Sentiment Classification</span>
                <span class="badge-tag {badge_class}">{sentiment.upper()}</span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 16px;">
                <span style="font-size: 2.2rem; font-weight: 800; color: #ffffff;">{confidence:.1f}%</span>
                <span style="color: #64748b; font-size: 0.9rem;">Confidence Score</span>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    probs = result.get(
        "probabilities", {"Negative": 0.1, "Neutral": 0.2, "Positive": 0.7}
    )
    df_probs = pd.DataFrame(list(probs.items()), columns=["Class", "Score"])

    fig = go.Figure(
        go.Bar(
            x=df_probs["Score"],
            y=df_probs["Class"],
            orientation="h",
            marker=dict(
                color=["#f87171", "#fbbf24", "#4ade80"],
                line=dict(color="rgba(255, 255, 255, 0.05)", width=1),
            ),
            text=[f"{v * 100:.1f}%" for v in df_probs["Score"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#0b0f19", size=12, weight="bold"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Class Probability Distribution",
            font=dict(color="#94a3b8", size=12),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(l=0, r=20, t=30, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(
            tickfont=dict(color="#94a3b8", size=11),
            categoryorder="array",
            categoryarray=["Positive", "Neutral", "Negative"],
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_explainability_card(key_drivers: list):
    if not key_drivers:
        return

    st.markdown(
        """
          <div class="dashboard-card" style="border-left: 4px solid #6366f1;">
              <div style="font-size: 0.8rem; color: #a5b4fc; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">
                  Explainable AI (XAI) • Model Decision Drivers
              </div>
              <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 12px;">
                  Tokens that heavily influenced the model's prediction:
              </p>
          </div>
      """,
        unsafe_allow_html=True,
    )

    df_drivers = pd.DataFrame(key_drivers)
    df_drivers["Color"] = df_drivers["score"].apply(
        lambda s: "#4ade80" if s > 0 else "#f87171"
    )

    fig = go.Figure(
        go.Bar(
            x=df_drivers["score"],
            y=df_drivers["word"],
            orientation="h",
            marker=dict(color=df_drivers["Color"]),
            text=[f"{s:+.2f}" for s in df_drivers["score"]],
            textposition="outside",
            textfont=dict(color="#cbd5e1", size=11),
        )
    )

    fig.update_layout(
        title=dict(
            text="Token-Level Attribution Weights",
            font=dict(color="#94a3b8", size=12),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=10, r=40, t=30, b=10),
        xaxis=dict(
            showgrid=True,
            gridcolor="#1e293b",
            zeroline=True,
            zerolinecolor="#64748b",
        ),
        yaxis=dict(tickfont=dict(color="#e2e8f0", size=11, family="monospace")),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_complaint_cluster(cluster_info: dict):
    """عرض كرت تحليل الشكاوى بدون إيموجي"""
    if not cluster_info:
        return

    category = cluster_info.get("category_name", "General Anomaly")
    keywords = cluster_info.get("top_keywords", [])
    chips_html = "".join(
        [f'<span class="semantic-chip">{k}</span>' for k in keywords]
    )

    actions = {
        "Fit & Sizing Issues": (
            "Update garment dimensional specs on web portal, publish customer size"
            " guidance, and review vendor garment cuts."
        ),
        "Fabric & Material Quality": (
            "Audit supplier fabric batch thickness, inspect thread"
            " tear-resistance, and verify wash colorfastness."
        ),
        "Design & Construction Flaws": (
            "Notify QA engineering on zipper seam resilience and stress-point"
            " reinforcement."
        ),
        "Return & Order Fulfillment": (
            "Optimize customer return logistics and establish shipping parcel"
            " inspection."
        ),
    }
    action_text = actions.get(
        category,
        "Review customer care communication and offer return resolution.",
    )

    st.markdown(
        f"""
        <div class="dashboard-card" style="border-left: 4px solid #f87171;">
            <div style="font-size: 0.8rem; color: #f87171; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">
                Customer Pain-Point Identified (K-Means)
            </div>
            <h4 style="color: #ffffff; font-weight: 700; margin-bottom: 12px;">{category}</h4>

            <p style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">Extracted Root-Cause Keywords:</p>
            <div style="margin-bottom: 14px;">{chips_html}</div>

            <div style="background: rgba(248, 113, 113, 0.05); border: 1px solid rgba(248, 113, 113, 0.15); border-radius: 8px; padding: 12px;">
                <div style="color: #f87171; font-weight: 600; font-size: 0.8rem; margin-bottom: 2px;">Recommended Operational Action:</div>
                <div style="color: #cbd5e1; font-size: 0.85rem;">{action_text}</div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_eda_section(
        data_path: str = "../data/processed/cleaned_reviews.csv",
):
    """لوحة التحليلات الاستكشافية"""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.warning(
            "Processed dataset not found at"
            " `../data/processed/cleaned_reviews.csv`."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        fig_rating = px.histogram(
            df,
            x="Rating",
            color="Rating",
            title="Customer Ratings Histogram",
            color_discrete_sequence=[
                "#ef4444",
                "#f97316",
                "#eab308",
                "#84cc16",
                "#22c55e",
            ],
        )
        fig_rating.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rating, use_container_width=True)

    with col2:
        dept_counts = df["Department Name"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Reviews Count"]
        fig_dept = px.pie(
            dept_counts,
            names="Department",
            values="Reviews Count",
            title="Feedback Volume by Merchandising Department",
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_dept.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dept, use_container_width=True)


def render_site_footer():
    """تذييل الموقع المؤسسي"""
    st.markdown(
        """
          <div class="site-footer">
              <div>© 2026 FeedbackIQ Platform • Information Technology Institute (ITI) Final Project</div>
              <div style="display: flex; gap: 15px;">
                  <span>FastAPI Microservice</span>
                  <span>•</span>
                  <span>Scikit-Learn Pipeline</span>
                  <span>•</span>
                  <span>Streamlit Cloud</span>
              </div>
          </div>
      """,
        unsafe_allow_html=True,
    )
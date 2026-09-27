from pathlib import Path
from api_client import FeedbackAPIClient
from components import (
    render_complaint_cluster,
    render_eda_section,
    render_explainability_card,
    render_hero_section,
    render_kpi_summary,
    render_sentiment_card,
    render_site_footer,
    render_top_navbar,
)
import streamlit as st

logo_icon_path = Path(__file__).parent / "assets" / "logo.png"
fav_icon = str(logo_icon_path) if logo_icon_path.exists() else None

st.set_page_config(
    page_title="FeedbackIQ • Customer Feedback Intelligence",
    page_icon=fav_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
  with open(css_path, "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

client = FeedbackAPIClient()
is_online = client.check_health()

render_top_navbar(is_online)

render_hero_section()

render_kpi_summary()

tab1, tab2 = st.tabs(
    ["Live Customer Feedback Inspector", "Executive Dataset Analytics"]
)

with tab1:
  st.markdown(
      """
        <div style="margin: 15px 0 10px 0;">
            <h3 style="color: #ffffff; font-size: 1.3rem; font-weight: 700; margin-bottom: 4px;">Live Inference Playground</h3>
            <p style="color: #64748b; font-size: 0.9rem; margin: 0;">Analyze customer satisfaction, detect negative risk signals, and extract root-cause complaint clusters.</p>
        </div>
    """,
      unsafe_allow_html=True,
  )

  st.markdown(
      "<span style='color: #64748b; font-size: 0.8rem; font-weight: 600;'>Load"
      " Customer Review Template:</span>",
      unsafe_allow_html=True,
  )
  preset_cols = st.columns([1, 1, 1])
  preset_text = (
      "I loved the fabric and color, but it was way too small around the waist"
      " and tore easily."
  )

  # أزرار القوالب بعد إزالة الـ Emojis
  if preset_cols[0].button("Defect: Sizing & Quality", use_container_width=True):
    preset_text = (
        "The skirt was outrageously small in the waist and the zipper broke"
        " after first wear."
    )
  if preset_cols[1].button("Positive: Brand Advocate", use_container_width=True):
    preset_text = (
        "Absolutely gorgeous dress! Soft fabric, fits true to size, and received"
        " tons of compliments."
    )
  if preset_cols[2].button(
      "Issue: Logistics & Return", use_container_width=True
  ):
    preset_text = (
        "Waited 3 weeks for delivery. Arrived damaged and customer service"
        " refused to process exchange."
    )

  meta_col1, meta_col2 = st.columns(2)
  with meta_col1:
    selected_dept = st.selectbox(
        "Garment Department:",
        ["Dresses", "Tops", "Bottoms", "Intimate", "Jackets", "Trend"],
    )
  with meta_col2:
    customer_age = st.slider(
        "Customer Age:", min_value=18, max_value=85, value=35
    )

  user_input = st.text_area(
      label="Customer Review Text",
      value=preset_text,
      height=110,
      placeholder="Type or paste customer feedback here...",
      label_visibility="collapsed",
  )

  col_btn, _ = st.columns([1, 3])
  with col_btn:
    analyze_clicked = st.button(
        "Run AI Diagnostics", type="primary", use_container_width=True
    )

  if analyze_clicked:
    if not user_input.strip():
      st.warning("Please enter a customer review.")
    else:
      with st.spinner("Executing NLP Pipeline & Model Vectorization..."):
        res, err = client.analyze_feedback(
            text=user_input, age=customer_age, department=selected_dept
        )

        if res:
          st.markdown(
              "<div style='height: 15px;'></div>", unsafe_allow_html=True
          )
          col1, col2 = st.columns([1, 1], gap="medium")
          with col1:
            render_sentiment_card(res)
            if res.get("key_drivers"):
              render_explainability_card(res["key_drivers"])
          with col2:
            if res.get("is_complaint") and res.get("complaint_cluster"):
              render_complaint_cluster(res["complaint_cluster"])
            else:
              st.markdown(
                  """
                        <div class="dashboard-card" style="border-left: 4px solid #4ade80;">
                            <div style="color: #4ade80; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">
                                Brand Satisfaction Confirmed
                            </div>
                            <h4 style="color: #ffffff; font-weight: 700; margin-bottom: 10px;">Positive / Neutral Sentiment</h4>
                            <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5; margin: 0;">
                                No structural complaint patterns or quality defects detected. Review represents satisfied consumer advocacy.
                            </p>
                        </div>
                    """,
                  unsafe_allow_html=True,
              )
        else:
          st.error(f"Inference Service Failed: {err}")

with tab2:
  st.markdown(
      """
        <div style="margin: 15px 0 10px 0;">
            <h3 style="color: #ffffff; font-size: 1.3rem; font-weight: 700; margin-bottom: 4px;">Executive Performance Dashboard</h3>
            <p style="color: #64748b; font-size: 0.9rem; margin: 0;">Aggregated voice-of-customer trends across departments and product categories.</p>
        </div>
    """,
      unsafe_allow_html=True,
  )
  render_eda_section()

render_site_footer()
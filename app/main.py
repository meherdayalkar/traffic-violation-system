# app/main.py — UI Engineer's Streamlit application
# Run: streamlit run app/main.py

import streamlit as st
import cv2
import numpy as np
import tempfile
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Traffic Violation System",
    page_icon   = "🚦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .violation-card {
        background: #1a1a2e;
        border: 1px solid #e63946;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .challan-card {
        background: #0f3460;
        border: 1px solid #16213e;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .stat-box {
        background: #16213e;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .confirmed { color: #e63946; font-weight: bold; }
    .needs_review { color: #f4a261; font-weight: bold; }
    .stMetric { background: #0f3460; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
if "pipeline"    not in st.session_state: st.session_state.pipeline    = None
if "violations"  not in st.session_state: st.session_state.violations  = []
if "challans"    not in st.session_state: st.session_state.challans    = []
if "processing"  not in st.session_state: st.session_state.processing  = False
if "frame_count" not in st.session_state: st.session_state.frame_count = 0


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Police_lights_on.svg/120px-Police_lights_on.svg.png", width=80)
    except Exception:
        st.markdown("# 🚨")
    st.title("🚦 Traffic VMS")
    st.caption("Prototype — Nagpur Traffic Dept.")
    st.divider()

    st.subheader("⚙️ Configuration")
    conf_thresh = st.slider("Detection Confidence", 0.2, 0.9, config.YOLO_CONFIDENCE, 0.05)
    skip_frames = st.slider("Process every Nth frame", 1, 5, 2,
                            help="2 = process every 2nd frame (faster)")
    enable_anpr = st.toggle("Enable ANPR (plate reading)", value=True)
    show_tracks = st.toggle("Show tracking trails", value=True)

    st.divider()
    st.subheader("📊 Session Stats")
    stat1 = st.empty()
    stat2 = st.empty()
    stat3 = st.empty()
    stat4 = st.empty()

    st.divider()
    if st.button("🗑️ Reset Session", use_container_width=True):
        for key in ["pipeline","violations","challans","frame_count"]:
            st.session_state[key] = [] if key in ["violations","challans"] else None
            if key == "frame_count":
                st.session_state[key] = 0
        st.rerun()


# ─── Main layout ──────────────────────────────────────────────────────────────
st.title("🚦 Traffic Violation Detection System")
st.caption(f"Model: YOLOv8 + SORT Tracker | Device: {config.DEVICE.upper()} | {datetime.now().strftime('%d %b %Y')}")

tab_live, tab_challans, tab_evidence, tab_export = st.tabs([
    "📹 Live Detection", "📋 Challans", "🖼️ Evidence", "📤 Export"
])


# ═══════════════════════════════════════════════════════
# TAB 1 — Live Detection
# ═══════════════════════════════════════════════════════
with tab_live:
    col_video, col_violations = st.columns([3, 2])

    with col_video:
        st.subheader("📹 Video Input")
        source_type = st.radio("Source", ["Upload Video", "Webcam"], horizontal=True)

        video_source = None
        if source_type == "Upload Video":
            uploaded = st.file_uploader(
                "Upload traffic video", type=["mp4","avi","mov","mkv"],
                help="Best: 720p or 1080p, MP4 format"
            )
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read())
                tfile.close()
                video_source = tfile.name
                st.success(f"Loaded: {uploaded.name}")
        else:
            video_source = 0
            st.info("Webcam will be used")

        frame_display = st.empty()
        progress_bar  = st.empty()
        fps_display   = st.empty()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_btn = st.button(
                "▶ Start Detection",
                type="primary",
                disabled=(video_source is None),
                use_container_width=True,
            )
        with btn_col2:
            stop_btn = st.button("⏹ Stop", use_container_width=True)

    with col_violations:
        st.subheader("🚨 Live Violations")
        violation_feed = st.empty()

    # Cached pipeline loader
    @st.cache_resource(show_spinner="Loading AI models (first time ~30s)...")
    def _load_pipeline(anpr_enabled: bool):
        from pipeline import Pipeline
        return Pipeline(enable_anpr=anpr_enabled)

    # ── Detection loop ────────────────────────────────────────────────────────
    if start_btn and video_source is not None:
        st.session_state.processing = True
        st.session_state.violations = []
        st.session_state.challans   = []

        # Apply UI settings to config
        config.YOLO_CONFIDENCE = conf_thresh

        pipeline = _load_pipeline(enable_anpr)

        # Get video properties for progress bar
        cap_probe = cv2.VideoCapture(str(video_source) if video_source != 0 else 0)
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_probe.release()

        try:
            for pf in pipeline.process_video(video_source, skip_frames=skip_frames):
                if stop_btn:
                    break

                # Display annotated frame
                frame_rgb = cv2.cvtColor(pf.frame, cv2.COLOR_BGR2RGB)
                frame_display.image(frame_rgb, use_container_width=True, caption=f"Frame #{pf.frame_id}")

                # Progress
                if total_frames > 0:
                    progress = min(pf.frame_id / total_frames, 1.0)
                    progress_bar.progress(progress, text=f"Frame {pf.frame_id}/{total_frames}")
                fps_display.caption(f"⚡ {pf.fps:.1f} FPS | Tracks: {len(pf.tracked.tracks)}")

                # Accumulate violations
                for v in pf.new_violations:
                    st.session_state.violations.append(v)
                for c in pf.new_challans:
                    st.session_state.challans.append(c)

                # Update sidebar stats
                stat1.metric("Violations", len(st.session_state.violations))
                stat2.metric("Challans",   len(st.session_state.challans))
                stat3.metric("Frames",     pf.frame_id)
                stat4.metric("Tracks",     len(pf.tracked.tracks))

                # Refresh violation feed
                if st.session_state.violations:
                    recent = st.session_state.violations[-5:]
                    md = ""
                    for v in reversed(recent):
                        icon = {"no_helmet":"🪖","triple_riding":"👨‍👩‍👦","wrong_direction":"⬅️"}.get(v.violation_type,"⚠️")
                        status_class = v.status
                        md += f"""
<div class='violation-card'>
{icon} <b>{v.violation_type.replace('_',' ').upper()}</b><br>
🆔 Track #{v.track_id} &nbsp;|&nbsp; 📋 Plate: <code>{v.plate_number}</code><br>
📊 Conf: {v.confidence*100:.0f}% &nbsp;|&nbsp; <span class='{status_class}'>{v.status.upper()}</span><br>
🕐 {v.timestamp[11:19]}
</div>"""
                    violation_feed.markdown(md, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Pipeline error: {e}")

        st.session_state.processing = False
        progress_bar.empty()
        st.success(f"✅ Done! {len(st.session_state.violations)} violations detected, "
                   f"{len(st.session_state.challans)} challans generated.")


# ═══════════════════════════════════════════════════════
# TAB 2 — Challans
# ═══════════════════════════════════════════════════════
with tab_challans:
    st.subheader("📋 Generated Challans")
    challans = st.session_state.challans

    if not challans:
        st.info("No challans yet. Run detection first.")
    else:
        # Summary row
        total_fine = sum(c.fine_amount for c in challans)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Challans", len(challans))
        c2.metric("Total Fine", f"₹{total_fine:,}")
        c3.metric("No Helmet",
                  sum(1 for c in challans if c.violation_type=="no_helmet"))
        c4.metric("Triple Riding",
                  sum(1 for c in challans if c.violation_type=="triple_riding"))

        st.divider()

        # Filter
        filter_type = st.selectbox("Filter by type", ["All"] +
            list({c.violation_type for c in challans}))
        filtered = challans if filter_type == "All" else \
                   [c for c in challans if c.violation_type == filter_type]

        # Challan table
        import pandas as pd
        rows = [c.to_display() for c in filtered]
        df = pd.DataFrame(rows)
        st.table(df)

        # Individual challan cards
        st.subheader("Challan Details")
        for c in reversed(filtered[-10:]):
            with st.expander(f"🎫 {c.challan_id} — {c.violation_desc} — {c.vehicle_number}"):
                cols = st.columns(2)
                with cols[0]:
                    st.write(f"**Vehicle Number:** `{c.vehicle_number}`")
                    st.write(f"**Violation:** {c.violation_desc}")
                    st.write(f"**Fine Amount:** ₹{c.fine_amount:,}")
                    st.write(f"**Location:** {c.location}")
                with cols[1]:
                    st.write(f"**Timestamp:** {c.timestamp}")
                    st.write(f"**Confidence:** {c.confidence*100:.0f}%")
                    st.write(f"**Status:** `{c.status.upper()}`")
                    st.write(f"**Officer:** {c.officer_id}")

                # Show evidence if available
                if c.evidence_path and os.path.exists(c.evidence_path):
                    img = cv2.imread(c.evidence_path)
                    if img is not None:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                 caption="Evidence Frame", use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 3 — Evidence Gallery
# ═══════════════════════════════════════════════════════
with tab_evidence:
    st.subheader("🖼️ Evidence Gallery")
    challans = st.session_state.challans

    with_evidence = [c for c in challans if c.evidence_path and os.path.exists(c.evidence_path)]
    if not with_evidence:
        st.info("No evidence images saved yet.")
    else:
        cols = st.columns(3)
        for i, c in enumerate(with_evidence):
            img = cv2.imread(c.evidence_path)
            if img is not None:
                with cols[i % 3]:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                             caption=f"{c.challan_id} | {c.violation_desc}",
                             use_container_width=True)
                    st.caption(f"Plate: `{c.vehicle_number}` | ₹{c.fine_amount:,}")


# ═══════════════════════════════════════════════════════
# TAB 4 — Export
# ═══════════════════════════════════════════════════════
with tab_export:
    st.subheader("📤 Export Session Data")
    challans = st.session_state.challans

    if not challans:
        st.info("No data to export.")
    else:
        # JSON export
        data = {
            "export_time": datetime.now().isoformat(),
            "total_challans": len(challans),
            "total_fine_inr": sum(c.fine_amount for c in challans),
            "challans": [c.to_dict() for c in challans],
        }
        json_str = json.dumps(data, indent=2)

        st.download_button(
            label     = "⬇️ Download Challans (JSON)",
            data      = json_str,
            file_name = f"challans_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime      = "application/json",
            type      = "primary",
        )

        # CSV export
        try:
            import pandas as pd
            df = pd.DataFrame([c.to_dict() for c in challans])
            csv = df.to_csv(index=False)
            st.download_button(
                label     = "⬇️ Download Challans (CSV)",
                data      = csv,
                file_name = f"challans_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime      = "text/csv",
            )
        except ImportError:
            pass

        st.divider()
        st.subheader("Preview JSON")
        st.json(data)

"""
Driver Pulse — Real-time Safety & Earnings Dashboard
=====================================================
Streamlit web app for the she++ hackathon submission.

Sections built:
  1. Trip Overview          (sensor data — Jatin)
  2. Flagged Moments        (sensor data — Jatin)
  3. Decision Log           (sensor data — Jatin)
  4. Earnings Velocity      (placeholder — teammate)
"""

import streamlit as st
import pandas as pd
import os

# ─── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Driver Pulse",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Data loading ─────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'backend', 'processed_outputs')
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'backend', 'driver_pulse_hackathon_data')


@st.cache_data
def load_data():
    data = {}
    # Processed outputs
    for name, fname in [
        ('flagged_moments', 'flagged_moments.csv'),
        ('trip_summaries', 'trip_summaries.csv'),
        ('decision_log', 'decision_log.csv'),
        ('earnings_velocity', 'earnings_velocity.csv'),
    ]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
        else:
            data[name] = pd.DataFrame()

    # Raw reference
    drv_path = os.path.join(RAW_DIR, 'drivers', 'drivers.csv')
    if os.path.exists(drv_path):
        data['drivers'] = pd.read_csv(drv_path)
    else:
        data['drivers'] = pd.DataFrame()

    trips_path = os.path.join(RAW_DIR, 'trips', 'trips.csv')
    if os.path.exists(trips_path):
        data['trips'] = pd.read_csv(trips_path)
    else:
        data['trips'] = pd.DataFrame()

    return data


data = load_data()
flags = data['flagged_moments']
trips = data['trip_summaries']
dlog = data['decision_log']
drivers = data['drivers']

# ─── Sidebar ──────────────────────────────────────────────
st.sidebar.title("🚗 Driver Pulse")
st.sidebar.markdown("**Real-time Safety & Earnings Dashboard**")

# Driver selector
driver_ids = sorted(trips['driver_id'].unique()) if 'driver_id' in trips.columns else []
selected_driver = st.sidebar.selectbox(
    "Select Driver",
    options=["All Drivers"] + driver_ids,
    index=0,
)

# Page navigation
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "⚠️ Flagged Moments", "📋 Decision Log", "💰 Earnings Velocity"],
)

# Filter data by driver
if selected_driver != "All Drivers":
    f_trips = trips[trips['driver_id'] == selected_driver] if 'driver_id' in trips.columns else trips
    f_flags = flags[flags['driver_id'] == selected_driver] if 'driver_id' in flags.columns else flags
else:
    f_trips = trips
    f_flags = flags

# ─── Helper ───────────────────────────────────────────────
SEVERITY_COLORS = {'low': '🟢', 'medium': '🟡', 'high': '🔴', 'none': '⚪'}
QUALITY_COLORS = {'excellent': '🟢', 'good': '🟡', 'fair': '🟠', 'poor': '🔴'}


# ═════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🏠 Trip Overview Dashboard")

    if selected_driver != "All Drivers" and not drivers.empty:
        drv = drivers[drivers['driver_id'] == selected_driver]
        if not drv.empty:
            d = drv.iloc[0]
            st.markdown(
                f"**Driver:** {d.get('name', selected_driver)} · "
                f"**City:** {d.get('city', '—')} · "
                f"**Rating:** ⭐ {d.get('rating', '—')} · "
                f"**Experience:** {d.get('experience_months', '—')} months"
            )
            st.divider()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trips", len(f_trips))
    col2.metric("Flagged Moments", len(f_flags))
    avg_stress = f_trips['stress_score'].mean() if 'stress_score' in f_trips.columns and len(f_trips) > 0 else 0
    col3.metric("Avg Stress Score", f"{avg_stress:.2f}")
    if 'trip_quality_rating' in f_trips.columns and len(f_trips) > 0:
        excellent_pct = (f_trips['trip_quality_rating'] == 'excellent').sum() / len(f_trips) * 100
        col4.metric("Excellent Trips", f"{excellent_pct:.0f}%")
    else:
        col4.metric("Excellent Trips", "—")

    st.divider()

    # Trip Quality Distribution
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Trip Quality Distribution")
        if 'trip_quality_rating' in f_trips.columns and len(f_trips) > 0:
            quality_counts = f_trips['trip_quality_rating'].value_counts().reindex(
                ['excellent', 'good', 'fair', 'poor'], fill_value=0
            )
            for rating, count in quality_counts.items():
                icon = QUALITY_COLORS.get(rating, '⚪')
                pct = count / len(f_trips) * 100
                st.markdown(f"{icon} **{rating.title()}**: {count} trips ({pct:.0f}%)")
                st.progress(pct / 100)
        else:
            st.info("No trip data available.")

    with col_right:
        st.subheader("Flag Severity Breakdown")
        if 'severity' in f_flags.columns and len(f_flags) > 0:
            sev_counts = f_flags['severity'].value_counts().reindex(
                ['low', 'medium', 'high'], fill_value=0
            )
            for sev, count in sev_counts.items():
                icon = SEVERITY_COLORS.get(sev, '⚪')
                pct = count / len(f_flags) * 100
                st.markdown(f"{icon} **{sev.title()}**: {count} flags ({pct:.0f}%)")
                st.progress(pct / 100)
        else:
            st.info("No flagged moments.")

    st.divider()

    # Flag type breakdown
    st.subheader("Flagged Event Types")
    if 'flag_type' in f_flags.columns and len(f_flags) > 0:
        type_counts = f_flags['flag_type'].value_counts()
        cols = st.columns(min(len(type_counts), 5))
        for i, (ftype, count) in enumerate(type_counts.items()):
            with cols[i % len(cols)]:
                st.metric(ftype.replace('_', ' ').title(), count)
    else:
        st.info("No flagged events to display.")

    st.divider()

    # Trip table
    st.subheader("All Trips")
    if len(f_trips) > 0:
        display_cols = ['trip_id', 'driver_id', 'date', 'duration_min', 'distance_km',
                        'fare', 'flagged_moments_count', 'max_severity', 'stress_score',
                        'trip_quality_rating']
        available = [c for c in display_cols if c in f_trips.columns]
        st.dataframe(
            f_trips[available].sort_values('stress_score', ascending=False),
            width='stretch',
            hide_index=True,
        )
    else:
        st.info("No trip data available.")


# ═════════════════════════════════════════════════════════
# PAGE: FLAGGED MOMENTS
# ═════════════════════════════════════════════════════════
elif page == "⚠️ Flagged Moments":
    st.title("⚠️ Flagged Moments")
    st.caption("Each row is a stress/safety event detected from accelerometer + audio sensors.")

    if len(f_flags) == 0:
        st.info("No flagged moments for the selected driver.")
    else:
        # Filters row
        col_ft, col_sev, col_trip = st.columns(3)
        with col_ft:
            flag_types = ['All'] + sorted(f_flags['flag_type'].unique().tolist())
            sel_type = st.selectbox("Flag Type", flag_types)
        with col_sev:
            severities = ['All'] + sorted(f_flags['severity'].unique().tolist())
            sel_sev = st.selectbox("Severity", severities)
        with col_trip:
            trip_ids = ['All'] + sorted(f_flags['trip_id'].unique().tolist())
            sel_trip = st.selectbox("Trip", trip_ids)

        view = f_flags.copy()
        if sel_type != 'All':
            view = view[view['flag_type'] == sel_type]
        if sel_sev != 'All':
            view = view[view['severity'] == sel_sev]
        if sel_trip != 'All':
            view = view[view['trip_id'] == sel_trip]

        st.markdown(f"**Showing {len(view)} of {len(f_flags)} flags**")

        # Card-style display for each flag
        for _, row in view.iterrows():
            sev_icon = SEVERITY_COLORS.get(row.get('severity', ''), '⚪')
            with st.expander(
                f"{sev_icon} {row.get('flag_type', '').replace('_', ' ').title()} — "
                f"{row.get('trip_id', '')} @ {row.get('timestamp', '')} "
                f"[{row.get('severity', '').upper()}]",
                expanded=False,
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Motion Score", f"{row.get('motion_score', 0):.2f}")
                c2.metric("Audio Score", f"{row.get('audio_score', 0):.2f}")
                c3.metric("Combined Score", f"{row.get('combined_score', 0):.2f}")

                st.markdown(f"**Explanation:** {row.get('explanation', '—')}")
                st.markdown(f"**Context:** {row.get('context', '—')}")
                st.markdown(
                    f"**Driver:** {row.get('driver_id', '—')} · "
                    f"**Elapsed:** {row.get('elapsed_seconds', 0)}s into trip"
                )


# ═════════════════════════════════════════════════════════
# PAGE: DECISION LOG
# ═════════════════════════════════════════════════════════
elif page == "📋 Decision Log":
    st.title("📋 Structured Decision Log")
    st.caption(
        "Every sensor reading that exceeded a threshold, "
        "traced to the raw value and detection rule."
    )

    if dlog.empty:
        st.info("No decision log data. Run `decision_log_generator.py` first.")
    else:
        # Filters
        col_sig, col_evt = st.columns(2)
        with col_sig:
            sig_types = ['All'] + sorted(dlog['signal_type'].unique().tolist())
            sel_sig = st.selectbox("Signal Type", sig_types)
        with col_evt:
            evt_labels = ['All'] + sorted(dlog['event_label'].unique().tolist())
            sel_evt = st.selectbox("Event Label", evt_labels)

        view = dlog.copy()
        if sel_sig != 'All':
            view = view[view['signal_type'] == sel_sig]
        if sel_evt != 'All':
            view = view[view['event_label'] == sel_evt]

        st.markdown(f"**Showing {len(view)} of {len(dlog)} entries**")

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Accelerometer Events",
                   len(view[view['signal_type'] == 'ACCELEROMETER']))
        c2.metric("Audio Events",
                   len(view[view['signal_type'] == 'AUDIO']))
        c3.metric("Total Detections", len(view))

        st.divider()

        # Event breakdown
        st.subheader("Event Breakdown")
        evt_counts = view['event_label'].value_counts()
        cols = st.columns(min(len(evt_counts), 5)) if len(evt_counts) > 0 else []
        for i, (label, count) in enumerate(evt_counts.items()):
            with cols[i % len(cols)]:
                st.metric(label.replace('_', ' ').title(), count)

        st.divider()

        # Full log table
        st.subheader("Full Log")
        st.dataframe(view, width='stretch', hide_index=True)


# ═════════════════════════════════════════════════════════
# PAGE: EARNINGS VELOCITY (placeholder for teammate)
# ═════════════════════════════════════════════════════════
elif page == "💰 Earnings Velocity":
    st.title("💰 Earnings Velocity")

    st.info(
        "🚧 **This section is under development by the earnings team.**\n\n"
        "It will display:\n"
        "- Real-time earnings speed vs. target\n"
        "- Goal completion forecast (on track / at risk / behind)\n"
        "- Daily progress chart\n"
        "- Projected shift earnings\n\n"
        "The backend data (`earnings_velocity.csv`, `driver_goals.csv`) "
        "is already generated and ready to wire up."
    )

    # Show raw data preview if available
    ev = data.get('earnings_velocity', pd.DataFrame())
    if not ev.empty:
        st.subheader("Raw Earnings Data Preview")
        st.dataframe(ev.head(20), width='stretch', hide_index=True)


# ─── Footer ──────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("Driver Pulse · she++ Hackathon 2026")

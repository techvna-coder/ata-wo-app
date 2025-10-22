# ui/progress_components.py
"""
Streamlit UI components for progress tracking.

Usage in Streamlit app:
    from ui.progress_components import show_progress_monitor, realtime_progress_widget
"""
import streamlit as st
from typing import Optional, Dict, Any
import time
from pathlib import Path


# ============================================================================
# PROGRESS MONITOR (Real-time)
# ============================================================================
def show_progress_monitor(
    tracker,
    container=None,
    show_metrics: bool = True,
    show_logs: bool = True
):
    """
    Show real-time progress monitor in Streamlit.
    
    Args:
        tracker: ProgressTracker instance
        container: Streamlit container (or None for main)
        show_metrics: Show success/fail/skip metrics
        show_logs: Show processing logs
    
    Usage:
        tracker = StreamlitProgressTracker(100, "Processing")
        show_progress_monitor(tracker)
        
        # Then in your processing loop:
        for item in items:
            process(item)
            tracker.update(current_item=item)
    """
    container = container or st
    
    # Main progress bar
    progress_bar = container.progress(0)
    status_text = container.empty()
    
    # Metrics
    if show_metrics:
        metrics_container = container.container()
        col1, col2, col3, col4 = metrics_container.columns(4)
        metric_success = col1.empty()
        metric_failed = col2.empty()
        metric_skipped = col3.empty()
        metric_speed = col4.empty()
    
    # Log area
    if show_logs:
        with container.expander("📋 Processing Log", expanded=False):
            log_area = st.empty()
            logs = []
    
    # Update function (call this in loop)
    def update_ui(state):
        """Update UI with current state."""
        # Progress bar
        progress_bar.progress(state.percentage / 100)
        
        # Status text
        elapsed = state.elapsed_seconds
        eta = state.eta_seconds
        
        status_msg = f"**Processing:** {state.processed}/{state.total} ({state.percentage:.1f}%)"
        status_msg += f" | ⏱ {elapsed:.0f}s elapsed, {eta:.0f}s remaining"
        
        if state.current_item:
            status_msg += f"\n\n🔄 *Current:* `{state.current_item}`"
        
        status_text.markdown(status_msg)
        
        # Metrics
        if show_metrics:
            metric_success.metric("✅ Success", state.success, delta=None)
            metric_failed.metric("❌ Failed", state.failed, delta=None)
            metric_skipped.metric("⏭️ Skipped", state.skipped, delta=None)
            
            speed = state.processed / elapsed if elapsed > 0 else 0
            metric_speed.metric("⚡ Speed", f"{speed:.1f}/s", delta=None)
        
        # Logs
        if show_logs and state.current_item:
            timestamp = time.strftime("%H:%M:%S")
            status_icon = "✅" if state.status != "error" else "❌"
            logs.append(f"`{timestamp}` {status_icon} {state.current_item}")
            
            # Keep last 50 logs
            recent_logs = logs[-50:]
            log_area.text("\n".join(recent_logs))
    
    return update_ui


# ============================================================================
# COMPACT PROGRESS WIDGET
# ============================================================================
def compact_progress_widget(
    total: int,
    processed: int,
    description: str = "Processing",
    container=None
):
    """
    Compact progress widget (single line).
    
    Usage:
        for i, item in enumerate(items):
            process(item)
            compact_progress_widget(len(items), i+1, "Processing WO")
    """
    container = container or st
    
    pct = (processed / total * 100) if total > 0 else 0
    
    # Progress bar
    container.progress(pct / 100)
    
    # Single line status
    container.caption(f"{description}: {processed}/{total} ({pct:.1f}%)")


# ============================================================================
# FILE-BASED PROGRESS MONITOR (for background jobs)
# ============================================================================
def monitor_background_progress(
    progress_file: str = "data_store/ingest_progress.json",
    refresh_interval: float = 1.0
):
    """
    Monitor progress of background job from file.
    
    Usage:
        # In main app
        if st.button("Start Background Job"):
            start_background_job()
            st.rerun()
        
        # Show monitor
        if Path("progress.json").exists():
            monitor_background_progress("progress.json")
    """
    from core.progress_tracker import read_progress_file
    
    st.subheader("📊 Background Job Monitor")
    
    # Create placeholder
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.container()
    
    # Auto-refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
    
    # Read and display
    while True:
        progress_data = read_progress_file(progress_file)
        
        if not progress_data:
            status_text.warning("⏳ Waiting for job to start...")
            time.sleep(refresh_interval)
            if not auto_refresh:
                break
            continue
        
        # Update UI
        pct = progress_data['percentage']
        progress_bar.progress(pct / 100)
        
        status = progress_data['status']
        processed = progress_data['processed']
        total = progress_data['total']
        
        if status == "completed":
            status_text.success(f"✅ Completed: {processed}/{total}")
            break
        elif status == "error":
            status_text.error(f"❌ Error: {progress_data.get('current_item', 'Unknown error')}")
            break
        else:
            elapsed = progress_data['elapsed_seconds']
            eta = progress_data['eta_seconds']
            status_text.info(f"🔄 Processing: {processed}/{total} ({pct:.1f}%) | {elapsed:.0f}s elapsed, {eta:.0f}s remaining")
        
        # Metrics
        with metrics_container:
            col1, col2, col3 = st.columns(3)
            col1.metric("Success", progress_data['success'])
            col2.metric("Failed", progress_data['failed'])
            col3.metric("Skipped", progress_data['skipped'])
        
        # Refresh or break
        if not auto_refresh or status in ["completed", "error"]:
            break
        
        time.sleep(refresh_interval)
        st.rerun()


# ============================================================================
# BATCH UPLOAD WITH PROGRESS
# ============================================================================
def batch_upload_with_progress():
    """
    Complete batch upload component with progress.
    
    Usage in Streamlit app:
        batch_upload_with_progress()
    """
    st.subheader("📤 Batch WO Upload")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload WO Excel files",
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        key="wo_batch_upload"
    )
    
    if not uploaded_files:
        st.info("📁 Select Excel files to upload")
        return
    
    st.success(f"✓ {len(uploaded_files)} files selected")
    
    # Preview
    with st.expander("📋 Files to process"):
        for f in uploaded_files:
            st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")
    
    # Process button
    if st.button("🚀 Start Processing", type="primary"):
        from core.progress_tracker import StreamlitProgressTracker
        from core.store_optimized import append_wo_training
        
        # Create tracker
        tracker = StreamlitProgressTracker(len(uploaded_files), "Uploading WO files")
        
        # Progress container
        progress_container = st.container()
        
        with progress_container:
            tracker.initialize_ui()
            tracker.start()
            
            results = []
            
            for uploaded_file in uploaded_files:
                try:
                    # Read Excel
                    df = pd.read_excel(uploaded_file)
                    
                    # Append to store
                    append_wo_training(df)
                    
                    # Update progress
                    tracker.update(
                        success=True,
                        current_item=f"{uploaded_file.name} ({len(df)} rows)"
                    )
                    
                    results.append({
                        'File': uploaded_file.name,
                        'Status': '✅ Success',
                        'Rows': len(df)
                    })
                    
                except Exception as e:
                    tracker.update(
                        success=False,
                        current_item=f"{uploaded_file.name}: {str(e)}"
                    )
                    
                    results.append({
                        'File': uploaded_file.name,
                        'Status': '❌ Failed',
                        'Error': str(e)
                    })
            
            tracker.complete()
        
        # Results summary
        st.success("✅ Batch upload completed!")
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        success_count = results_df['Status'].str.contains('Success').sum()
        failed_count = len(results_df) - success_count
        total_rows = results_df.get('Rows', pd.Series([0])).sum()
        
        col1.metric("✅ Successful", success_count)
        col2.metric("❌ Failed", failed_count)
        col3.metric("📝 Total Rows", total_rows)


# ============================================================================
# PREDICTION BATCH WITH PROGRESS
# ============================================================================
def batch_predict_with_progress(records: list[dict]):
    """
    Batch prediction with progress UI.
    
    Args:
        records: List of {'defect': ..., 'action': ...}
    
    Usage:
        records = [{'defect': 'FAULT', 'action': 'FIXED'}, ...]
        batch_predict_with_progress(records)
    """
    from core.progress_tracker import StreamlitProgressTracker
    from core.ata_catalog_optimized import ATACatalog
    
    st.subheader("🔮 Batch ATA Prediction")
    
    # Load catalog
    with st.spinner("Loading ATA catalog..."):
        catalog = ATACatalog()
    
    # Create tracker
    tracker = StreamlitProgressTracker(len(records), "Predicting ATA codes")
    
    # Progress container
    with st.container():
        tracker.initialize_ui()
        tracker.start()
        
        predictions = []
        
        for record in records:
            try:
                # Predict
                best, _ = catalog.predict(
                    record['defect'],
                    record['action']
                )
                
                predictions.append({
                    'Defect': record['defect'][:50],
                    'Action': record['action'][:50],
                    'Predicted ATA': best['ata04'] if best else 'N/A',
                    'Confidence': f"{best['score']:.2f}" if best else '0.00',
                    'Status': '✅'
                })
                
                tracker.update(
                    success=True,
                    current_item=f"{record['defect'][:30]}..."
                )
                
            except Exception as e:
                predictions.append({
                    'Defect': record['defect'][:50],
                    'Action': record['action'][:50],
                    'Predicted ATA': 'ERROR',
                    'Confidence': '0.00',
                    'Status': '❌'
                })
                
                tracker.update(success=False)
        
        tracker.complete()
    
    # Show results
    st.success(f"✅ Predicted {len(predictions)} records")
    
    results_df = pd.DataFrame(predictions)
    st.dataframe(results_df, use_container_width=True)
    
    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions",
        data=csv,
        file_name="ata_predictions.csv",
        mime="text/csv"
    )


# ============================================================================
# SIMPLE DEMO
# ============================================================================
if __name__ == "__main__":
    import pandas as pd
    
    st.title("Progress Tracking Demo")
    
    # Demo 1: Simple progress
    st.header("Demo 1: Simple Progress")
    
    if st.button("Run Demo 1"):
        for i in range(100):
            time.sleep(0.02)
            compact_progress_widget(100, i+1, "Processing items")
        
        st.success("✅ Demo 1 complete!")
    
    # Demo 2: Full progress monitor
    st.header("Demo 2: Full Progress Monitor")
    
    if st.button("Run Demo 2"):
        from core.progress_tracker import StreamlitProgressTracker
        
        tracker = StreamlitProgressTracker(50, "Demo processing")
        
        with st.container():
            tracker.initialize_ui()
            tracker.start()
            
            for i in range(50):
                time.sleep(0.05)
                
                success = i % 5 != 0
                tracker.update(
                    success=success,
                    current_item=f"Item {i+1}"
                )
            
            tracker.complete()
        
        st.success("✅ Demo 2 complete!")

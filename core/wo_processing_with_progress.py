# examples/wo_processing_with_progress.py
"""
Examples of using progress tracker in WO processing workflows.
"""
from pathlib import Path
import pandas as pd
import time


# ============================================================================
# EXAMPLE 1: Console Progress for Batch Ingestion
# ============================================================================
def ingest_wo_batch_with_progress(excel_files: list[str]):
    """
    Ingest multiple WO Excel files with console progress bar.
    
    Usage:
        files = ["wo_batch_1.xlsx", "wo_batch_2.xlsx", ...]
        ingest_wo_batch_with_progress(files)
    """
    from core.progress_tracker import ConsoleProgressTracker
    from core.store_optimized import append_wo_training
    
    tracker = ConsoleProgressTracker(len(excel_files), "Ingesting WO files")
    tracker.start()
    
    results = {
        'success': 0,
        'failed': 0,
        'total_rows': 0
    }
    
    for filepath in excel_files:
        try:
            # Read Excel
            df = pd.read_excel(filepath)
            
            # Append to store
            append_wo_training(df)
            
            # Update progress
            tracker.update(
                success=True,
                current_item=Path(filepath).name
            )
            
            results['success'] += 1
            results['total_rows'] += len(df)
            
        except Exception as e:
            tracker.update(
                success=False,
                current_item=f"{Path(filepath).name}: {str(e)}"
            )
            results['failed'] += 1
    
    tracker.complete()
    
    print(f"\n✓ Ingestion complete:")
    print(f"  Success: {results['success']} files")
    print(f"  Failed: {results['failed']} files")
    print(f"  Total rows: {results['total_rows']}")
    
    return results


# ============================================================================
# EXAMPLE 2: Streamlit Progress for UI
# ============================================================================
def ingest_wo_streamlit(uploaded_files):
    """
    Ingest WO files in Streamlit with progress UI.
    
    Usage in Streamlit app:
        uploaded = st.file_uploader("Upload WO files", accept_multiple_files=True)
        if st.button("Process") and uploaded:
            ingest_wo_streamlit(uploaded)
    """
    import streamlit as st
    from core.progress_tracker import StreamlitProgressTracker
    from core.store_optimized import append_wo_training
    
    # Create progress tracker
    tracker = StreamlitProgressTracker(len(uploaded_files), "Processing WO files")
    
    # Initialize UI in Streamlit container
    with st.container():
        tracker.initialize_ui()
        tracker.start()
        
        results = []
        
        for uploaded_file in uploaded_files:
            try:
                # Read file
                df = pd.read_excel(uploaded_file)
                
                # Process
                append_wo_training(df)
                
                # Update progress
                tracker.update(
                    success=True,
                    current_item=uploaded_file.name
                )
                
                results.append({
                    'file': uploaded_file.name,
                    'status': 'success',
                    'rows': len(df)
                })
                
            except Exception as e:
                tracker.update(
                    success=False,
                    current_item=f"{uploaded_file.name}: {str(e)}"
                )
                
                results.append({
                    'file': uploaded_file.name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        tracker.complete()
    
    # Show results
    st.success(f"✓ Processed {len(uploaded_files)} files")
    
    # Results table
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)
    
    return results


# ============================================================================
# EXAMPLE 3: Background Job with File Progress
# ============================================================================
def ingest_wo_background(excel_files: list[str], progress_file: str = "data_store/ingest_progress.json"):
    """
    Ingest WO files as background job with file-based progress.
    
    Monitor progress from another process:
        from core.progress_tracker import read_progress_file
        progress = read_progress_file("data_store/ingest_progress.json")
        print(f"Progress: {progress['percentage']:.1f}%")
    
    Usage:
        # Start background job
        import subprocess
        subprocess.Popen([
            "python", "-c",
            "from examples.wo_processing_with_progress import ingest_wo_background; "
            "ingest_wo_background(['file1.xlsx', 'file2.xlsx'])"
        ])
    """
    from core.progress_tracker import FileProgressTracker
    from core.store_optimized import append_wo_training
    
    tracker = FileProgressTracker(len(excel_files), "Background WO ingestion", progress_file)
    tracker.start()
    
    for filepath in excel_files:
        try:
            df = pd.read_excel(filepath)
            append_wo_training(df)
            
            tracker.update(
                success=True,
                current_item=Path(filepath).name
            )
            
        except Exception as e:
            tracker.update(
                success=False,
                current_item=f"{Path(filepath).name}: {str(e)}"
            )
    
    tracker.complete()


# ============================================================================
# EXAMPLE 4: Batch Prediction with Progress
# ============================================================================
def predict_batch_with_progress(wo_records: list[dict], mode: str = "console"):
    """
    Make batch predictions with progress tracking.
    
    Args:
        wo_records: List of dicts with keys: defect, action
        mode: 'console', 'streamlit', or 'file'
    
    Returns:
        List of predictions
    
    Usage:
        records = [
            {'defect': 'ECAM FAULT', 'action': 'REPLACED LRU'},
            {'defect': 'HYDRAULIC LEAK', 'action': 'RESEALED FITTING'},
            ...
        ]
        predictions = predict_batch_with_progress(records)
    """
    from core.progress_tracker import create_progress_tracker
    from core.ata_catalog_optimized import ATACatalog
    
    # Load catalog
    catalog = ATACatalog()
    
    # Create tracker
    tracker = create_progress_tracker(
        len(wo_records),
        "Predicting ATA codes",
        mode=mode
    )
    tracker.start()
    
    predictions = []
    
    for record in wo_records:
        try:
            # Predict
            best, candidates = catalog.predict(
                record['defect'],
                record['action']
            )
            
            predictions.append({
                'defect': record['defect'],
                'action': record['action'],
                'predicted_ata': best['ata04'] if best else None,
                'confidence': best['score'] if best else 0,
                'status': 'success'
            })
            
            tracker.update(
                success=True,
                current_item=f"{record['defect'][:30]}..."
            )
            
        except Exception as e:
            predictions.append({
                'defect': record['defect'],
                'action': record['action'],
                'predicted_ata': None,
                'confidence': 0,
                'status': 'failed',
                'error': str(e)
            })
            
            tracker.update(success=False)
    
    tracker.complete()
    
    return predictions


# ============================================================================
# EXAMPLE 5: Context Manager Style
# ============================================================================
def process_wo_files_simple(files: list[str]):
    """
    Simplest usage with context manager.
    
    Usage:
        files = ["wo1.xlsx", "wo2.xlsx", "wo3.xlsx"]
        process_wo_files_simple(files)
    """
    from core.progress_tracker import track_progress
    from core.store_optimized import append_wo_training
    
    with track_progress(files, "Processing WO files", mode="console") as progress:
        for filepath in progress:
            df = pd.read_excel(filepath)
            append_wo_training(df)
    
    print("\n✓ All files processed!")


# ============================================================================
# EXAMPLE 6: Custom Callback
# ============================================================================
def process_with_custom_logging(excel_files: list[str], log_file: str = "processing.log"):
    """
    Process with custom logging callback.
    
    Usage:
        process_with_custom_logging(["wo1.xlsx", "wo2.xlsx"])
    """
    from core.progress_tracker import ConsoleProgressTracker
    from core.store_optimized import append_wo_training
    
    # Setup logging
    log = open(log_file, 'w')
    
    def log_callback(state):
        """Custom callback to log progress."""
        log.write(f"[{state.percentage:.1f}%] {state.current_item}\n")
        log.flush()
    
    # Create tracker with callback
    tracker = ConsoleProgressTracker(len(excel_files), "Processing with logging")
    tracker.add_callback(log_callback)
    tracker.start()
    
    for filepath in excel_files:
        try:
            df = pd.read_excel(filepath)
            append_wo_training(df)
            tracker.update(success=True, current_item=Path(filepath).name)
        except Exception as e:
            tracker.update(success=False, current_item=f"{Path(filepath).name}: {e}")
    
    tracker.complete()
    log.close()
    
    print(f"✓ Processing complete. Log saved to {log_file}")


# ============================================================================
# DEMO RUNNER
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("WO PROCESSING WITH PROGRESS - DEMO")
    print("="*70)
    
    # Demo 1: Simulate batch ingestion
    print("\nDemo 1: Console Progress Bar")
    print("-"*70)
    
    # Simulate files
    demo_files = [f"demo_wo_batch_{i}.xlsx" for i in range(10)]
    
    from core.progress_tracker import ConsoleProgressTracker
    
    tracker = ConsoleProgressTracker(len(demo_files), "Demo ingestion")
    tracker.start()
    
    for i, filepath in enumerate(demo_files):
        time.sleep(0.2)  # Simulate processing
        
        # Simulate occasional failure
        success = i % 7 != 0
        
        tracker.update(
            success=success,
            current_item=filepath
        )
    
    tracker.complete()
    
    # Demo 2: Context manager
    print("\n\nDemo 2: Context Manager Style")
    print("-"*70)
    
    from core.progress_tracker import track_progress
    
    items = [f"WO-{i:04d}" for i in range(20)]
    
    with track_progress(items, "Simple processing", mode="console") as progress:
        for item in progress:
            time.sleep(0.1)
    
    print("\n✓ All demos completed!")

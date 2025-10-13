# core/mapping.py - COMPLETE VERSION

# ============================================================
# COLUMN MAPPING: External names → Internal names
# ============================================================

MAP_IN2INTERNAL = {
    "ATA": "ATA04_Entered",
    "W/O Description": "Defect_Text",
    "W/O Action": "Rectification_Text",
    "Issued": "Open_Date",
    "Closed": "Close_Date",
    "A/C": "AC_Registration",
    "Type": "WO_Type",
    "ATA 04 Corrected": "ATA04_Final",
    "WO_Number": "WO_Number",
}

# Alias mapping cho các tên cột biến thể
MAP_ALIASES = {
    # ATA codes
    "ATA": "ATA04_Entered",
    "ATA 04": "ATA04_Entered",
    "ATA04": "ATA04_Entered",
    "ATA Code": "ATA04_Entered",
    "ATA_Code": "ATA04_Entered",
    
    # Description
    "W/O Description": "Defect_Text",
    "WO Description": "Defect_Text",
    "Description": "Defect_Text",
    "Defect": "Defect_Text",
    "Symptom": "Defect_Text",
    
    # Action
    "W/O Action": "Rectification_Text",
    "WO Action": "Rectification_Text",
    "Action": "Rectification_Text",
    "Rectification": "Rectification_Text",
    "Corrective Action": "Rectification_Text",
    
    # Dates
    "Issued": "Open_Date",
    "Open Date": "Open_Date",
    "Opened": "Open_Date",
    "Closed": "Close_Date",
    "Close Date": "Close_Date",
    
    # Aircraft
    "A/C": "AC_Registration",
    "AC": "AC_Registration",
    "Aircraft": "AC_Registration",
    "Registration": "AC_Registration",
    "Reg": "AC_Registration",
    
    # Type
    "Type": "WO_Type",
    "WO Type": "WO_Type",
    "Work Order Type": "WO_Type",
    
    # Final ATA
    "ATA 04 Corrected": "ATA04_Final",
    "ATA04 Corrected": "ATA04_Final",
    "ATA Corrected": "ATA04_Final",
    "ATA Final": "ATA04_Final",
    "ATA04_Final": "ATA04_Final",
    
    # WO Number
    "WO_Number": "WO_Number",
    "WO Number": "WO_Number",
    "Work Order": "WO_Number",
    "Number": "WO_Number",
}

# Internal required columns (mandatory in processing)
INCOLS = [
    "ATA04_Entered",
    "Defect_Text",
    "Rectification_Text",
    "Open_Date",
    "Close_Date",
    "AC_Registration",
    "WO_Type",
]

# Output column order (for result Excel)
ORDER_OUTCOLS = [
    "Is_Technical_Defect",
    "ATA04_Entered",
    "ATA04_From_Cited",
    "Cited_Manual",
    "Cited_Task",
    "Cited_Exists",
    "ATA04_Derived",
    "Derived_Task",
    "Derived_DocType",
    "Derived_Score",
    "Evidence_Snippet",
    "Evidence_Source",
    "Decision",
    "ATA04_Final",
    "Confidence",
    "Reason",
]

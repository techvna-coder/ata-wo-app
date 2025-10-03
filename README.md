# WO → ATA04 Checker (Drive + Incremental Memory)


Ứng dụng Streamlit để:
- Đồng bộ thư mục Google Drive (WO đã làm sạch + bảng ATA) vào **bộ nhớ cục bộ**.
- Xây **Catalog TF‑IDF** từ bộ nhớ (incremental, không cần SGML).
- Suy luận **ATA04** và xuất **ATA corrected** với quyết định CONFIRM/CORRECT/REVIEW + Confidence + Evidence snippet.


## 1) Chuẩn bị
- Python 3.10+
- Tạo Service Account (GCP) và tải file JSON. Chia sẻ thư mục Drive cho email của service account.


```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

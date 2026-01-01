# 🐍 Hướng Dẫn Cài Đặt Python cho Windows

## Phương Án 1: Cài Đặt Python Từ Microsoft Store (Khuyến Nghị - Nhanh Nhất)

### Bước 1: Mở Microsoft Store
1. Nhấn phím **Windows** trên bàn phím
2. Gõ "Microsoft Store" và nhấn Enter
3. Hoặc click vào biểu tượng Microsoft Store trên taskbar

### Bước 2: Tìm và Cài Python
1. Trong Microsoft Store, tìm kiếm: **"Python 3.12"** hoặc **"Python 3.11"**
2. Chọn **Python 3.12** (phiên bản mới nhất ổn định)
3. Click nút **"Get"** hoặc **"Install"**
4. Đợi quá trình cài đặt hoàn tất (khoảng 2-3 phút)

### Bước 3: Kiểm Tra Cài Đặt
Mở PowerShell hoặc Command Prompt và gõ:
```powershell
python --version
```

Nếu hiển thị `Python 3.12.x` → **Thành công!** ✅

---

## Phương Án 2: Cài Đặt Từ Python.org (Tùy Chỉnh Nhiều Hơn)

### Bước 1: Download Python
1. Mở trình duyệt và truy cập: **https://www.python.org/downloads/**
2. Click nút **"Download Python 3.12.x"** (phiên bản mới nhất)
3. File installer sẽ được tải về (khoảng 25-30 MB)

### Bước 2: Chạy Installer
1. Mở file `python-3.12.x-amd64.exe` vừa tải
2. **QUAN TRỌNG:** ✅ Tích vào ô **"Add Python to PATH"** (ở dưới cùng)
3. Click **"Install Now"** (cài đặt mặc định)
4. Hoặc click **"Customize installation"** nếu muốn tùy chỉnh

### Bước 3: Hoàn Tất Cài Đặt
1. Đợi quá trình cài đặt (2-3 phút)
2. Click **"Close"** khi hoàn tất
3. **Khởi động lại PowerShell/Command Prompt** để PATH có hiệu lực

### Bước 4: Kiểm Tra
```powershell
python --version
pip --version
```

Nếu cả 2 lệnh đều hiển thị phiên bản → **Thành công!** ✅

---

## Sau Khi Cài Python - Cài Đặt Dependencies

### Bước 1: Mở PowerShell
```powershell
cd C:\Users\Administrator\Downloads\NLP\Tai
```

### Bước 2: Cài Đặt Packages
```powershell
pip install -r requirements.txt
```

**Lưu ý:** Quá trình này sẽ tải và cài đặt:
- Flask (web framework)
- numpy (tính toán số học)
- scikit-learn (machine learning)
- matplotlib (visualization)
- networkx (graph algorithms)
- requests (HTTP requests)
- underthesea (Vietnamese NLP)

**Thời gian:** Khoảng 3-5 phút tùy tốc độ mạng

### Bước 3: Chạy Ứng Dụng
```powershell
python app.py
```

Bạn sẽ thấy:
```
🚀 Text Summarization System
============================================================
📚 3 Pipelines Available:
  A: TF-IDF + PageRank + Logistic Regression
  B: TextRank (Overlap-based)
  C: Multi-Document Ranking
============================================================
🌐 Open browser: http://127.0.0.1:5000/
============================================================
 * Running on http://127.0.0.1:5000
```

### Bước 4: Mở Trình Duyệt
Truy cập: **http://127.0.0.1:5000/**

---

## Xử Lý Lỗi Thường Gặp

### Lỗi 1: "python is not recognized"
**Nguyên nhân:** Python chưa được thêm vào PATH

**Giải pháp:**
1. Gỡ cài đặt Python
2. Cài lại và **nhớ tích** "Add Python to PATH"
3. Hoặc thêm PATH thủ công:
   - Mở System Properties → Environment Variables
   - Thêm `C:\Users\Administrator\AppData\Local\Programs\Python\Python312` vào PATH

### Lỗi 2: "pip install" bị lỗi
**Giải pháp:**
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Lỗi 3: "underthesea" cài đặt lâu
**Nguyên nhân:** Package này khá lớn (cần tải model tiếng Việt)

**Giải pháp:** Đợi kiên nhẫn, hoặc cài từng package:
```powershell
pip install flask numpy scikit-learn matplotlib networkx requests
pip install underthesea
```

### Lỗi 4: Port 5000 đã được sử dụng
**Giải pháp:** Sửa file `app.py` dòng cuối:
```python
app.run(debug=True, port=5001)  # Đổi sang port 5001
```

---

## Kiểm Tra Nhanh

Sau khi cài xong, chạy test:
```powershell
python tests\test_sample.py
```

Nếu thấy:
```
✓ Sentence splitting test passed
✓ Tokenization test passed
✓ TF-IDF pipeline test passed
✓ TextRank pipeline test passed
✓ Multi-doc ranking test passed

✅ All tests passed!
```

→ **Hệ thống hoạt động hoàn hảo!** 🎉

---

## Phiên Bản Python Khuyến Nghị

| Phiên Bản | Trạng Thái | Ghi Chú |
|-----------|------------|---------|
| **Python 3.12** | ✅ Khuyến nghị | Mới nhất, ổn định |
| **Python 3.11** | ✅ Tốt | Nhanh hơn 3.10 |
| **Python 3.10** | ✅ OK | Tương thích tốt |
| Python 3.9 | ⚠️ Cũ | Vẫn hoạt động nhưng nên nâng cấp |
| Python 3.8 trở xuống | ❌ Không khuyến nghị | Quá cũ |

---

## Tóm Tắt Các Lệnh

```powershell
# 1. Kiểm tra Python
python --version

# 2. Di chuyển vào thư mục dự án
cd C:\Users\Administrator\Downloads\NLP\Tai

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Chạy ứng dụng
python app.py

# 5. Mở browser
# Truy cập: http://127.0.0.1:5000/
```

---

## Cần Hỗ Trợ?

Nếu gặp vấn đề, hãy:
1. Kiểm tra lại từng bước
2. Đảm bảo đã tích "Add Python to PATH"
3. Khởi động lại PowerShell sau khi cài Python
4. Kiểm tra kết nối internet (để tải packages)

**Chúc bạn cài đặt thành công!** 🚀

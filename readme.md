# 🐱🐶 Dog & Cat Deep Learning Project

Dự án xây dựng hệ thống nhận dạng chó và mèo bằng các kỹ thuật Deep Learning trong lĩnh vực Thị giác Máy tính (Computer Vision). Mục tiêu của dự án là nghiên cứu, triển khai và đánh giá hiệu năng của nhiều mô hình khác nhau trên hai bài toán chính:

* **Image Classification (Phân loại ảnh)**
* **Object Detection (Phát hiện đối tượng)**

Thông qua việc triển khai từ các mô hình CNN cơ bản đến các kiến trúc hiện đại như Faster R-CNN và YOLOv8, dự án giúp so sánh hiệu năng, tốc độ và khả năng ứng dụng thực tế của từng phương pháp trên cùng một bộ dữ liệu chó mèo.

---

# 🎯 Nội dung dự án

## 1. Image Classification

Bài toán Classification nhằm xác định một ảnh đầu vào thuộc lớp **Dog (Chó)** hay **Cat (Mèo)**.

Dự án triển khai và đánh giá 4 mô hình phân loại:

* Custom CNN
* ResNet50
* MobileNetV2
* EfficientNet-B0

Các mô hình được đánh giá dựa trên các chỉ số:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 2. Object Detection

Bài toán Object Detection không chỉ xác định đối tượng là chó hay mèo mà còn xác định chính xác vị trí của chúng trong ảnh thông qua Bounding Box.

Dự án triển khai 3 phương pháp phát hiện đối tượng:

### Sliding Window + CNN (Thủ công)

Phương pháp quét cửa sổ trượt (Sliding Window) kết hợp với mô hình CNN Classification để phát hiện đối tượng. Đây là phương pháp được xây dựng thủ công nhằm minh họa nguyên lý cơ bản của bài toán Object Detection trước khi áp dụng các mô hình phát hiện hiện đại.

### Faster R-CNN

Mô hình Two-Stage Detector sử dụng Region Proposal Network (RPN) để đề xuất vùng chứa đối tượng trước khi thực hiện phân loại và hồi quy Bounding Box.

### YOLOv8s

Mô hình One-Stage Detector thuộc họ YOLO, tối ưu cho tốc độ suy luận nhanh trong khi vẫn duy trì độ chính xác cao.

Các mô hình Detection được đánh giá bằng:

* mAP@0.5
* mAP@0.5:0.95
* AP theo từng lớp
* Visualization Bounding Box

---

# 📊 Dataset

Dự án sử dụng bộ dữ liệu ảnh chó và mèo phục vụ cho cả hai bài toán **Image Classification** và **Object Detection**.

## Thông tin chung

* Tổng số ảnh: **2029**
* Số lớp: **2**

  * Dog (Chó)
  * Cat (Mèo)

Bộ dữ liệu được sử dụng xuyên suốt trong toàn bộ dự án nhằm đảm bảo tính nhất quán khi so sánh hiệu năng giữa các mô hình.

---

## Phân chia dữ liệu

Dữ liệu được chia thành ba tập:

| Tập dữ liệu   |      Cat |      Dog |     Tổng |
| ------------- | -------: | -------: | -------: |
| Train         |      804 |      818 |     1622 |
| Validation    |      101 |      102 |      203 |
| Test          |      101 |      103 |      204 |
| **Tổng cộng** | **1006** | **1023** | **2029** |

Có thể thấy dữ liệu được phân bố khá cân bằng giữa hai lớp Dog và Cat, giúp hạn chế hiện tượng mất cân bằng dữ liệu (Class Imbalance) trong quá trình huấn luyện.

---

## Image Classification Dataset

Trong bài toán Classification, mỗi ảnh chỉ chứa một nhãn duy nhất:

* Dog
* Cat

Dữ liệu được tổ chức theo cấu trúc thư mục:

```text
images/
├── train/
│   ├── Cat/
│   └── Dog/
├── val/
│   ├── Cat/
│   └── Dog/
└── test/
    ├── Cat/
    └── Dog/
```

Mục tiêu của bài toán là xác định ảnh đầu vào thuộc lớp Dog hay Cat.

---

## Object Detection Dataset

Trong bài toán Object Detection, mỗi ảnh được gán nhãn:

* Class Label (Dog hoặc Cat)
* Bounding Box

Dữ liệu được lưu dưới định dạng YOLO và được chuyển đổi tự động sang định dạng phù hợp khi huấn luyện Faster R-CNN.

### Định dạng nhãn YOLO

Mỗi ảnh tương ứng với một file `.txt`.

Cấu trúc mỗi dòng:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Trong đó:

* `class_id`

  * 0: Dog
  * 1: Cat
* `x_center`, `y_center`: tọa độ tâm Bounding Box
* `width`, `height`: kích thước Bounding Box

Các giá trị tọa độ được chuẩn hóa về khoảng `[0,1]`.

Ví dụ:

```text
0 0.512500 0.463281 0.325000 0.412500
```

---

### Định dạng Bounding Box của Faster R-CNN

Trước khi huấn luyện Faster R-CNN, nhãn YOLO được chuyển đổi sang dạng tọa độ pixel:

```text
[x_min, y_min, x_max, y_max]
```

Ví dụ:

```text
[120, 80, 350, 420]
```

Trong đó:

* `x_min`, `y_min`: góc trên bên trái Bounding Box
* `x_max`, `y_max`: góc dưới bên phải Bounding Box

---

## 📁 Cấu trúc thư mục

```text
project/
├── main.ipynb                  # File notebook chính
├── requirements.txt            # Danh sách thư viện
├── dl_modules/
│   ├── data.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   ├── fine_tuning.py
│   ├── tuning.py
│   ├── visualization.py
│   └── manual_object_detection.py
└── README.md
└── training_log.txt
```

---

## 🚀 Cách chạy trên Google Colab (khuyến nghị)

Google Colab cung cấp GPU miễn phí, phù hợp nhất để chạy dự án này.

### Bước 1 — Mở notebook

1. Truy cập [colab.research.google.com](https://colab.research.google.com)
2. Chọn **File → Upload notebook** → chọn file `main.ipynb`

### Bước 2 — Bật GPU

1. Vào **Runtime → Change runtime type**
2. Chọn **Hardware accelerator: GPU (T4)**
3. Nhấn **Save**

### Bước 3 — Upload các file module

> Đây là bước bắt buộc trước khi chạy bất kỳ cell nào.

1. Nhìn sang **thanh bên trái**, click icon 📁 (Files)
2. Click **chuột phải vào vùng trống** trong ổ đĩa
3. Chọn **"Upload"**
4. Chọn **toàn bộ các file trong thư mục `dl_modules/`** (chọn nhiều file cùng lúc bằng Ctrl+Click)
5. Chờ upload xong — các file sẽ xuất hiện trong thư mục `/content/`

> ⚠️ Lưu ý: Nếu runtime bị ngắt kết nối (disconnect), bạn phải upload lại các file module từ đầu vì Colab không lưu file local giữa các session.

### Bước 4 — Cài thư viện

Chạy cell đầu tiên trong notebook (Section 1), notebook sẽ tự động cài:

```python
!pip install optuna torchmetrics ultralytics
```

Hoặc cài thủ công toàn bộ từ `requirements.txt`:

```python
!pip install -r requirements.txt
```

### Bước 5 — Mount Google Drive

Notebook sẽ yêu cầu kết nối Google Drive để lưu model và kết quả. Chạy cell mount và làm theo hướng dẫn xác thực hiện ra.

### Bước 6 — Chạy notebook

Chạy tuần tự từ trên xuống: **Runtime → Run all**, hoặc chạy từng cell bằng **Shift + Enter**.

---

## 💻 Cách chạy trên máy cục bộ (Local)

"Chạy local" nghĩa là chạy trực tiếp trên máy tính của bạn, không cần internet sau khi đã cài xong. Có 2 cách: dùng **Jupyter Notebook** (giao diện trình duyệt) hoặc **VS Code** (phần mềm editor).

### Yêu cầu hệ thống

- Python 3.9 trở lên — tải tại [python.org](https://www.python.org/downloads/)
- GPU có CUDA (khuyến nghị) — không có GPU vẫn chạy được nhưng rất chậm
- RAM tối thiểu 8GB

### Bước 1 — Tải project về máy

Giải nén toàn bộ file project vào một thư mục, ví dụ `C:\DogCatProject\` (Windows) hoặc `~/DogCatProject/` (macOS/Linux).

### Bước 2 — Mở terminal / command prompt

- **Windows**: nhấn `Win + R` → gõ `cmd` → Enter
- **macOS**: mở **Terminal** trong Applications
- Dùng lệnh `cd` để di chuyển vào thư mục project:

```bash
cd C:\DogCatProject       # Windows
cd ~/DogCatProject        # macOS / Linux
```

### Bước 3 — Tạo môi trường ảo

Môi trường ảo giúp cài thư viện riêng cho project, không ảnh hưởng đến máy tính.

```bash
# Tạo môi trường ảo tên "venv"
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (macOS / Linux)
source venv/bin/activate
```

Sau khi kích hoạt thành công, terminal sẽ hiện `(venv)` ở đầu dòng.

### Bước 4 — Cài thư viện

```bash
pip install -r requirements.txt
```

Lệnh này đọc file `requirements.txt` và tự động cài tất cả thư viện cần thiết, không cần cài từng cái một.

### Bước 5 — Cài PyTorch đúng phiên bản CUDA (nếu có GPU)

Truy cập [pytorch.org/get-started](https://pytorch.org/get-started/locally/) để lấy lệnh cài đúng với CUDA version của máy. Ví dụ với CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Bước 6 — Điều chỉnh path trong notebook

Các cell liên quan đến Google Drive sẽ báo lỗi khi chạy local. Comment các dòng sau:

```python
# from google.colab import drive
# drive.mount('/content/drive')
```

Và chỉnh `WORKSPACE_DIR` thành thư mục local:

```python
WORKSPACE_DIR = "./DogCat_Workspace"   # thay vì /content/drive/MyDrive/...
```

---

### 🌐 Cách A — Chạy bằng Jupyter Notebook (giao diện trình duyệt)

Jupyter Notebook cho phép chạy notebook ngay trên trình duyệt web, không cần cài thêm phần mềm nào khác.

**Cài Jupyter** (nếu chưa có):

```bash
pip install notebook
```

**Mở notebook:**

```bash
jupyter notebook main.ipynb
```

Trình duyệt sẽ tự động mở tại địa chỉ `http://localhost:8888`. Chạy từng cell bằng **Shift + Enter** hoặc chọn **Cell → Run All**.

---

### 🖥️ Cách B — Chạy bằng VS Code

VS Code là phần mềm editor có hỗ trợ chạy notebook trực tiếp, tiện hơn cho việc chỉnh sửa code.

**Bước 1** — Tải VS Code tại [code.visualstudio.com](https://code.visualstudio.com/) và cài đặt.

**Bước 2** — Mở VS Code, vào **Extensions** (icon 4 ô vuông bên trái), tìm và cài 2 extension:
- `Python` (của Microsoft)
- `Jupyter` (của Microsoft)

**Bước 3** — Mở thư mục project: **File → Open Folder** → chọn thư mục `DogCatProject`.

**Bước 4** — Mở file `main.ipynb`, VS Code sẽ hiển thị giao diện notebook.

**Bước 5** — Chọn đúng Python interpreter (môi trường ảo vừa tạo):
1. Nhấn **Ctrl + Shift + P** → gõ `Python: Select Interpreter`
2. Chọn interpreter có chữ `venv` trong đường dẫn

**Bước 6** — Chạy notebook bằng nút **▶ Run All** ở trên cùng hoặc nhấn **Shift + Enter** từng cell.

---

## 📦 Giải thích file requirements.txt

File `requirements.txt` là danh sách tất cả thư viện cần thiết cho dự án, mỗi dòng là một thư viện kèm phiên bản tối thiểu. Cách dùng rất đơn giản:

```bash
# Cài tất cả thư viện trong file
pip install -r requirements.txt

# Nếu dùng Colab thì thêm dấu ! phía trước
!pip install -r requirements.txt
```

Thay vì phải nhớ và cài từng thư viện một, chỉ cần chạy 1 lệnh là xong.

---

## ❗ Các lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-------------|------------|
| `ModuleNotFoundError: No module named 'data'` | Chưa upload file module | Upload lại toàn bộ file trong `dl_modules/` lên Colab |
| `CUDA out of memory` | GPU không đủ VRAM | Giảm `batch_size` xuống 2 hoặc 1 |
| `FileNotFoundError: frcnn_warmup.pth` | Runtime bị reset | Chạy lại từ cell Phase A |
| `No module named 'google.colab'` | Đang chạy local | Comment các cell `drive.mount` |

import os
import glob
from skimage.feature import hog
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import shutil
import numpy as np
import cv2
import random
from pathlib import Path



def flatten_image_folders(
    img_dir,
    splits=("train", "val", "test"),
    remove_empty_dirs=True,
    verbose=True
):
    """
    Công dụng:
        Phẳng hóa cấu trúc thư mục chứa ảnh bằng cách đưa toàn bộ ảnh từ các thư mục con phân lớp 
        ra ngoài thư mục phân tách dữ liệu gốc (ví dụ: train/). Thường dùng để chuẩn bị dữ liệu dạng object detection thô.

    Tham số truyền vào:
        - img_dir (str hoặc Path): Đường dẫn đến thư mục gốc chứa tập dữ liệu.
        - splits (tuple): Danh sách các phân đoạn dữ liệu cần xử lý. Mặc định là ("train", "val", "test").
        - remove_empty_dirs (bool): Có tự động xóa thư mục con phân lớp sau khi đã dời hết ảnh hay không. Mặc định là True.
        - verbose (bool): Có hiển thị tiến trình chi tiết ra màn hình console hay không. Mặc định là True.

    Output:
        - None
    """

    img_dir = Path(img_dir)

    for split in splits:

        split_dir = img_dir / split

        if not split_dir.exists():
            if verbose:
                print(f"Không tìm thấy: {split_dir}")
            continue

        moved_count = 0

        for class_dir in split_dir.iterdir():

            if not class_dir.is_dir():
                continue

            if verbose:
                print(f"Processing: {class_dir.name}")

            for img_file in class_dir.iterdir():

                if not img_file.is_file():
                    continue

                dst_file = split_dir / img_file.name

                if dst_file.exists():
                    print(f"Bỏ qua (trùng tên): {img_file.name}")
                    continue

                shutil.move(str(img_file), str(dst_file))
                moved_count += 1

            if remove_empty_dirs:
                try:
                    class_dir.rmdir()
                except OSError:
                    pass

        if verbose:
            print(f"---{split}: moved {moved_count} files")

    if verbose:
        print("---> Hoàn tất flatten dataset.")


def organize_dataset(base_path):
    """
    Công dụng:
        Tự động phân loại và di chuyển các tệp ảnh nằm ở thư mục phân đoạn (train, val, test) 
        vào các thư mục con phân lớp tương ứng ('Cat' hoặc 'Dog') dựa trên ký tự tiền tố của 
        tên tệp (ví dụ: 'cat_01.jpg' vào thư mục 'Cat', 'cho_02.jpg' vào thư mục 'Dog'). 
        Thường dùng để chuyển đổi sang dạng ImageFolder phục vụ bài toán Classification.

    Tham số truyền vào:
        - base_path (str): Đường dẫn đến thư mục gốc của tập dữ liệu chứa các nhánh train/val/test.

    Output:
        - None
    """

    splits = ['train', 'val', 'test']
    categories = ['Cat', 'Dog']

    for split in splits:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            continue

        for cat in categories:
            os.makedirs(os.path.join(split_path, cat), exist_ok=True)

        files = [f for f in os.listdir(split_path) if os.path.isfile(os.path.join(split_path, f))]

        move_count = 0
        for f in files:
            src = os.path.join(split_path, f)
            fname = f.lower()
            if fname.startswith('cat') or fname.startswith('meo'):
                dest = os.path.join(split_path, 'Cat', f)
                shutil.move(src, dest)
                move_count += 1
            elif fname.startswith('dog') or fname.startswith('cho'):
                dest = os.path.join(split_path, 'Dog', f)
                shutil.move(src, dest)
                move_count += 1

        print(f"Moved {move_count} images for the {split} set.")


def get_dataloaders(data_dir, batch_size=32, num_workers=2, pin_memory=True):
    """
    Công dụng:
        Khởi tạo các đối tượng Dataset và DataLoader của PyTorch cho cả 3 tập (train, val, test) 
        phục vụ bài toán phân lớp ảnh (Image Classification). Đồng thời áp dụng các kỹ thuật 
        tăng cường dữ liệu (Data Augmentation) phù hợp cho tập train và chuẩn hóa ảnh (Normalize) 
        theo các chỉ số của bộ ImageNet cho cả 3 tập.

    Tham số truyền vào:
        - data_dir (str): Đường dẫn đến thư mục chứa dữ liệu cấu trúc dạng ImageFolder (gồm train, val, test).
        - batch_size (int): Số lượng mẫu ảnh trong một batch dữ liệu. Mặc định là 32.
        - num_workers (int): Số luồng CPU phụ trợ dùng để nạp và tiền xử lý dữ liệu. Mặc định là 2.
        - pin_memory (bool): Nếu True, DataLoader sẽ sao chép Tensors vào bộ nhớ Page-locked của RAM, giúp tăng tốc độ đẩy dữ liệu lên GPU. Mặc định là True.

    Output:
        Trả về một tuple gồm 4 thành phần:
        - dataloaders (dict): Dictionary chứa các DataLoader tương ứng với key ["train", "val", "test"].
        - dataset_sizes (dict): Dictionary chứa tổng số lượng mẫu ảnh của từng tập dữ liệu.
        - class_names (list): Danh sách tên các nhãn/lớp (ví dụ: ['Cat', 'Dog']).
        - image_datasets (dict): Dictionary chứa các đối tượng Dataset gốc thuộc class 'datasets.ImageFolder'.
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    valid_extensions = (
        ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm",
        ".tif", ".tiff", ".webp", ".img",
    )

    def is_valid_image(path):
        if not path.lower().endswith(valid_extensions):
            return False

        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    image_datasets = {
        "train": datasets.ImageFolder(
            os.path.join(data_dir, "train"),
            train_transforms,
            is_valid_file=is_valid_image,
        ),
        "val": datasets.ImageFolder(
            os.path.join(data_dir, "val"),
            val_test_transforms,
            is_valid_file=is_valid_image,
        ),
        "test": datasets.ImageFolder(
            os.path.join(data_dir, "test"),
            val_test_transforms,
            is_valid_file=is_valid_image,
        ),
    }

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            image_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val", "test"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names, image_datasets



#=============================== EXTRACTION FEATURE HOG ===================================
def load_hog_data(data_path, num_samples=500):
    """
    Công dụng:
        Đọc các ảnh từ thư mục phân lớp, thực hiện chuyển đổi sang ảnh xám, thay đổi 
        kích thước về 64x64 và trích xuất đặc trưng Histogram of Oriented Gradients (HOG) 
        từ ảnh. Thường dùng để chuẩn bị ma trận đặc trưng đầu vào cho các mô hình Machine 
        Learning truyền thống như SVM.

    Tham số truyền vào:
        - data_path (str): Đường dẫn tới thư mục chứa các thư mục con phân lớp (ví dụ: Cat, Dog).
        - num_samples (int, tùy chọn): Số lượng mẫu ảnh tối đa cần lấy ra từ mỗi lớp để trích xuất đặc trưng. Mặc định là 500.

    Output:
        Trả về một tuple gồm 2 mảng NumPy:
        - features (np.ndarray): Mảng 2 chiều lưu trữ các vector đặc trưng HOG của toàn bộ các mẫu được đọc.
        - labels (np.ndarray): Mảng 1 chiều chứa nhãn số nguyên tương ứng của các mẫu ảnh
    """
    features = []
    labels = []
    categories = ['Cat', 'Dog']

    for category in categories:
        path = os.path.join(data_path, category)
        if not os.path.exists(path):
            continue

        label = categories.index(category)
        files = os.listdir(path)[:num_samples]

        for img_name in files:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (64, 64))
            fd          = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=False)
            features.append(fd)
            labels.append(label)

    return np.array(features), np.array(labels)

#================================ MANUAL)OBJECT_DETECTION============================
def load_rgb_image(image_path: Path):
    """
    Công dụng:
        Đọc một tệp ảnh từ ổ cứng, đảm bảo ảnh ở hệ màu RGB tiêu chuẩn và trả về dưới 
        cả hai định dạng phổ biến: Đối tượng Image của thư viện PIL và mảng ma trận NumPy.

    Tham số truyền vào:
        - image_path (Path hoặc str): Đường dẫn đến tệp tin hình ảnh cần đọc.

    Output:
        Trả về một tuple gồm 2 thành phần:
        - image_pil (PIL.Image.Image): Đối tượng hình ảnh định dạng PIL phục vụ cho tiền xử lý hoặc Torchvision.
        - image_rgb (np.ndarray): Ma trận điểm ảnh 3 chiều định dạng NumPy (H x W x C) hệ màu RGB phục vụ cho OpenCV/Matplotlib.
    """

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_pil = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image_pil)
    return image_pil, image_rgb

# ====================================================================================
# FASTER R-CNN — DATA MODULE

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".img", ".tif", ".tiff")


class FrcnnAugTransform:
    """
    Công dụng:
        Lớp thực hiện xử lý tăng cường dữ liệu đồng thời (Đồng bộ hóa) cho bài toán 
        Object Detection. Khi ảnh bị lật ngang hoặc lật dọc ngẫu nhiên, tọa độ các 
        khung bao (bounding boxes) nằm trong 'target' cũng sẽ tự động được tính toán 
        và biến đổi lại cho chính xác theo hệ trục tọa độ mới của ảnh.
    """

    def __init__(
        self,
        hflip_prob   = 0.5,
        vflip_prob   = 0.3,
        color_jitter = True,
        brightness   = 0.3,
        contrast     = 0.3,
        saturation   = 0.3,
    ):
        """
        Công dụng:
            Khởi tạo các tham số và xác suất cho quy trình tăng cường ảnh và hộp bao.

        Tham số truyền vào:
            - hflip_prob (float, tùy chọn): Xác suất lật ngang ảnh. Mặc định là 0.5.
            - vflip_prob (float, tùy chọn): Xác suất lật dọc ảnh. Mặc định là 0.3.
            - color_jitter (bool, tùy chọn): Có kích hoạt biến đổi màu sắc ngẫu nhiên hay không. Mặc định là True.
            - brightness (float, tùy chọn): Biên độ thay đổi độ sáng tối ngẫu nhiên. Mặc định là 0.3.
            - contrast (float, tùy chọn): Biên độ thay đổi độ tương phản ngẫu nhiên. Mặc định là 0.3.
            - saturation (float, tùy chọn): Biên độ thay đổi độ bão hòa màu sắc ngẫu nhiên. Mặc định là 0.3.

        Output:
            - None.
        """

        self.hflip_prob   = hflip_prob
        self.vflip_prob   = vflip_prob
        self.color_jitter = color_jitter
        self.brightness   = brightness
        self.contrast     = contrast
        self.saturation   = saturation

    def __call__(self, image: torch.Tensor, target: dict):
        """
        Công dụng:
            Áp dụng các phép biến đổi tăng cường dữ liệu tuần tự lên ảnh Tensor và hiệu 
            chỉnh lại mảng tọa độ Bounding Box tương ứng, đồng thời loại bỏ các box bị lỗi/hỏng sau biến đổi.

        Tham số truyền vào:
            - image (torch.Tensor): Tensor hình ảnh đầu vào có kích thước (C x H x W).
            - target (dict): Dictionary chứa thông tin nhãn của ảnh, bao gồm key "boxes" lưu trữ tọa độ Tensor dạng [x1, y1, x2, y2].

        Output:
            Trả về một tuple gồm 2 thành phần đã được đồng bộ hóa đặc trưng:
            - image (torch.Tensor): Tensor hình ảnh mới sau khi tăng cường.
            - target (dict): Dictionary chứa thông tin các nhãn, tọa độ boxes, diện tích area mới đã được cập nhật lại theo ảnh.
        """

        import random as _random
        _, H, W = image.shape

        #------1. RANDOM HORIZONTAL FLIP --------------------------
        # Lật ngang: cột x đổi chiều
        #   x1_new = W - x2_old
        #   x2_new = W - x1_old
        if _random.random() < self.hflip_prob:
            image = TF.hflip(image)
            if len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, 0] = W - target["boxes"][:, 2]  # x1_new = W - x2_old
                boxes[:, 2] = W - target["boxes"][:, 0]  # x2_new = W - x1_old
                target["boxes"] = boxes

        #------2. RANDOM VERTICAL FLIP--------------------------------
        # Lật dọc: hàng y đổi chiều
        #   y1_new = H - y2_old
        #   y2_new = H - y1_old
        if _random.random() < self.vflip_prob:
            image = TF.vflip(image)
            if len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, 1] = H - target["boxes"][:, 3]  # y1_new = H - y2_old
                boxes[:, 3] = H - target["boxes"][:, 1]  # y2_new = H - y1_old
                target["boxes"] = boxes

        #------3. COLOR JITTER -------------------------------------
        # Chỉ thay đổi giá trị pixel, không ảnh hưởng tọa độ box
        if self.color_jitter:
            bf = _random.uniform(1 - self.brightness, 1 + self.brightness)
            image = TF.adjust_brightness(image, bf)

            cf = _random.uniform(1 - self.contrast, 1 + self.contrast)
            image = TF.adjust_contrast(image, cf)

            sf = _random.uniform(1 - self.saturation, 1 + self.saturation)
            image = TF.adjust_saturation(image, sf)

        # ---- Clamp pixel về [0, 1] phòng jitter làm tràn số ----
        image = torch.clamp(image, 0.0, 1.0)

        # ---- Clamp box vào trong biên ảnh ---------------------------
        if len(target["boxes"]) > 0:
            boxes = target["boxes"]
            boxes[:, 0].clamp_(min=0, max=W)   # x1
            boxes[:, 1].clamp_(min=0, max=H)   # y1
            boxes[:, 2].clamp_(min=0, max=W)   # x2
            boxes[:, 3].clamp_(min=0, max=H)   # y2

            # Lọc box bị degenerate sau clamp (x1>=x2 hoặc y1>=y2)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            target["boxes"]   = boxes[keep]
            target["labels"]  = target["labels"][keep]
            target["iscrowd"] = target["iscrowd"][keep]

            # Tính lại diện tích box sau augmentation
            boxes = target["boxes"]
            target["area"] = (
                (boxes[:, 2] - boxes[:, 0]) *
                (boxes[:, 3] - boxes[:, 1])
            )

        return image, target


class FrcnnDataset(Dataset):
    """
    Công dụng:
        Tùy biến lớp Dataset của PyTorch dành riêng cho mô hình Faster R-CNN (Object Detection). 
        Thực hiện quét cặp tệp tin (Ảnh và nhãn tọa độ dạng text định dạng chuẩn), đọc dữ liệu 
        tọa độ hộp bao [x1, y1, x2, y2], gán nhãn lớp dịch chuyển tăng thêm 1 đơn vị (chừa lớp 
        0 cho Background nội bộ của Faster R-CNN) và đóng gói thành cấu trúc Tensor Dictionary chuẩn đầu vào.
    """

    def __init__(self, img_dir: str, label_dir: str, transforms=None):
        """
        Công dụng:
            Khởi tạo đối tượng Dataset, thiết lập đường dẫn và lập chỉ mục các mẫu dữ liệu hợp lệ.

        Tham số truyền vào:
            - img_dir (str): Thư mục chứa các tệp hình ảnh (.jpg, .png,...).
            - label_dir (str): Thư mục chứa các file nhãn tọa độ văn bản tương ứng (.txt).
            - transforms (callable, tùy chọn): Hàm hoặc lớp thực hiện biến đổi/ tăng cường dữ liệu (ví dụ: FrcnnAugTransform).

        Output:
            - None.
        """

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.samples = self._frcnn_build_index()

    def _frcnn_find_image(self, stem: str):
        """
        Công dụng:
            Tìm kiếm đường dẫn tệp ảnh tương ứng dựa trên tên gốc (stem name) của file nhãn, 
            kiểm tra tuần tự qua các phần mở rộng ảnh phổ biến để tìm file thực tế tồn tại.

        Tham số truyền vào:
            - stem (str): Tên file không bao gồm phần mở rộng (ví dụ: 'image_001').

        Output:
            - (str): Đường dẫn tuyệt đối/tương đối đầy đủ đến file ảnh nếu tìm thấy.
            - (None): Trả về None nếu không tìm thấy file ảnh nào khớp với tên nhãn.
        """
        for ext in IMG_EXTENSIONS:
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.exists(p):
                return p
        return None

    def _frcnn_build_index(self):
        """
        Công dụng:
            Quét toàn bộ thư mục nhãn `.txt`, tìm tệp ảnh tương ứng tương thích và thiết lập 
            danh sách các cặp (đường dẫn ảnh, đường dẫn nhãn) hợp lệ để phục vụ truy xuất theo chỉ mục.

        Tham số truyền vào:
            - Không có tham số truyền vào (Sử dụng thuộc tính nội bộ của lớp).

        Output:
            - samples (list): Danh sách chứa các tuple dạng (img_path, txt_label_path) của toàn bộ tập dữ liệu.
        """
        samples = []
        for txt in sorted(glob.glob(os.path.join(self.label_dir, "*.txt"))):
            stem = os.path.splitext(os.path.basename(txt))[0]
            img_path = self._frcnn_find_image(stem)
            if img_path is None:
                continue
            samples.append((img_path, txt))
        return samples

    def __len__(self):
        """
        Công dụng:
            Trả về tổng số lượng mẫu dữ liệu (cặp ảnh-nhãn) hợp lệ có trong Dataset.
        Output:
            - (int): Tổng số lượng phần tử.
        """
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Công dụng:
            Truy xuất, đọc thông tin và tiền xử lý một mẫu dữ liệu dựa trên chỉ mục (index). 
            Chuyển đổi ảnh thành Tensor, đọc file tọa độ hộp bao, tính diện tích hộp bao và 
            đóng gói thành định dạng Tensor Tuple/Dictionary tương thích với Faster R-CNN của torchvision.

        Tham số truyền vào:
            - idx (int): Chỉ mục của phần tử cần lấy.

        Output:
            Trả về một tuple gồm 2 thành phần:
            - img_tensor (torch.Tensor): Tensor hình ảnh sau chuyển đổi/tăng cường có kích thước (C x H x W).
            - target (dict): Dictionary chứa các thuộc tính Tensor của nhãn:
                - "boxes": Tensor (N x 4) chứa tọa độ dạng [x1, y1, x2, y2].
                - "labels": Tensor (N,) chứa mã lớp đối tượng (đã cộng thêm 1).
                - "area": Tensor (N,) chứa diện tích của từng hộp bao đối tượng.
                - "iscrowd": Tensor (N,) chứa nhãn đánh dấu phân biệt đám đông (mặc định toàn bộ là 0).
                - "image_id": Tensor (1,) chứa chỉ mục nguyên của ảnh trong luồng dữ liệu.
        """
        img_path, lbl_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img_tensor = TF.to_tensor(img)

        boxes, labels = [], []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0]) + 1  # 0→1(cho), 1→2(meo); 0 dành cho background
                x1, y1, x2, y2 = map(float, parts[1:5])
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)

        if len(boxes) == 0:
            boxes   = torch.zeros((0, 4), dtype=torch.float32)
            labels  = torch.zeros((0,),   dtype=torch.int64)
            area    = torch.zeros((0,),   dtype=torch.float32)
            iscrowd = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes   = torch.as_tensor(boxes,  dtype=torch.float32)
            labels  = torch.as_tensor(labels, dtype=torch.int64)
            area    = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "area":     area,
            "iscrowd":  iscrowd,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def frcnn_collate_fn(batch):
    """
    Công dụng:
        Hàm ghép nhóm (collate function) tùy chỉnh được truyền vào DataLoader. Do đặc trưng của 
        bài toán Object Detection, các bức ảnh có thể có số lượng khung bao đối tượng khác nhau 
        (kích thước 'target' không đồng đều), hàm này ép cấu trúc batch thành một Tuple chứa các 
        phần tử tách rời thay vì gộp ma trận mặc định (mặc định sẽ báo lỗi kích thước Tensor).

    Tham số truyền vào:
        - batch (list): Danh sách các tuple (image, target) được sinh ra từ lớp Dataset tương ứng với batch_size.

    Output:
        - (tuple): Trả về một Tuple gồm 2 Tuple con: Tuple thứ nhất gom tất cả các Tensor ảnh, Tuple thứ hai gom tất cả các dict nhãn của batch đó.
    """
    return tuple(zip(*batch))


def frcnn_get_dataloaders(img_dir: str, label_dir: str, batch_size: int = 4,
                          num_workers: int = 2):
    """
    Công dụng:
        Khởi tạo các đối tượng FrcnnDataset và cấu hình hệ thống các DataLoader chuyên dụng 
        cho bài toán Object Detection bằng mô hình Faster R-CNN cho cả 3 tập (train, val, test), 
        đồng thời liên kết hàm gom cụm dữ liệu không đồng đều 'frcnn_collate_fn'.

    Tham số truyền vào:
        - img_dir (str): Đường dẫn đến thư mục chứa dữ liệu hình ảnh của 3 tập phân đoạn.
        - label_dir (str): Đường dẫn đến thư mục chứa dữ liệu tệp nhãn văn bản (.txt) của 3 tập phân đoạn.
        - batch_size (int, tùy chọn): Số lượng mẫu ảnh trong một batch dữ liệu. Mặc định là 4.
        - num_workers (int, tùy chọn): Số luồng CPU phụ trợ dùng để nạp dữ liệu từ ổ đĩa. Mặc định là 2.

    Output:
        Trả về một tuple gồm 3 thành phần:
        - frcnn_dataloaders (dict): Dictionary chứa các DataLoader tương ứng với key ["train", "val", "test"].
        - dataset_sizes (dict): Dictionary chứa thông tin tổng số mẫu ảnh đọc được của từng tập dữ liệu.
        - frcnn_datasets (dict): Dictionary chứa các đối tượng Dataset gốc thuộc lớp tự định nghĩa 'FrcnnDataset'.
    """

    frcnn_datasets = {
        split: FrcnnDataset(
            img_dir=os.path.join(img_dir, split),
            label_dir=os.path.join(label_dir, split),
        )
        for split in ("train", "val", "test")
    }

    frcnn_dataloaders = {
        "train": DataLoader(
            frcnn_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=frcnn_collate_fn,
            pin_memory=True,
        ),
        "val": DataLoader(
            frcnn_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=frcnn_collate_fn,
        ),
        "test": DataLoader(
            frcnn_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=frcnn_collate_fn,
        ),
    }

    dataset_sizes = {split: len(frcnn_datasets[split]) for split in frcnn_datasets}

    return frcnn_dataloaders, dataset_sizes, frcnn_datasets

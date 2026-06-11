

from __future__ import annotations

import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from typing import Dict, List, Tuple

from models import build_model  # file chứa build_model và CustomCNN


# --------------------------------------------------------------------------- #
#  Hằng số                                                                     #
# --------------------------------------------------------------------------- #
LABELS      = ["dog", "cat"]   # index 0 = dog, 1 = cat
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

PREPROCESS  = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# --------------------------------------------------------------------------- #
#  Lớp Predictor                                                               #
# --------------------------------------------------------------------------- #
class Predictor:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        num_classes: int = 2,
        device: torch.device | None = None,
    ):
        """
        Khởi tạo đối tượng Predictor và nạp mô hình đã được huấn luyện.

        Hàm xây dựng kiến trúc mạng tương ứng với tên mô hình được chỉ định,
        sau đó nạp trọng số từ file .pth và chuyển mô hình sang chế độ đánh giá
        (evaluation mode). Nếu người dùng không chỉ định thiết bị thực thi,
        hàm sẽ tự động lựa chọn GPU nếu khả dụng, ngược lại sẽ sử dụng CPU.

        Parameters:
            model_name : str
                Tên mô hình cần sử dụng để dự đoán.
            model_path : str
                Đường dẫn tới file trọng số đã huấn luyện.
            num_classes : int, optional
                Số lớp đầu ra của mô hình.
            device : torch.device | None, optional
                Thiết bị dùng để thực hiện suy luận.

        Returns:
            None
                Hàm khởi tạo đối tượng Predictor và nạp mô hình vào bộ nhớ.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = build_model(
            model_name,
            num_classes=num_classes,
            is_feature_extraction=False,
            device=device,
        )

        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Đã load thành công {model_name} từ {model_path}")

    # ----------------------------------------------------------------------- #
    #  Inference                                                                #
    # ----------------------------------------------------------------------- #
    @torch.no_grad()
    def predict_batch(self, crops_rgb: List[np.ndarray]) -> np.ndarray:
        """
        Thực hiện dự đoán trên nhiều ảnh cùng lúc.

        Hàm tiền xử lý toàn bộ ảnh đầu vào, chuyển chúng thành tensor,
        thực hiện suy luận bằng mô hình và tính xác suất dự đoán cho từng lớp
        thông qua hàm Softmax.

        Parameters:
            crops_rgb : List[np.ndarray]
                Danh sách các ảnh RGB cần dự đoán.

        Returns:
            np.ndarray
                Ma trận xác suất có kích thước (N, num_classes), trong đó
                mỗi hàng tương ứng với phân phối xác suất của một ảnh.
        """
        tensors = [PREPROCESS(Image.fromarray(c).convert("RGB")) for c in crops_rgb]
        inputs  = torch.stack(tensors).to(self.device)
        probs   = torch.softmax(self.model(inputs), dim=1)
        return probs.cpu().numpy()

    @torch.no_grad()
    def predict_one(self, crop_rgb: np.ndarray) -> Dict:
        """
        Thực hiện phân loại một ảnh duy nhất.

        Hàm sử dụng mô hình để dự đoán xác suất của từng lớp, xác định lớp có
        xác suất cao nhất và trả về kết quả dưới dạng từ điển bao gồm nhãn,
        độ tin cậy và xác suất của tất cả các lớp.

        Parameters:
            crop_rgb : np.ndarray
                Ảnh RGB cần dự đoán.

        Returns:
            Dict
                Từ điển chứa nhãn dự đoán, độ tin cậy của dự đoán và xác suất
                tương ứng của từng lớp.
        """
        probs    = self.predict_batch([crop_rgb])[0]
        class_id = int(np.argmax(probs))
        return {
            "label": LABELS[class_id] if class_id < len(LABELS) else f"class_{class_id}",
            "score": float(probs[class_id]),
            "probs": {
                (LABELS[i] if i < len(LABELS) else f"class_{i}"): float(probs[i])
                for i in range(len(probs))
            },
        }

    # ----------------------------------------------------------------------- #
    #  Sliding window detection                                                 #
    # ----------------------------------------------------------------------- #
    def detect(
        self,
        image_rgb: np.ndarray,
        window_sizes: List[Tuple[int, int]] | None = None,
        scales: List[float] | None = None,
        step_ratio: float = 0.25,
        score_threshold: float = 0.90,
        iou_threshold: float = 0.35,
        batch_size: int = 64,
    ) -> List[Dict]:
        """
        Phát hiện đối tượng trong ảnh bằng phương pháp Sliding Window.

        Hàm quét ảnh ở nhiều kích thước cửa sổ và nhiều tỉ lệ khác nhau,
        sau đó sử dụng mô hình phân loại để đánh giá từng vùng ảnh. Các vùng
        có độ tin cậy cao sẽ được giữ lại và tiếp tục xử lý bằng thuật toán
        Non-Maximum Suppression nhằm loại bỏ các bounding box trùng lặp.

        Parameters:
            image_rgb : np.ndarray
                Ảnh RGB đầu vào.
            window_sizes : List[Tuple[int, int]] | None, optional
                Danh sách kích thước cửa sổ quét.
            scales : List[float] | None, optional
                Danh sách tỉ lệ resize ảnh.
            step_ratio : float, optional
                Tỉ lệ bước dịch chuyển của cửa sổ trượt.
            score_threshold : float, optional
                Ngưỡng xác suất tối thiểu để giữ lại ứng viên.
            iou_threshold : float, optional
                Ngưỡng IoU dùng trong Non-Maximum Suppression.
            batch_size : int, optional
                Số lượng vùng ảnh xử lý trong mỗi lần suy luận.

        Returns:
            List[Dict]
                Danh sách các đối tượng được phát hiện cùng bounding box,
                nhãn và độ tin cậy tương ứng.
        """
        if window_sizes is None:
            window_sizes = [(128, 128), (192, 192), (256, 256)]
        if scales is None:
            scales = [0.5, 0.75, 1.0]

        candidates = self._sliding_window_scan(
            image_rgb, window_sizes, scales, step_ratio, score_threshold, batch_size
        )
        return self._non_max_suppression(candidates, iou_threshold)

    # ----------------------------------------------------------------------- #
    #  Visualize                                                                #
    # ----------------------------------------------------------------------- #
    def draw(
        self,
        image_rgb: np.ndarray,
        detections: List[Dict],
        title: str = "Detections",
    ) -> None:
        """
        Hiển thị kết quả phát hiện đối tượng trên ảnh.

        Hàm vẽ bounding box và nhãn dự đoán lên ảnh đầu vào, sau đó hiển thị
        kết quả bằng thư viện Matplotlib để phục vụ việc trực quan hóa và
        đánh giá chất lượng phát hiện.

        Parameters:
            image_rgb : np.ndarray
                Ảnh RGB cần hiển thị.
            detections : List[Dict]
                Danh sách các đối tượng đã được phát hiện.
            title : str, optional
                Tiêu đề của hình ảnh hiển thị.

        Returns:
            None
                Hàm chỉ hiển thị kết quả và không trả về giá trị.
        """
        canvas = image_rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label, score    = det["label"], det["score"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 220, 40), 3)
            cv2.putText(
                canvas, f"{label} {score:.2f}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (20, 220, 40), 2, cv2.LINE_AA,
            )
        plt.figure(figsize=(11, 8))
        plt.imshow(canvas)
        plt.axis("off")
        plt.title(title)
        plt.show()

    # ----------------------------------------------------------------------- #
    #  Private helpers                                                          #
    # ----------------------------------------------------------------------- #
    def _sliding_window_scan(
        self,
        image_rgb: np.ndarray,
        window_sizes: List[Tuple[int, int]],
        scales: List[float],
        step_ratio: float,
        score_threshold: float,
        batch_size: int,
    ) -> List[Dict]:
        """
        Quét toàn bộ ảnh bằng kỹ thuật Sliding Window ở nhiều tỉ lệ khác nhau.

        Hàm sinh các vùng ảnh ứng viên từ nhiều kích thước cửa sổ và nhiều
        mức scale khác nhau, sau đó gom các vùng ảnh thành từng batch để
        thực hiện suy luận hiệu quả hơn. Các ứng viên đạt ngưỡng xác suất sẽ
        được lưu lại để tiếp tục xử lý.

        Parameters:
            image_rgb : np.ndarray
                Ảnh RGB đầu vào.
            window_sizes : List[Tuple[int, int]]
                Danh sách kích thước cửa sổ quét.
            scales : List[float]
                Danh sách tỉ lệ resize ảnh.
            step_ratio : float
                Tỉ lệ bước dịch chuyển của cửa sổ.
            score_threshold : float
                Ngưỡng xác suất giữ lại ứng viên.
            batch_size : int
                Kích thước batch dùng cho suy luận.

        Returns:
            List[Dict]
                Danh sách các bounding box ứng viên đạt ngưỡng xác suất.
        """
        candidates = []

        for scale in scales:
            scaled = self._resize_by_scale(image_rgb, scale)
            inv    = 1.0 / scale

            for win_w, win_h in window_sizes:
                if scaled.shape[1] < win_w or scaled.shape[0] < win_h:
                    continue

                step         = max(8, int(min(win_w, win_h) * step_ratio))
                batch_crops  = []
                batch_boxes  = []

                for x, y, crop in self._sliding_windows(scaled, (win_w, win_h), step):
                    batch_crops.append(crop)
                    batch_boxes.append((x, y, x + win_w, y + win_h))

                    if len(batch_crops) == batch_size:
                        candidates += self._filter_candidates(batch_crops, batch_boxes, inv, score_threshold)
                        batch_crops, batch_boxes = [], []

                if batch_crops:
                    candidates += self._filter_candidates(batch_crops, batch_boxes, inv, score_threshold)

        return candidates

    def _filter_candidates(
        self,
        crops: List[np.ndarray],
        boxes_scaled: List[Tuple],
        inv_scale: float,
        threshold: float,
    ) -> List[Dict]:
        """
        Lọc các vùng ảnh ứng viên dựa trên kết quả dự đoán của mô hình.

        Hàm thực hiện suy luận trên một nhóm vùng ảnh, xác định lớp dự đoán
        có xác suất cao nhất và chỉ giữ lại các ứng viên vượt qua ngưỡng
        độ tin cậy được chỉ định.

        Parameters:
            crops : List[np.ndarray]
                Danh sách ảnh crop cần dự đoán.
            boxes_scaled : List[Tuple]
                Danh sách bounding box tương ứng với từng ảnh crop.
            inv_scale : float
                Hệ số chuyển đổi tọa độ về ảnh gốc.
            threshold : float
                Ngưỡng xác suất tối thiểu để giữ lại ứng viên.

        Returns:
            List[Dict]
                Danh sách các đối tượng đạt điều kiện lọc.
        """
        results = []
        for (x1, y1, x2, y2), probs in zip(boxes_scaled, self.predict_batch(crops)):
            class_id = int(np.argmax(probs))
            score    = float(probs[class_id])
            if score >= threshold:
                results.append({
                    "box": [
                        int(x1 * inv_scale), int(y1 * inv_scale),
                        int(x2 * inv_scale), int(y2 * inv_scale),
                    ],
                    "label": LABELS[class_id] if class_id < len(LABELS) else f"class_{class_id}",
                    "score": score,
                })
        return results

    @staticmethod
    def _resize_by_scale(image: np.ndarray, scale: float) -> np.ndarray:
        """
        Thay đổi kích thước ảnh theo hệ số scale.

        Parameters:
            image : np.ndarray
                Ảnh đầu vào.
            scale : float
                Hệ số phóng to hoặc thu nhỏ ảnh.

        Returns:
            np.ndarray
                Ảnh sau khi được thay đổi kích thước.
        """
        h, w   = image.shape[:2]
        new_w  = max(1, int(w * scale))
        new_h  = max(1, int(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _sliding_windows(image: np.ndarray, window_size: Tuple[int, int], step: int):
        """
        Sinh tuần tự các cửa sổ trượt trên ảnh.

        Hàm duyệt qua toàn bộ ảnh với bước dịch chuyển cố định và trả về
        tọa độ cùng nội dung của từng cửa sổ ảnh được cắt ra.

        Parameters:
            image : np.ndarray
                Ảnh cần quét.
            window_size : Tuple[int, int]
                Kích thước của cửa sổ trượt.
            step : int
                Khoảng cách dịch chuyển giữa hai cửa sổ liên tiếp.

        Returns:
            Generator
                Sinh lần lượt tọa độ và vùng ảnh tương ứng của từng cửa sổ.
        """
        win_w, win_h = window_size
        h, w         = image.shape[:2]
        for y in range(0, h - win_h + 1, step):
            for x in range(0, w - win_w + 1, step):
                yield x, y, image[y:y + win_h, x:x + win_w]

    @staticmethod
    def _non_max_suppression(candidates: List[Dict], iou_threshold: float) -> List[Dict]:
        """
        Loại bỏ các bounding box chồng lấp bằng thuật toán Non-Maximum Suppression.

        Hàm giữ lại các bounding box có độ tin cậy cao nhất và loại bỏ các
        bounding box khác nếu mức độ giao nhau vượt quá ngưỡng IoU cho phép.

        Parameters:
            candidates : List[Dict]
                Danh sách các đối tượng ứng viên.
            iou_threshold : float
                Ngưỡng IoU dùng để loại bỏ bounding box trùng lặp.

        Returns:
            List[Dict]
                Danh sách bounding box sau khi đã được lọc.
        """
        candidates = sorted(candidates, key=lambda d: d["score"], reverse=True)
        kept = []
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [d for d in candidates if Predictor._iou(best["box"], d["box"]) < iou_threshold]
        return kept

    @staticmethod
    def _iou(box_a: List[int], box_b: List[int]) -> float:
        """
        Tính chỉ số Intersection over Union (IoU) giữa hai bounding box.

        IoU được sử dụng để đo mức độ chồng lấp giữa hai vùng dự đoán và là
        chỉ số quan trọng trong các bài toán phát hiện đối tượng.

        Parameters:
            box_a : List[int]
                Bounding box thứ nhất.
            box_b : List[int]
                Bounding box thứ hai.

        Returns:
            float
                Giá trị IoU nằm trong khoảng từ 0 đến 1.
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1           = max(ax1, bx1), max(ay1, by1)
        ix2, iy2           = min(ax2, bx2), min(ay2, by2)
        inter              = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union              = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0
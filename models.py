import torch
import torch.nn as nn
from torchvision import models


def set_parameter_requires_grad(model, feature_extracting):
    """
    Thiết lập trạng thái cập nhật trọng số của mô hình.

    Hàm được sử dụng trong quá trình transfer learning hoặc feature extraction.
    Khi feature_extracting được đặt là True, toàn bộ tham số của mô hình sẽ
    được đóng băng và không tham gia vào quá trình cập nhật gradient. Ngược lại,
    tất cả tham số sẽ được phép học và cập nhật trong quá trình huấn luyện.

    Parameters:
        model : torch.nn.Module
            Mô hình cần thiết lập trạng thái tham số.
        feature_extracting : bool
            Cờ xác định có sử dụng chế độ feature extraction hay không.

    Returns:
        None
            Hàm thay đổi trực tiếp thuộc tính requires_grad của các tham số
            trong mô hình.
    """
    for param in model.parameters():
        param.requires_grad = not feature_extracting


class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        """
        Khởi tạo kiến trúc mạng nơ-ron tích chập CustomCNN.

        Mô hình bao gồm bốn khối Convolution-BatchNorm-ReLU-MaxPooling nhằm
        trích xuất đặc trưng từ ảnh đầu vào. Sau phần trích xuất đặc trưng,
        một bộ phân loại gồm các tầng tuyến tính, Batch Normalization và
        Dropout được sử dụng để thực hiện phân loại ảnh.

        Parameters:
            num_classes : int, optional
                Số lượng lớp cần phân loại.

        Returns:
            None
                Hàm khởi tạo toàn bộ kiến trúc mạng và các thành phần bên trong.
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: Mở rộng từ 32 lên 64 channels
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: Mở rộng từ 64 lên 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: Mở rộng từ 128 lên 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4: Mở rộng từ 256 lên 512 channels để bắt các đặc trưng sâu
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            # Đầu vào lớp Tuyến tính bây giờ là: 512 channels * không gian 2x2 = 2048
            nn.Linear(512 * 2 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Thực hiện lan truyền xuôi qua mạng CustomCNN.

        Dữ liệu đầu vào được đưa qua phần trích xuất đặc trưng để tạo ra
        các đặc trưng mức cao, sau đó được chuyển đến bộ phân loại để tạo
        ra vector đầu ra tương ứng với số lớp cần dự đoán.

        Parameters:
            x : torch.Tensor
                Tensor ảnh đầu vào có kích thước (batch_size, channels, height, width).

        Returns:
            torch.Tensor
                Tensor chứa điểm dự đoán (logits) của từng lớp.
        """
        return self.classifier(self.features(x))

def build_model(model_name, num_classes=2, is_feature_extraction=True, device=None):
    """
    Khởi tạo mô hình học sâu theo tên được chỉ định.

    Hàm hỗ trợ nhiều kiến trúc khác nhau như ResNet50, EfficientNet-B0,
    MobileNetV2 và CustomCNN. Đối với các mô hình pretrained, lớp phân loại
    cuối sẽ được thay thế để phù hợp với số lớp của bài toán hiện tại.
    Ngoài ra, hàm cũng hỗ trợ chế độ feature extraction bằng cách đóng băng
    các tham số của backbone.

    Parameters:
        model_name : str
            Tên mô hình cần khởi tạo.
        num_classes : int, optional
            Số lượng lớp đầu ra của mô hình.
        is_feature_extraction : bool, optional
            Xác định có sử dụng chế độ feature extraction hay không.
        device : torch.device, optional
            Thiết bị dùng để lưu trữ và thực thi mô hình.

    Returns:
        torch.nn.Module
            Mô hình đã được khởi tạo và chuyển tới thiết bị chỉ định.
    """
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        set_parameter_requires_grad(model, is_feature_extraction)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        set_parameter_requires_grad(model, is_feature_extraction)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        set_parameter_requires_grad(model, is_feature_extraction)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "custom":
        model = CustomCNN(num_classes)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def load_model(model_name, path, num_classes=2, device=None):
    """
    Nạp mô hình đã được huấn luyện từ file trọng số.

    Hàm khởi tạo kiến trúc mô hình tương ứng, sau đó nạp các trọng số
    đã được lưu trước đó từ file .pth. Sau khi nạp thành công, mô hình
    được chuyển sang chế độ đánh giá để phục vụ suy luận.

    Parameters:
        model_name : str
            Tên mô hình cần nạp.
        path : str
            Đường dẫn tới file trọng số đã huấn luyện.
        num_classes : int, optional
            Số lượng lớp đầu ra của mô hình.
        device : torch.device, optional
            Thiết bị dùng để thực hiện suy luận.

    Returns:
        torch.nn.Module | None
            Trả về mô hình đã được nạp thành công. Nếu xảy ra lỗi,
            hàm trả về None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name, num_classes=num_classes, is_feature_extraction=False, device=device)

    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Đã load thành công {model_name} từ {path}")
        return model
    except Exception as e:
        print(f"Lỗi khi load model {model_name}: {e}")
        return None


# ════════════════════════════════════════════════════════════════
# FASTER R-CNN — MODELS MODULE
# ════════════════════════════════════════════════════════════════


import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def frcnn_set_parameter_requires_grad(model, freeze: bool):
    """
    Thiết lập trạng thái cập nhật tham số cho mô hình Faster R-CNN.

    Hàm được sử dụng trong quá trình fine-tuning để đóng băng hoặc mở khóa
    toàn bộ tham số của mô hình. Việc đóng băng backbone giúp tận dụng các
    đặc trưng đã học từ tập dữ liệu lớn, trong khi mở khóa toàn bộ mạng cho
    phép tinh chỉnh sâu hơn trên dữ liệu mới.

    Parameters:
        model : torchvision.models.detection.FasterRCNN
            Mô hình Faster R-CNN cần thiết lập trạng thái tham số.
        freeze : bool
            True để đóng băng tham số, False để cho phép cập nhật.

    Returns:
        None
            Hàm thay đổi trực tiếp thuộc tính requires_grad của mô hình.
    """
    for param in model.parameters():
        param.requires_grad = not freeze


def frcnn_get_discriminative_params(model, base_lr: float):
    """
    Tạo các nhóm tham số để áp dụng chiến lược Discriminative Learning Rate.

    Hàm tách các tham số của Faster R-CNN thành hai nhóm gồm backbone và
    head dự đoán. Backbone được gán learning rate nhỏ hơn nhằm bảo toàn
    các đặc trưng đã học trước đó, trong khi phần head sử dụng learning
    rate lớn hơn để thích nghi nhanh với tập dữ liệu mới.

    Parameters:
        model : torchvision.models.detection.FasterRCNN
            Mô hình Faster R-CNN cần lấy tham số.
        base_lr : float
            Learning rate cơ sở áp dụng cho phần head.

    Returns:
        list
            Danh sách các nhóm tham số theo định dạng yêu cầu của PyTorch
            Optimizer.
    """

    head_params     = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "box_predictor" in name or "roi_heads" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": base_lr / 10.0},
        {"params": head_params,     "lr": base_lr},
    ]


def frcnn_build_model(num_classes: int,
                      is_feature_extraction: bool = True,
                      device: torch.device = None) -> torchvision.models.detection.FasterRCNN:
    """
    Khởi tạo mô hình Faster R-CNN cho bài toán phát hiện đối tượng.

    Hàm tải mô hình Faster R-CNN ResNet50-FPN đã được huấn luyện trước,
    thay thế bộ phân loại cuối để phù hợp với số lớp của bài toán hiện tại,
    đồng thời hỗ trợ chế độ feature extraction hoặc fine-tuning toàn bộ mạng.

    Parameters:
        num_classes : int
            Tổng số lớp của mô hình, bao gồm cả lớp background.
        is_feature_extraction : bool, optional
            Xác định có sử dụng chế độ feature extraction hay không.
        device : torch.device, optional
            Thiết bị dùng để thực thi mô hình.

    Returns:
        torchvision.models.detection.FasterRCNN
            Mô hình Faster R-CNN đã được khởi tạo và sẵn sàng huấn luyện.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model   = fasterrcnn_resnet50_fpn(weights=weights)

    frcnn_set_parameter_requires_grad(model, freeze=is_feature_extraction)

    in_features = model.roi_heads.box_predictor.cls_score.in_features  # = 1024
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model.to(device)


def frcnn_print_model_summary(model):
    """
    Hiển thị thống kê số lượng tham số của mô hình Faster R-CNN.

    Hàm tính toán tổng số tham số của mô hình, số tham số có thể huấn luyện
    và số tham số đang bị đóng băng. Thông tin này giúp kiểm tra cấu hình
    feature extraction hoặc fine-tuning trước khi bắt đầu huấn luyện.

    Parameters:
        model : torchvision.models.detection.FasterRCNN
            Mô hình cần thống kê tham số.

    Returns:
        None
            Hàm chỉ hiển thị thông tin ra màn hình và không trả về giá trị.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"  Tổng params   : {total:>15,}")
    print(f"  Trainable     : {trainable:>15,}  ({trainable/total*100:.1f}%)")
    print(f"  Frozen        : {frozen:>15,}  ({frozen/total*100:.1f}%)")

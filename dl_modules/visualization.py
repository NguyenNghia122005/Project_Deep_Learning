import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import glob
import math
import matplotlib.patches as patches

from data import get_dataloaders
from models import build_model
from ultralytics import YOLO
import random



def plot_from_logs(model_name, workspace_dir):
    """
    Trực quan hóa lịch sử huấn luyện từ file log.

    Hàm đọc các thông tin loss và accuracy được ghi trong file training_log.txt,
    sau đó vẽ biểu đồ thể hiện sự thay đổi của train loss, validation loss,
    train accuracy và validation accuracy theo từng epoch nhằm hỗ trợ đánh giá
    quá trình fine-tuning của mô hình.

    Parameters:
        model_name : str
            Tên mô hình cần hiển thị kết quả huấn luyện.
        workspace_dir : str
            Thư mục chứa file training_log.txt.

    Returns:
        None
            Hàm hiển thị biểu đồ và không trả về giá trị.
    """
    log_path = os.path.join(workspace_dir, "training_log.txt")

    if not os.path.exists(log_path):
        print(f"Khong tim thay file log tai {log_path}")
        return

    epochs, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            print("File log dang trong.")
            return

        for line in lines:
            if f"FINE-TUNE {model_name}" in line:
                match = re.search(
                    r"Epoch (\d+).*?Loss\(Train/Val\): ([\d.]+)/([\d.]+).*?"
                    r"Acc\(Train/Val\): ([\d.]+)/([\d.]+)",
                    line,
                )
                if match:
                    epochs.append(int(match.group(1)))
                    train_losses.append(float(match.group(2)))
                    val_losses.append(float(match.group(3)))
                    train_accs.append(float(match.group(4)))
                    val_accs.append(float(match.group(5)))

    if not epochs:
        print(f"Khong tim thay du lieu khop cho {model_name}. Kiem tra lai dinh dang log.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Fine-tuning Progress: {model_name.upper()}", fontsize=14)

    ax1.plot(epochs, train_losses, "bo-", label="Train Loss")
    ax1.plot(epochs, val_losses, "ro-", label="Val Loss")
    ax1.set_title("Loss History")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, "b--", label="Train Acc")
    ax2.plot(epochs, val_accs, "go-", label="Val Acc")
    ax2.set_title("Accuracy History")
    ax2.legend()
    ax2.grid(True)

    plt.show()

def show_misclassified_examples(y_true, y_pred, data_path, title, num_examples=5, correct=True):
    """
    Hiển thị các ảnh được dự đoán đúng hoặc dự đoán sai.

    Hàm lựa chọn ngẫu nhiên một số mẫu từ tập dữ liệu dựa trên kết quả dự đoán
    của mô hình và hiển thị ảnh cùng nhãn thực tế, nhãn dự đoán để hỗ trợ
    phân tích chất lượng mô hình.

    Parameters:
        y_true : np.ndarray
            Nhãn thực tế của tập dữ liệu.
        y_pred : np.ndarray
            Nhãn dự đoán của mô hình.
        data_path : str
            Đường dẫn tới thư mục dữ liệu.
        title : str
            Tiêu đề hiển thị của biểu đồ.
        num_examples : int, optional
            Số lượng ảnh cần hiển thị.
        correct : bool, optional
            True để hiển thị các ảnh dự đoán đúng, False để hiển thị các ảnh dự đoán sai.

    Returns:
        None
            Hàm chỉ hiển thị kết quả trực quan.
    """
    indices = np.where((y_true == y_pred) if correct else (y_true != y_pred))[0]
    if len(indices) == 0:
        print(f"Không tìm thấy trường hợp {title}")
        return

    # Lấy danh sách đường dẫn ảnh thực tế để hiển thị
    all_image_paths = []
    for cat in ['Cat', 'Dog']:
        cat_path = os.path.join(data_path, 'test', cat)
        images = sorted(os.listdir(cat_path))
        all_image_paths.extend([os.path.join(cat_path, img) for img in images])

    selected_indices = np.random.choice(indices, min(num_examples, len(indices)), replace=False)

    plt.figure(figsize=(20, 5))
    for i, idx in enumerate(selected_indices):
        img_path = all_image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_name = 'Cat' if y_true[idx] == 0 else 'Dog'
        pred_name = 'Cat' if y_pred[idx] == 0 else 'Dog'
        color = 'green' if correct else 'red'

        plt.subplot(1, num_examples, i + 1)
        plt.imshow(img)
        plt.title(f"True: {label_name}\nPred: {pred_name}", color=color, fontsize=11, y=1.02)
        plt.axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.08)
    plt.tight_layout()
    plt.show()



def visualize_misclassified(model_name, data_dir, workspace_dir, device, num_images=3, subset="test"):
    """
    So sánh ảnh gốc và ảnh sau tiền xử lý đối với các trường hợp dự đoán sai.

    Hàm tải mô hình đã huấn luyện, tìm các ảnh bị phân loại sai trong tập dữ liệu,
    sau đó hiển thị đồng thời ảnh gốc và ảnh sau khi được tiền xử lý để phân tích
    nguyên nhân gây ra lỗi dự đoán.

    Parameters:
        model_name : str
            Tên mô hình cần đánh giá.
        data_dir : str
            Thư mục chứa dữ liệu.
        workspace_dir : str
            Thư mục chứa checkpoint của mô hình.
        device : torch.device
            Thiết bị thực thi mô hình.
        num_images : int, optional
            Số lượng ảnh lỗi cần hiển thị.
        subset : str, optional
            Tập dữ liệu cần kiểm tra, ví dụ "test" hoặc "val".

    Returns:
        None
            Hàm hiển thị kết quả và không trả về giá trị.
    """
    print(f"\n--- SO SANH ANH GOC VS ANH XU LY ({subset.upper()}): {model_name.upper()} ---")

    model = build_model(model_name, num_classes=2, is_feature_extraction=False, device=device)
    model_save_path = os.path.join(workspace_dir, "models", f"FINETUNED_best_{model_name}.pth")
    if not os.path.exists(model_save_path):
        print(f"Khong tim thay model tai {model_save_path}")
        return

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    _, _, class_names, image_datasets = get_dataloaders(data_dir, batch_size=1)
    dataset = image_datasets[subset]

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    misclassified_data = []
    with torch.no_grad():
        for i in range(len(dataset)):
            img_path, true_label = dataset.samples[i]
            raw_image = Image.open(img_path).convert("RGB")
            transformed_image = preprocess(raw_image)

            input_tensor = to_tensor_norm(transformed_image).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            pred_label = preds.item()

            if pred_label != true_label:
                misclassified_data.append({
                    "raw": raw_image,
                    "transformed": transformed_image,
                    "true": class_names[true_label],
                    "pred": class_names[pred_label],
                    "filename": os.path.basename(img_path),
                })

            if len(misclassified_data) >= num_images:
                break

    if not misclassified_data:
        print(f"Khong tim thay loi tren tap {subset}.")
        if subset == "test":
            visualize_misclassified(model_name, data_dir, workspace_dir, device, num_images, subset="val")
        return

    fig, axes = plt.subplots(len(misclassified_data), 2, figsize=(12, 5 * len(misclassified_data)))
    if len(misclassified_data) == 1:
        axes = [axes]

    for i, data in enumerate(misclassified_data):
        axes[i][0].imshow(data["raw"])
        axes[i][0].set_title(f"GOC: {data['filename']}\nNhan: {data['true']}")
        axes[i][0].axis("off")

        axes[i][1].imshow(data["transformed"])
        axes[i][1].set_title(
            f"XU LY (Model nhin): {data['transformed'].size}\nDu doan: {data['pred']}",
            color="red",
        )
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()


class GradCAM:
    def __init__(self, model, target_layer):
        """
        Khởi tạo đối tượng Grad-CAM.

        Hàm đăng ký các hook để lưu lại activation và gradient tại một tầng cụ thể
        của mô hình. Các thông tin này được sử dụng để sinh bản đồ nhiệt Grad-CAM
        phục vụ việc giải thích quyết định của mô hình.

        Parameters:
            model : torch.nn.Module
                Mô hình cần giải thích.
            target_layer : torch.nn.Module
                Tầng được sử dụng để tạo Grad-CAM.

        Returns:
            None
                Hàm khởi tạo đối tượng GradCAM.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, target_class=None):
        """
        Sinh bản đồ nhiệt Grad-CAM cho một ảnh đầu vào.

        Hàm thực hiện lan truyền xuôi và lan truyền ngược để xác định các vùng
        trên ảnh có ảnh hưởng lớn nhất tới quyết định của mô hình đối với lớp mục tiêu.

        Parameters:
            input_image : torch.Tensor
                Tensor ảnh đầu vào.
            target_class : int, optional
                Lớp mục tiêu cần giải thích. Nếu không chỉ định, hàm sử dụng
                lớp có xác suất dự đoán cao nhất.

        Returns:
            tuple
                Bao gồm bản đồ nhiệt Grad-CAM và chỉ số lớp được sử dụng.
        """
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * self.activations, dim=1).squeeze()
        grad_cam = F.relu(grad_cam)

        grad_cam = grad_cam.detach().cpu().numpy()
        grad_cam = cv2.resize(grad_cam, (224, 224))
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
        return grad_cam, target_class


def _get_target_layer(model, model_name):
    """
    Trả về target layer phù hợp cho Grad-CAM theo từng kiến trúc model.

    Mapping dựa trên build_model() trong models.py:
      - "resnet50"    : model.layer4[-1]
                        → Bottleneck cuối của stage layer4
      - "mobilenet"   : model.features[-1]
                        → Conv2dNormActivation cuối (MobileNetV2)
      - "efficientnet": model.features[-1]
                        → MBConv block cuối (EfficientNet-B0)
      - "custom"      : model.features[9]
                        → Conv2d(256→512) là conv cuối trước AdaptiveAvgPool2d
                           Layout của CustomCNN.features (flat Sequential):
                           [0] Conv2d 3→64   [1] BN  [2] ReLU  [3] MaxPool
                           [4] Conv2d 64→128 [5] BN  [6] ReLU  [7] MaxPool
                           [8] Conv2d 128→256[9]... wait — xem lại:
                           idx 0  Conv2d(3,64)
                           idx 1  BN64 / idx 2 ReLU / idx 3 MaxPool
                           idx 4  Conv2d(64,128)
                           idx 5  BN128 / idx 6 ReLU / idx 7 MaxPool
                           idx 8  Conv2d(128,256)
                           idx 9  BN256 / idx 10 ReLU / idx 11 MaxPool
                           idx 12 Conv2d(256,512)   ← TARGET
                           idx 13 BN512 / idx 14 ReLU / idx 15 MaxPool
                           idx 16 AdaptiveAvgPool2d
    Parameters:
        model : torch.nn.Module
            Mô hình cần xác định tầng mục tiêu.
        model_name : str
            Tên kiến trúc mô hình.

    Returns:
        torch.nn.Module
            Tầng được sử dụng để tạo Grad-CAM.
    """
    if model_name == "resnet50":
        # layer4[-1] = Bottleneck cuối, output shape (B,2048,H,W)
        return model.layer4[-1]

    if model_name == "mobilenet":
        # MobileNetV2: features[-1] = Conv2dNormActivation(320→1280)
        return model.features[-1]

    if model_name == "efficientnet":
        # EfficientNet-B0: features[-1] = Conv2dNormActivation cuối
        return model.features[-1]

    if model_name == "custom":
        # CustomCNN: features là flat Sequential 17 layers
        # Conv2d(256→512) nằm tại index 12 — lớp conv sâu nhất trước pool cuối
        return model.features[12]

    raise ValueError(
        f"Khong xac dinh duoc target layer cho model '{model_name}'. "
        "Cac model hop le: 'resnet50', 'mobilenet', 'efficientnet', 'custom'."
    )


def visualize_gradcam_multi(model_name, data_dir, workspace_dir, device, samples_per_class=5):
    """
    Hiển thị Grad-CAM trên nhiều ảnh thuộc các lớp khác nhau.

    Hàm tải mô hình đã huấn luyện, sinh bản đồ nhiệt Grad-CAM cho một số mẫu
    đại diện của từng lớp và hiển thị đồng thời ảnh gốc, heatmap và ảnh overlay
    để hỗ trợ giải thích hành vi của mô hình.

    Parameters:
        model_name : str
            Tên mô hình cần trực quan hóa.
        data_dir : str
            Thư mục chứa dữ liệu.
        workspace_dir : str
            Thư mục chứa checkpoint mô hình.
        device : torch.device
            Thiết bị thực thi mô hình.
        samples_per_class : int, optional
            Số lượng mẫu cần hiển thị cho mỗi lớp.

    Returns:
        None
            Hàm hiển thị kết quả trực quan và không trả về giá trị.
    """
    model = build_model(model_name, num_classes=2, is_feature_extraction=False, device=device)

    # Thử lần lượt các đường dẫn checkpoint
    candidate_paths = [
        os.path.join(workspace_dir, "models", f"GRADUAL_{model_name}_BEST.pth"),
        os.path.join(workspace_dir, "models", f"FINETUNED_best_{model_name}.pth"),
    ]
    model_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if model_path is None:
        print(f"Khong tim thay checkpoint cho '{model_name}' tai:\n" +
              "\n".join(f"  {p}" for p in candidate_paths))
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    print(f"[{model_name.upper()}] Da tai weights tu: {model_path}")

    # Xác định target layer
    try:
        target_layer = _get_target_layer(model, model_name)
    except ValueError as e:
        print(e)
        return
    print(f"[{model_name.upper()}] Target layer: {target_layer.__class__.__name__}")

    cam_extractor = GradCAM(model, target_layer)
    dataloaders, _, class_names, _ = get_dataloaders(data_dir, batch_size=1)

    # Thu thập mẫu đủ mỗi lớp
    class_samples = {idx: [] for idx in range(len(class_names))}
    for img, label in dataloaders["test"]:
        label_idx = label.item()
        if len(class_samples[label_idx]) < samples_per_class:
            class_samples[label_idx].append(img)
        if all(len(v) == samples_per_class for v in class_samples.values()):
            break

    all_samples = [
        (img, label_idx)
        for label_idx in sorted(class_samples.keys())
        for img in class_samples[label_idx]
    ]

    num_total = len(all_samples)
    if num_total == 0:
        print("Khong lay duoc mau nao tu dataloader.")
        return

    fig, axes = plt.subplots(num_total, 3, figsize=(15, 4 * num_total))
    fig.suptitle(f"Grad-CAM: {model_name.upper()}", fontsize=16, y=1.01)

    # Đảm bảo axes luôn là 2-D
    if num_total == 1:
        axes = [axes]

    for i, (img, label_idx) in enumerate(all_samples):
        img_tensor = img.to(device)
        mask, pred_idx = cam_extractor.generate(img_tensor)

        # Chuyển tensor ảnh về numpy (denormalize)
        orig_img = img.squeeze().permute(1, 2, 0).numpy()
        orig_img = (orig_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        orig_img = np.clip(orig_img, 0, 1)

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        combined = np.clip(0.5 * heatmap + 0.5 * orig_img, 0, 1)

        correct = (label_idx == pred_idx)
        pred_color = "green" if correct else "red"

        axes[i][0].imshow(orig_img)
        axes[i][0].set_title(f"Anh goc\nTrue: {class_names[label_idx]}")
        axes[i][0].axis("off")

        axes[i][1].imshow(heatmap)
        axes[i][1].set_title("Grad-CAM Heatmap")
        axes[i][1].axis("off")

        axes[i][2].imshow(combined)
        axes[i][2].set_title(f"Overlay\nPred: {class_names[pred_idx]}", color=pred_color)
        axes[i][2].axis("off")

    plt.tight_layout()
    plt.show()

# ════════════════════════════════════════════════════════════════
# FASTER R-CNN — VISUALIZATION MODULE
# ════════════════════════════════════════════════════════════════


FRCNN_CLASS_NAMES  = {0: "background", 1: "cho", 2: "meo"}
FRCNN_CLASS_COLORS = {1: "dodgerblue", 2: "tomato"}


def frcnn_show_batch(dataset, class_names=None, class_colors=None,
                     n: int = 6, title: str = "Mẫu dataset"):
    """
    Hiển thị ngẫu nhiên một số mẫu từ tập dữ liệu Faster R-CNN.

    Hàm trực quan hóa ảnh cùng với các bounding box và nhãn tương ứng nhằm
    kiểm tra chất lượng dữ liệu và xác nhận quá trình đọc dữ liệu diễn ra chính xác.

    Parameters:
        dataset : Dataset
            Dataset cần hiển thị.
        class_names : dict, optional
            Từ điển ánh xạ chỉ số lớp sang tên lớp.
        class_colors : dict, optional
            Từ điển ánh xạ lớp sang màu hiển thị.
        n : int, optional
            Số lượng ảnh cần hiển thị.
        title : str, optional
            Tiêu đề của biểu đồ.

    Returns:
        None
            Hàm hiển thị dữ liệu và không trả về giá trị.
    """
    if class_names is None:
        class_names  = FRCNN_CLASS_NAMES
    if class_colors is None:
        class_colors = FRCNN_CLASS_COLORS

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        img_tensor, target = dataset[idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        for box, lbl in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            color = class_colors.get(lbl.item(), "white")
            cls   = class_names.get(lbl.item(), "?")
            rect  = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, cls, color=color, fontsize=9,
                    bbox=dict(facecolor="black", alpha=0.5, pad=1))
        ax.axis("off")
        ax.set_title(f"idx={idx}")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def frcnn_show_augmentation_effect(dataset_no_aug, aug,
                                    class_names=None, class_colors=None,
                                    idx: int = None, n_aug: int = 3):
    """
    So sánh ảnh gốc và ảnh sau khi áp dụng augmentation.

    Hàm hiển thị nhiều phiên bản tăng cường dữ liệu của cùng một ảnh nhằm
    kiểm tra xem bounding box có được biến đổi chính xác theo ảnh hay không.

    Parameters:
        dataset_no_aug : Dataset
            Dataset chưa áp dụng augmentation.
        aug : callable
            Hàm augmentation cần kiểm tra.
        class_names : dict, optional
            Từ điển tên lớp.
        class_colors : dict, optional
            Từ điển màu lớp.
        idx : int, optional
            Chỉ số mẫu cần kiểm tra.
        n_aug : int, optional
            Số lượng lần augmentation cần hiển thị.

    Returns:
        None
            Hàm hiển thị kết quả trực quan.
    """
    import numpy as _np

    if class_names is None:
        class_names  = FRCNN_CLASS_NAMES
    if class_colors is None:
        class_colors = FRCNN_CLASS_COLORS

    if idx is None:
        idx = random.randint(0, len(dataset_no_aug) - 1)

    img_orig, tgt_orig = dataset_no_aug[idx]

    ncols = n_aug + 1
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    def _draw(ax, img, tgt, title):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = _np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        for box, lbl in zip(tgt["boxes"], tgt["labels"]):
            x1, y1, x2, y2 = box.tolist()
            color = class_colors.get(lbl.item(), "yellow")
            rect  = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 6, class_names.get(lbl.item(), "?"),
                    color=color, fontsize=9,
                    bbox=dict(facecolor="black", alpha=0.5, pad=1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Cột 0: ảnh gốc
    _draw(axes[0], img_orig, tgt_orig, f"Gốc (idx={idx})")

    # Cột 1..n: mỗi lần augment khác nhau
    for i in range(n_aug):
        img_aug, tgt_aug = aug(
            img_orig.clone(),
            {k: v.clone() for k, v in tgt_orig.items()},
        )
        _draw(axes[i + 1], img_aug, tgt_aug, f"Augmented #{i + 1}")

    plt.suptitle(
        "Kiểm tra Augmentation — box phải dịch chuyển đúng theo ảnh",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def frcnn_visualize_predictions(model, dataset, device,
                                 n: int = 12, score_thresh: float = 0.5,
                                 save_path: str = "predictions_test.png",
                                 indices_to_show=None):
    """
    Trực quan hóa kết quả dự đoán của mô hình Faster R-CNN.

    Hàm hiển thị đồng thời Ground Truth và các bounding box dự đoán trên ảnh,
    qua đó giúp đánh giá trực quan chất lượng phát hiện đối tượng của mô hình.

    Parameters:
        model : torchvision.models.detection.FasterRCNN
            Mô hình cần đánh giá.
        dataset : Dataset
            Dataset dùng để trực quan hóa.
        device : torch.device
            Thiết bị thực thi mô hình.
        n : int, optional
            Số lượng ảnh cần hiển thị.
        score_thresh : float, optional
            Ngưỡng confidence tối thiểu.
        save_path : str, optional
            Đường dẫn lưu ảnh kết quả.
        indices_to_show : list, optional
            Danh sách chỉ số ảnh cần hiển thị.

    Returns:
        None
            Hàm lưu và hiển thị ảnh kết quả.
    """
    model.eval()

    if indices_to_show is not None:
        indices = indices_to_show[:n]
    else:
        indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    ncols = 4
    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if nrows > 1 else axes if ncols == 1 else axes.flatten()

    for ax, idx in zip(axes, indices):
        img_tensor, target = dataset[idx]
        output = model([img_tensor.to(device)])[0]

        img_np = img_tensor.permute(1, 2, 0).numpy()
        ax.imshow(img_np)

        # Ground Truth (nét đứt xanh lá)
        for box, lbl in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor="lime", facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)
            ax.text(x1, y2 + 5, f"GT:{FRCNN_CLASS_NAMES.get(lbl.item(), '?')}",
                    color="lime", fontsize=7,
                    bbox=dict(facecolor="black", alpha=0.4, pad=1))

        # Predictions (nét liền, màu theo class)
        mask = output["scores"] >= score_thresh
        for box, lbl, score in zip(output["boxes"][mask],
                                    output["labels"][mask],
                                    output["scores"][mask]):
            x1, y1, x2, y2 = box.tolist()
            color = FRCNN_CLASS_COLORS.get(lbl.item(), "yellow")
            rect  = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 6, f"{FRCNN_CLASS_NAMES.get(lbl.item(), '?')} {score:.2f}",
                    color=color, fontsize=7,
                    bbox=dict(facecolor="black", alpha=0.5, pad=1))

        ax.axis("off")
        ax.set_title(f"idx={idx}", fontsize=8)

    for ax in axes[len(indices):]:
        ax.axis("off")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="lime",       linestyle="--", lw=2, label="Ground Truth"),
        Line2D([0], [0], color="dodgerblue", linestyle="-",  lw=2, label="Pred: Chó"),
        Line2D([0], [0], color="tomato",     linestyle="-",  lw=2, label="Pred: Mèo"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f"Faster R-CNN — Predictions (score ≥ {score_thresh})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Đã lưu: {save_path}")


@torch.no_grad()
def frcnn_find_potential_label_errors(model, dataset, device,
                                       score_thresh: float = 0.5,
                                       count_diff_thresh: int = 1):
    """
    Tìm kiếm các ảnh có khả năng bị gán nhãn sai.

    Hàm so sánh số lượng đối tượng và tập nhãn giữa Ground Truth và kết quả
    dự đoán của mô hình để xác định các mẫu dữ liệu bất thường có thể chứa lỗi nhãn.

    Parameters:
        model : torchvision.models.detection.FasterRCNN
            Mô hình dùng để kiểm tra dữ liệu.
        dataset : Dataset
            Dataset cần phân tích.
        device : torch.device
            Thiết bị thực thi mô hình.
        score_thresh : float, optional
            Ngưỡng confidence tối thiểu.
        count_diff_thresh : int, optional
            Sai lệch số lượng đối tượng cho phép.

    Returns:
        list
            Danh sách chỉ số các ảnh có khả năng chứa lỗi nhãn.
    """
    model.eval()
    problematic = []

    print(f"\nTìm kiếm ảnh sai nhãn trong {len(dataset)} mẫu...")

    for idx in range(len(dataset)):
        try:
            img_tensor, target = dataset[idx]
        except Exception:
            continue

        output = model([img_tensor.to(device)])[0]
        mask   = output["scores"] >= score_thresh

        gt_boxes    = target["boxes"]
        gt_labels   = target["labels"]
        pred_boxes  = output["boxes"][mask]
        pred_labels = output["labels"][mask]

        has_gt   = len(gt_boxes) > 0
        has_pred = len(pred_boxes) > 0
        is_bad   = False

        if not has_gt and has_pred:
            is_bad = True
        if has_gt and not has_pred:
            is_bad = True
        if abs(len(gt_boxes) - len(pred_boxes)) > count_diff_thresh:
            is_bad = True
        if has_gt and has_pred:
            gt_set   = set(gt_labels.tolist())
            pred_set = set(pred_labels.cpu().tolist())
            if gt_set != pred_set:
                is_bad = True

        if is_bad:
            problematic.append(idx)

    print(f"Tìm thấy {len(problematic)}/{len(dataset)} ảnh có khả năng sai nhãn.")
    return problematic


def frcnn_plot_training_history(history: dict, save_path: str = "training_curves.png"):
    """
    Trực quan hóa lịch sử huấn luyện của Faster R-CNN.

    Hàm vẽ các biểu đồ thể hiện sự thay đổi của loss, mAP và AP theo từng epoch,
    giúp đánh giá quá trình hội tụ và hiệu năng của mô hình.

    Parameters:
        history : dict
            Lịch sử huấn luyện được lưu trong quá trình train.
        save_path : str, optional
            Đường dẫn lưu ảnh biểu đồ.

    Returns:
        None
            Hàm lưu và hiển thị biểu đồ.
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    ax1.set_title("Loss theo epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["val_map50"], "g-o", label="mAP@0.5")
    ax2.plot(epochs, history["val_map"],   "r-s", label="mAP@0.5:0.95")
    ax2.set_title("Validation mAP theo epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")
    ax2.legend()
    ax2.grid(True)

    if "val_ap_cho" in history and history["val_ap_cho"]:
        ax3.plot(epochs, history["val_ap_cho"], "b-^", label="AP Chó")
        ax3.plot(epochs, history["val_ap_meo"], "r-v", label="AP Mèo")
        ax3.set_title("AP per Class (Val)")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("AP@0.5")
        ax3.legend()
        ax3.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Đã lưu: {save_path}")


# ════════════════════════════════════════════════════════════════
# YOLO
# ════════════════════════════════════════════════════════════════

def yolo_visualize(model, class_names, image_paths, confidence_threshold=0.4, show_prediction=True):
    """
    Trực quan hóa kết quả phát hiện đối tượng của YOLO.

    Hàm hiển thị đồng thời nhãn gốc và kết quả dự đoán của mô hình trên ảnh,
    qua đó hỗ trợ đánh giá trực quan chất lượng phát hiện đối tượng.

    Parameters:
        model : YOLO
            Mô hình YOLO đã huấn luyện.
        class_names : dict
            Từ điển ánh xạ chỉ số lớp sang tên lớp.
        image_paths : str | list
            Đường dẫn tới ảnh hoặc danh sách ảnh.
        confidence_threshold : float, optional
            Ngưỡng confidence tối thiểu.
        show_prediction : bool, optional
            Có hiển thị kết quả dự đoán hay không.

    Returns:
        None
            Hàm hiển thị ảnh trực quan.
    """

    # Nếu truyền vào 1 chuỗi đường dẫn đơn lẻ thay vì List, tự động bọc lại thành List
    if isinstance(image_paths, str):
        image_paths = [image_paths]
        
    num_images = len(image_paths)
    if num_images == 0:
        print("Danh sách ảnh truyền vào trống.")
        return

    # Tự động tạo subplot theo số lượng ảnh
    if num_images == 1:
        cols = 1
        rows = 1
        fig, ax_init = plt.subplots(1, figsize=(10, 8))
        axes = [ax_init]
    else:
        cols = min(3, num_images) # tối đa 3 ảnh trên 1 dòng
        rows = math.ceil(num_images / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten() # Ép phẳng ma trận để dễ dùng index 1 chiều

    # lặp qua từng ảnh và vẽ
    for i, image_path in enumerate(image_paths):
        ax = axes[i]
        
        # Đọc ảnh thực tế
        img_pil = Image.open(image_path).convert('RGB')
        W, H = img_pil.size
        ax.imshow(img_pil)
        
        # Đọc và vẽ khung label thực tế (màu đỏ) ---
        label_path = image_path.replace("images", "labels")
        label_path = os.path.splitext(label_path)[0] + ".txt"

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    x_center, y_center, w_norm, h_norm = map(float, parts[1:])
                    
                    # Quy đổi về pixel tương ứng của bức ảnh
                    box_w = w_norm * W
                    box_h = h_norm * H
                    x1_gt = (x_center * W) - (box_w / 2)
                    y1_gt = (y_center * H) - (box_h / 2)
                    
                    # Vẽ nét đứt màu đỏ
                    rect_gt = patches.Rectangle((x1_gt, y1_gt), box_w, box_h, 
                                                linewidth=1.5, edgecolor='red', linestyle='--', facecolor='none')
                    ax.add_patch(rect_gt)
                    
                    try:
                        class_name =  class_names[cls_id]
                    except NameError:
                        class_name = model.names[cls_id] if model is not None else f"Class {cls_id}"
                        
                    ax.text(x1_gt, y1_gt - 3, f"GT: {class_name}", color='white', fontsize=8 if num_images > 1 else 10,
                            bbox=dict(facecolor='red', alpha=0.6, edgecolor='none', pad=1))

        # Chạy model và vẽ khung prediction (Màu xanh lá)
        if show_prediction and model is not None:
            prediction_results = model.predict(image_path, conf=confidence_threshold, verbose=False)[0]
            boxes = prediction_results.boxes.xyxy.cpu().numpy()
            classes = prediction_results.boxes.cls.cpu().numpy()
            scores = prediction_results.boxes.conf.cpu().numpy()

            for box, cls_id, score in zip(boxes, classes, scores):
                x1_pred, y1_pred, x2_pred, y2_pred = box
                
                # Vẽ nét liền màu xanh lá
                rect_pred = patches.Rectangle((x1_pred, y1_pred), x2_pred - x1_pred, y2_pred - y1_pred, 
                                              linewidth=1.5, edgecolor='green', facecolor='none')
                ax.add_patch(rect_pred)
                
                try:
                    class_name =  class_names[int(cls_id)]
                except NameError:
                    class_name = model.names[int(cls_id)]
                    
                ax.text(x1_pred, y2_pred + 12, f"Pred: {class_name} {score:.2f}", color='white', fontsize=8 if num_images > 1 else 10,
                        bbox=dict(facecolor='green', alpha=0.6, edgecolor='none', pad=1))

        # Định dạng tiêu đề nhỏ cho từng ô
        ax.set_title(os.path.basename(image_path), fontsize=9 if num_images > 1 else 12)
        ax.axis('off')

    # Xóa trục của các ô trống (không có ảnh)
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    gt_patch = patches.Patch(color='red', label='Truth (Nhãn gốc)')
    pred_patch = patches.Patch(color='green', label='YOLOv8 Prediction (Dự đoán)')
    
    if show_prediction and model is not None:
        handles = [gt_patch, pred_patch]
        title_text = "SO SÁNH KHUNG NHÃN GỐC VỚI KẾT QUẢ DỰ ĐOÁN YOLOv8"
    else:
        handles = [gt_patch]
        title_text = "TRỰC QUAN HÓA KHUNG NHÃN GỐC BAN ĐẦU"

    # Đặt vị trí hiển thị Legend tùy thuộc vào số ảnh vẽ
    if num_images == 1:
        axes[0].legend(handles=handles, loc='upper right')
        plt.title(title_text, fontsize=13, fontweight='bold')
    else:
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=len(handles))
        plt.suptitle(title_text, fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
    plt.show()




def yolo_visualize_errors(model, image_dir, label_dir, class_names, 
                          confidence_threshold=0.4, max_images=6):
    """
    Phân tích các trường hợp dự đoán sai của mô hình YOLO.

    Hàm quét toàn bộ tập dữ liệu, tìm các ảnh có sự khác biệt giữa Ground Truth
    và kết quả dự đoán, sau đó hiển thị các trường hợp lỗi để hỗ trợ phân tích.

    Parameters:
        model : YOLO
            Mô hình YOLO đã huấn luyện.
        image_dir : str
            Thư mục chứa ảnh.
        label_dir : str
            Thư mục chứa nhãn.
        class_names : dict
            Từ điển ánh xạ chỉ số lớp sang tên lớp.
        confidence_threshold : float, optional
            Ngưỡng confidence tối thiểu.
        max_images : int, optional
            Số lượng ảnh lỗi tối đa cần hiển thị.

    Returns:
        None
            Hàm hiển thị các trường hợp lỗi và không trả về giá trị.
    """

    all_images = glob.glob(os.path.join(image_dir, '*'))
    if len(all_images) == 0:
        print(f"Không tìm thấy ảnh nào trong: {image_dir}")
        return

    error_images = []

    # Quét tập dữ liệu để tìm ra các ảnh lỗi
    for img_path in all_images:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(label_dir, base_name + '.txt')

        # Đếm số lượng vật thể gốc (GT) trong file.txt
        gt_count = 0
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                gt_count = len([line for line in f.readlines() if len(line.strip().split()) == 5])

        # Chạy model dự đoán trên ảnh này
        results = model.predict(img_path, conf=confidence_threshold, verbose=False)[0]
        pred_count = len(results.boxes)

        # Điều kiện lỗi: Số lượng GT (thực tế) khác với Pred
        # Hoặc nếu bằng nhau nhưng có sự lệch pha về nhãn (Ví dụ: GT là Cat, Pred là Dog)
        is_error = False
        if gt_count != pred_count:
            is_error = True
        elif gt_count > 0 and pred_count > 0:
            gt_classes = []
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5: gt_classes.append(int(parts[0]))
            
            pred_classes = results.boxes.cls.cpu().numpy().astype(int).tolist()
            # Sắp xếp để so sánh chéo các cặp nhãn
            if sorted(gt_classes) != sorted(pred_classes):
                is_error = True

        if is_error:
            error_images.append((img_path, txt_path))

    # Vẽ các ảnh lỗi
    num_errors = len(error_images)
    print(f"Tìm thấy tất cả {num_errors} ảnh bị đoán sai hoặc thiếu.")
    
    if num_errors == 0:
        print("Mô hình dự đoán đúng hoàn toàn, không tìm thấy ảnh lỗi nào.")
        return

    num_to_show = min(max_images, num_errors)
    sampled_errors = random.sample(error_images, num_to_show)

    cols = min(3, num_to_show)
    rows = math.ceil(num_to_show / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    
    if num_to_show == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (img_path, txt_path) in enumerate(sampled_errors):
        ax = axes[i]
        img_pil = Image.open(img_path).convert('RGB')
        W, H = img_pil.size
        ax.imshow(img_pil)

        # Vẽ khung label thực tế (màu đỏ)
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    x_center, y_center, w_norm, h_norm = map(float, parts[1:])
                    box_w, box_h = w_norm * W, h_norm * H
                    x1 = (x_center * W) - (box_w / 2)
                    y1 = (y_center * H) - (box_h / 2)
                    
                    rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1.5, edgecolor='red', linestyle='--', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 4, f"GT: {class_names.get(cls_id)}", color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.6, edgecolor='none', pad=1))

        # Vẽ khung prediction màu xanh lá
        results = model.predict(img_path, conf=confidence_threshold, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        for box, cls_id, score in zip(boxes, classes, scores):
            x1_p, y1_p, x2_p, y2_p = box
            rect_pred = patches.Rectangle((x1_p, y1_p), x2_p - x1_p, y2_p - y1_p, linewidth=1.5, edgecolor='green', facecolor='none')
            ax.add_patch(rect_pred)
            ax.text(x1_p, y2_p + 10, f"Pred: {class_names.get(int(cls_id))} {score:.2f}", color='white', fontsize=8, bbox=dict(facecolor='green', alpha=0.6, edgecolor='none', pad=1))

        ax.set_title(os.path.basename(img_path), fontsize=9, y = 1.03)
        ax.axis('off')

    for j in range(num_to_show, len(axes)):
        axes[j].axis('off')

    gt_patch = patches.Patch(color='red', label='Ground Truth (Nhãn gốc)')
    pred_patch = patches.Patch(color='green', label='YOLOv8 Prediction')
    fig.legend(handles=[gt_patch, pred_patch], loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=2, fontsize=10)
    
    plt.suptitle("DANH SÁCH CÁC TRƯỜNG HỢP DỰ ĐOÁN SAI CỦA MÔ HÌNH (ERROR ANALYSIS)", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.83)
    plt.show()
    plt.show()


def show_colab_image(image_path):
    """
    Hiển thị ảnh từ đường dẫn được chỉ định.
    Hàm đọc file ảnh từ hệ thống lưu trữ và hiển thị bằng Matplotlib.
    Nếu đường dẫn không tồn tại, hàm sẽ thông báo lỗi cho người dùng.

    Parameters:
        image_path : str
            Đường dẫn tới file ảnh cần hiển thị.

    Returns:
        None
            Hàm hiển thị ảnh hoặc thông báo lỗi nếu file không tồn tại.
    """
    if os.path.exists(image_path):
        img = Image.open(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print(f"Không tìm thấy file tại: {image_path}")


def show_group_subplots(result_dir, filenames, rows, cols, group_title, figsize=(15, 12)):
    """
    Gom nhóm và hiển thị nhiều ảnh kết quả YOLO dưới dạng Subplots.
    
    Parameters:
        - result_dir (str): Đường dẫn đến thư mục chứa kết quả (ví dụ: runs/detect/train)
        - filenames (list): Danh sách tên các file ảnh cần hiển thị trong nhóm.
        - rows (int): Số hàng của subplot.
        - cols (int): Số cột của subplot.
        - group_title (str): Tiêu đề lớn tổng quan của nhóm ảnh.
        - figsize (tuple): Kích thước khung hình hiển thị.
    Returns:
        None: Hàm sẽ hiển thị trực tiếp các ảnh dưới dạng subplot và không trả về giá trị nào.
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(group_title, fontsize=16, fontweight='bold', color='darkred')
    
    # Ép mảng axes về 1 chiều để dễ duyệt vòng lặp
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    
    for idx, filename in enumerate(filenames):
        img_full_path = os.path.join(result_dir, filename)
        
        if os.path.exists(img_full_path) and idx < len(axes_flat):
            img = Image.open(img_full_path)
            axes_flat[idx].imshow(img)
            axes_flat[idx].set_title(filename, fontsize=12, color='blue')
        else:
            if not os.path.exists(img_full_path):
                axes_flat[idx].text(0.5, 0.5, f"Missing:\n{filename}", 
                                    ha='center', va='center', color='red')
            
        axes_flat[idx].axis('off')
        
    # Ẩn các ô subplot thừa nếu danh sách file ít hơn số ô (rows * cols)
    for j in range(len(filenames), len(axes_flat)):
        axes_flat[j].axis('off')
        
    plt.tight_layout()
    plt.show()


def plot_yolo_custom_results(csv_path: str, save_plot_dir: str = None):
    """
    Công dụng:
        Đọc file results.csv của YOLOv8, trích xuất dữ liệu để vẽ lại các biểu đồ 
        so sánh hiệu năng giữa Train và Validation theo từng cụm cặp 2-2 ở kích thước lớn.

    Tham số truyền vào:
        - csv_path (str): Đường dẫn trực tiếp đến file 'results.csv' (ví dụ: '/content/workspace/runs/.../results.csv').
        - save_plot_dir (str, tùy chọn): Thư mục để lưu lại các file ảnh biểu đồ mới sau khi vẽ xong.

    Kết quả trả về (Output):
        - Không trả về biến, hiển thị trực tiếp các cụm biểu đồ subplots to, rõ nét lên màn hình Colab.
    """
    if not os.path.exists(csv_path):
        print(f"-- Cảnh báo: Không tìm thấy file dữ liệu CSV tại: {csv_path}")
        return

    # Read data và làm sạch tên cột (bỏ khoảng trắng thừa nếu có)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    epochs = df['epoch']

    # Hàm phụ trợ để cấu hình style chung cho các biểu đồ nhỏ trong cụm
    def format_ax(ax, title, xlabel, ylabel, is_loss=True):
        ax.set_title(title, fontsize=12, fontweight='bold', color='darkblue')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right' if is_loss else 'lower right')

    print("---Vẽ các biểu đồ đánh giá loss và mAP trong results---\n")

    # Subplot 1: SO SÁNH BOX LOSS (TRAIN VS VAL) - 1 HÀNG X 2 CỘT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phân tích sai số vị trí (box plot)", fontsize=14, fontweight='bold', color='darkred')
    
    ax1.plot(epochs, df['train/box_loss'], 'b-o', label='Train Box Loss', linewidth=2)
    format_ax(ax1, "Train Box Loss qua từng Epoch", "Epoch", "Loss Value")
    
    ax2.plot(epochs, df['val/box_loss'], 'r-o', label='Val Box Loss', linewidth=2)
    format_ax(ax2, "Validation Box Loss qua từng Epoch", "Epoch", "Loss Value")
    
    plt.tight_layout()
    if save_plot_dir:
        os.makedirs(save_plot_dir, exist_ok=True)
        plt.savefig(os.path.join(save_plot_dir, "custom_box_loss_comparison.png"), dpi=300)
    plt.show()
    print("-" * 90)

    # Subplot 2: SO SÁNH CLASS LOSS (TRAIN VS VAL)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phân tích sai số phân loại (class loss)", fontsize=14, fontweight='bold', color='darkred')
    
    ax1.plot(epochs, df['train/cls_loss'], 'b-s', label='Train Class Loss', linewidth=2)
    format_ax(ax1, "Train Class Loss qua từng Epoch", "Epoch", "Loss Value")
    
    ax2.plot(epochs, df['val/cls_loss'], 'r-s', label='Val Class Loss', linewidth=2)
    format_ax(ax2, "Validation Class Loss qua từng Epoch", "Epoch", "Loss Value")
    
    plt.tight_layout()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir, "custom_cls_loss_comparison.png"), dpi=300)
    plt.show()
    print("-" * 90)

    # Subplot 3: SO SÁNH DFL LOSS (TRAIN VS VAL)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phân tích sai số phân phối bounding box (DFL loss)", fontsize=14, fontweight='bold', color='darkred')
    
    ax1.plot(epochs, df['train/dfl_loss'], 'b-^', label='Train DFL Loss', linewidth=2)
    format_ax(ax1, "Train DFL Loss qua từng Epoch", "Epoch", "Loss Value")
    
    ax2.plot(epochs, df['val/dfl_loss'], 'r-^', label='Val DFL Loss', linewidth=2)
    format_ax(ax2, "Validation DFL Loss qua từng Epoch", "Epoch", "Loss Value")
    
    plt.tight_layout()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir, "custom_dfl_loss_comparison.png"), dpi=300)
    plt.show()
    print("-" * 90)

    # Subplot 4: PHÂN TÍCH CHỈ SỐ ĐỘ CHÍNH XÁC ĐẦU RA (METRICS mAP)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phân tích chỉ số đánh giá chất lượng mô hình (metric mAP)", fontsize=14, fontweight='bold', color='darkred')
    
    ax1.plot(epochs, df['metrics/mAP50(B)'], 'g-d', label='mAP@0.5', linewidth=2)
    ax1.plot(epochs, df['metrics/mAP50-95(B)'], 'm--', label='mAP@0.5:0.95', linewidth=2)
    format_ax(ax1, "Chỉ số mAP Đạt Được Qua Từng Epoch", "Epoch", "Score", is_loss=False)
    
    ax2.plot(epochs, df['metrics/precision(B)'], 'c-', label='Precision', linewidth=1.5)
    ax2.plot(epochs, df['metrics/recall(B)'], 'y-', label='Recall', linewidth=1.5)
    format_ax(ax2, "Biến thiên Precision và Recall", "Epoch", "Score", is_loss=False)
    
    plt.tight_layout()
    if save_plot_dir:
        plt.savefig(os.path.join(save_plot_dir, "custom_metrics_comparison.png"), dpi=300)
    plt.show()
    
    if save_plot_dir:
        print(f"Đã lưu thành công các file ảnh biểu đồ vào: {save_plot_dir}")


import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models import build_model
from ultralytics import YOLO


def test_model(model_name, dataloaders, class_names, workspace_dir, device):
    """
    Công dụng:
        Tiến hành đánh giá nghiệm thu mô hình phân lớp (Classification) trên tập dữ liệu Test. 
        Hàm thực hiện nạp trọng số đã finetune, dự đoán nhãn, tính toán độ chính xác tổng quan (Accuracy), 
        in báo cáo chi tiết (Precision, Recall, F1-score cho từng lớp) và trực quan hóa ma trận nhầm lẫn 
        (Confusion Matrix) dưới dạng biểu đồ nhiệt (Heatmap).

    Tham số truyền vào:
        - model_name (str): Tên của mô hình cần đánh giá 
        - dataloaders (dict): Dictionary chứa PyTorch DataLoader của tập nghiệm thu với key ["test"].
        - class_names (list): Danh sách chuỗi chứa tên các nhãn phân lớp 
        - workspace_dir (str): Đường dẫn đến thư mục làm việc, nơi lưu trữ tệp tin trọng số mô hình.
        - device (str hoặc torch.device): Thiết bị phần cứng thực thi tính toán

    Output:
        - None
    """
    print(f"\n{'*' * 20} NGHIEM THU MO HINH: {model_name.upper()} {'*' * 20}")

    model = build_model(model_name, num_classes=2, is_feature_extraction=False, device=device)
    model_path = os.path.join(workspace_dir, "models", f"FINETUNED_best_{model_name}.pth")

    if not os.path.exists(model_path):
        print(f"Canh bao: Khong tim thay file trong so {model_path}. Bo qua danh gia.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Do chinh xac tren tap Test: {acc * 100:.2f}%")
    print("\nChi tiet bao cao phan loai:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix Final: {model_name.upper()}")
    plt.xlabel("Du doan (Predicted)")
    plt.ylabel("Thuc te (True)")
    plt.show()


# ====================================================
# FASTER R-CNN — EVALUATION MODULE
# ====================================================


FRCNN_CLASS_NAMES  = {0: "background", 1: "cho", 2: "meo"}


def frcnn_test_model(workspace_dir: str, test_loader, dataset, device,
                     num_classes: int = 3, score_thresh: float = 0.5):
    """
    Công dụng:
        Tiến hành đánh giá nghiệm thu hiệu năng của mô hình phát hiện đối tượng Faster R-CNN 
        trên tập dữ liệu Test. Hàm tự động tìm kiếm và nạp tệp trọng số tốt nhất (ưu tiên bản finetune), 
        tính toán các chỉ số mAP (mean Average Precision) ở các ngưỡng IoU khác nhau (mAP@0.5, mAP@0.75, 
        mAP@0.5:0.95), phân tích độ chính xác theo kích thước vật thể (small, medium, large) và 
        vẽ biểu đồ thanh (ASCII bar) thể hiện AP chi tiết cho từng lớp đối tượng cụ thể.

    Tham số truyền vào:
        - workspace_dir (str): Đường dẫn đến thư mục làm việc, nơi chứa thư mục con 'models' lưu trọng số.
        - test_loader (DataLoader): PyTorch DataLoader chứa dữ liệu ảnh và nhãn bounding box của tập Test.
        - dataset (Dataset): Đối tượng Dataset gốc tương ứng của tập dữ liệu nghiệm thu.
        - device (str hoặc torch.device): Thiết bị phần cứng thực thi đánh giá ('cuda' hoặc 'cpu').
        - num_classes (int, tùy chọn): Tổng số lượng lớp bao gồm cả lớp nền background. Mặc định là 3.
        - score_thresh (float, tùy chọn): Ngưỡng điểm tin cậy (Confidence Score) để lọc các bounding box hợp lệ khi dự đoán. Mặc định là 0.5.

    Output:
        Trả về một tuple gồm 2 thành phần:
        - model (torch.nn.Module hoặc None): Đối tượng mô hình Faster R-CNN đã được nạp trọng số hoàn chỉnh (trả về None nếu không tìm thấy tệp trọng số).
        - test_result (dict hoặc None): Dictionary chứa các giá trị Tensor chỉ số mAP chi tiết thu được sau quá trình đánh giá (trả về None nếu gặp lỗi file).
    """

    from models import frcnn_build_model
    from training import frcnn_evaluate_map

    print(f"\n{'*'*20} NGHIỆM THU: FASTER R-CNN {'*'*20}")

    model_path = os.path.join(workspace_dir, "models", "FINETUNED_best_frcnn.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(workspace_dir, "models", "best_frcnn.pth")

    if not os.path.exists(model_path):
        print(f"--- Cảnh báo: không tìm thấy model tại {model_path}. Bỏ qua.")
        return None, None

    model = frcnn_build_model(num_classes=num_classes, is_feature_extraction=False, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_result = frcnn_evaluate_map(model, test_loader, device, score_thresh)

    def fmt(val):
        return "N/A (không có object)" if abs(val + 1.0) < 1e-4 else f"{val:.4f}"

    print("\n" + "="*55)
    print("  KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*55)
    print(f"  mAP@0.5          : {test_result['map_50'].item():.4f}")
    print(f"  mAP@0.5:0.95     : {test_result['map'].item():.4f}")
    print(f"  mAP@0.75         : {test_result['map_75'].item():.4f}")
    print(f"  mAP (small)      : {fmt(test_result['map_small'].item())}")
    print(f"  mAP (medium)     : {fmt(test_result['map_medium'].item())}")
    print(f"  mAP (large)      : {fmt(test_result['map_large'].item())}")

    if "map_per_class" in test_result:
        print("\n  AP per class (IoU=0.5:0.95):")
        for i, ap in enumerate(test_result["map_per_class"]):
            cls_name = FRCNN_CLASS_NAMES.get(i + 1, f"class_{i+1}")
            print(f"    [{cls_name:4s}] AP@0.5:0.95 = {ap.item():.4f}")

    print("="*55)
    return model, test_result




def yolo_test_model(
    workspace_dir: str, 
    data_yaml_path: str, 
    model_name: str = "yolov8s", 
    device: str = "cpu", 
    score_thresh: float = 0.25, 
    save_dir: str = None,
    stage_folder: str = "yolov8_stage2_finetuned",
    weights_folder: str = "weights",
    filename_pt: str = "best.pt",
    default_eval_folder: str = "yolo_test_evaluation"
):
    """
    Công dụng:
            Đánh giá hiệu năng mô hình YOLO trên tập dữ liệu Test qua file data.yaml,
            tính toán các chỉ số mAP và tự động xuất hệ thống ảnh biểu đồ nghiệm thu.

        Tham số truyền vào:
            - workspace_dir (str): Đường dẫn thư mục làm việc gốc (ví dụ: '/content/workspace').
            - data_yaml_path (str): Đường dẫn đến tệp cấu hình dữ liệu 'data.yaml'.
            - model_name (str): Tên biến thể mô hình YOLOv8. Mặc định là "yolov8s".
            - device (str/int): Thiết bị tính toán sử dụng ('cpu', 'cuda' hoặc ID như 0).
            - score_thresh (float): Ngưỡng điểm tin cậy lọc bounding box. Mặc định là 0.25.
            - save_dir (str): Đường dẫn tùy chỉnh để lưu file ảnh kết quả đầu ra.
            - stage_folder (str): Tên thư mục chứa kết quả train trước đó.
            - weights_folder (str): Tên thư mục con chứa trọng số. Mặc định là "weights".
            - filename_pt (str): Tên file trọng số cần nạp. Mặc định là "best.pt".
            - default_eval_folder (str): Tên thư mục đầu ra mặc định nếu không truyền save_dir.

        Kết quả trả về (Output):
            - tuple (model, test_result): Gồm đối tượng mô hình YOLO đã nạp trọng số 
            và đối tượng chứa toàn bộ ma trận số liệu kết quả tập test (mAP50, mAP50-95,...).
    """
    print(f"\n{'*'*20} NGHIỆM THU TẬP TEST: YOLO ({model_name.upper()}) {'*'*20}")

    # Tìm tệp trọng số linh hoạt dựa trên các tham số truyền vào
    model_path = os.path.join(workspace_dir, "runs", stage_folder, weights_folder, filename_pt)
    
    if not os.path.exists(model_path):
        model_path = os.path.join(workspace_dir, "models", f"FINETUNED_best_{model_name}.pt")
    
    if not os.path.exists(model_path):
        print(f"--- Cảnh báo: Không tìm thấy tệp trọng số tại {model_path}. Bỏ qua nghiệm thu.")
        return None, None

    print(f"Đã tìm thấy file trọng số tại: {model_path}")
    model = YOLO(model_path)

    # Xử lý đường dẫn lưu kết quả đánh giá dựa trên save_dir hoặc default_eval_folder
    project_path = os.path.join(workspace_dir, "runs")
    folder_name = default_eval_folder

    if save_dir is not None:
        project_path = os.path.dirname(save_dir)
        folder_name = os.path.basename(save_dir)

    print(f"Đang tiến hành quét và tính toán các chỉ số mAP trên tập dữ liệu test:")

    # Kích hoạt chế độ kiểm định trên tập test
    test_result = model.val(
        data=data_yaml_path,
        split="test",        
        device=device,
        conf=score_thresh,
        plots=True,          
        verbose=False,       
        project=project_path,
        name=folder_name,
        exist_ok=True        
    )

    # In bảng kết quả hiển thị trực quan ra màn hình
    print("\n-------------KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:---------------")
    print(f"  mAP@0.5          : {test_result.box.map50:.4f}")
    print(f"  mAP@0.5:0.95     : {test_result.box.map:.4f}")
    print(f"  Precision (Chung): {test_result.box.mp:.4f}")
    print(f"  Recall (Chung)   : {test_result.box.mr:.4f}")
    
    if hasattr(test_result, 'box') and test_result.box.maps is not None:
        print("\n  Chi tiết AP theo từng lớp (IoU=0.5):")
        class_names_dict = model.names  
        for class_id, _ in enumerate(test_result.box.maps):
            ap_50 = test_result.box.ap50[class_id]
            cls_name = class_names_dict.get(class_id, f"class_{class_id}")
            print(f"    [{cls_name:10s}] AP@0.5 = {ap_50:.4f}")

    print(f"-----> Toàn bộ file ảnh biểu đồ nghiệm thu đã được lưu tại:\n  --> {os.path.join(project_path, folder_name)}")
    return model, test_result


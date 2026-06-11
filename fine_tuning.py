import logging
import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data import get_dataloaders
from models import build_model
from training import EarlyStopping, train_one_epoch, validate_one_epoch
import time as _time
from ultralytics import YOLO


def get_parameters_for_discriminative_lr(model, base_lr):
    """
    Tạo các nhóm tham số để áp dụng chiến lược Discriminative Learning Rate
    trong quá trình fine-tuning mô hình học sâu.

    Hàm duyệt qua toàn bộ tham số của mô hình và tách chúng thành hai nhóm:
    - Nhóm backbone (các tầng trích xuất đặc trưng): sử dụng learning rate nhỏ hơn.
    - Nhóm head/classifier (các tầng phân loại cuối): sử dụng learning rate lớn hơn.

    Chiến lược này giúp bảo toàn các đặc trưng đã học từ mô hình pretrained,
    đồng thời cho phép phần phân loại thích nghi nhanh hơn với tập dữ liệu mới.

    Parameters:
        model : torch.nn.Module
            Mô hình cần tách nhóm tham số.
        base_lr : float
            Learning rate cơ sở áp dụng cho các tầng phân loại.

    Returns:
        list
            Danh sách các nhóm tham số theo định dạng yêu cầu của PyTorch Optimizer.
            Backbone sử dụng learning rate bằng base_lr / 10, trong khi head sử dụng
            learning rate bằng base_lr.
    """
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if "fc" in name or "classifier" in name or "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": base_lr / 10.0},
        {"params": head_params, "lr": base_lr},
    ]

def train_model(model_name, best_params, data_dir, workspace_dir, device):
    """
    Thực hiện quá trình fine-tuning cho mô hình phân loại ảnh.

    Đối với mô hình CustomCNN, hàm huấn luyện toàn bộ tham số từ đầu.
    Đối với các mô hình pretrained, hàm áp dụng quy trình huấn luyện
    hai giai đoạn gồm warm-up và fine-tuning toàn bộ mạng. Trong quá trình
    huấn luyện, hàm theo dõi loss, accuracy, điều chỉnh learning rate bằng
    scheduler, lưu mô hình tốt nhất bằng Early Stopping và ghi log phục vụ
    phân tích kết quả.

    Parameters:
        model_name : str
            Tên mô hình cần huấn luyện.
        best_params : dict
            Bộ siêu tham số tối ưu được sử dụng cho quá trình huấn luyện.
        data_dir : str
            Đường dẫn tới thư mục dữ liệu.
        workspace_dir : str
            Thư mục lưu mô hình, log và TensorBoard.
        device : torch.device
            Thiết bị thực thi mô hình.

    Returns:
        None
            Hàm thực hiện huấn luyện và lưu kết quả trực tiếp lên đĩa,
            không trả về giá trị.
    """
    banner_line = "#" * 30
    header_str = f"\n{banner_line}\n### PHASE: TRAINING {model_name.upper()}\n{banner_line}"
    print(header_str)
    logging.info(header_str)

    batch_size = best_params.get("batch_size", 128)
    opt_name = best_params["optimizer"]
    dataloaders, _, _, _ = get_dataloaders(data_dir, batch_size=batch_size)
    base_lr = best_params["lr"]

    if model_name == "custom":
        # Custom model train toàn bộ tham số từ đầu
        model = build_model(model_name, num_classes=2, is_feature_extraction=False, device=device)

        # Truyền trực tiếp model.parameters() để TẤT CẢ các layer nhận chung base_lr
        if opt_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=base_lr)
        elif opt_name == "AdamW":
            # Weight_decay để tăng khả năng chống học vẹt (overfitting)
            optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
        else:
            optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)

        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(log_dir=os.path.join(workspace_dir, "runs", f"{model_name}_finetune"))

        # CosineAnnealingLR giúp giảm LR mịn theo hình sin, tránh nảy khỏi sweet spot
        # T_max=100 tương ứng với số lượng epoch tối đa của vòng lặp
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        model_save_path = os.path.join(workspace_dir, "models", f"FINETUNED_best_{model_name}.pth")
        early_stopping = EarlyStopping(patience=15, path=model_save_path)

        for epoch in range(100):
            train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
            val_loss, val_acc = validate_one_epoch(model, dataloaders["val"], criterion, device)

            # Cập nhật tự động theo epoch, không nhận val_loss
            scheduler.step()

            writer.add_scalar("FT_Loss/train", train_loss, epoch)
            writer.add_scalar("FT_Loss/val", val_loss, epoch)

            msg = (
                f"FINE-TUNE {model_name} | Epoch {epoch}: "
                f"Loss(Train/Val): {train_loss:.4f}/{val_loss:.4f} | "
                f"Acc(Train/Val): {train_acc:.4f}/{val_acc:.4f}"
            )
            print(msg)
            logging.info(msg)

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                stop_msg = f"   Early stopping triggered at epoch {epoch}"
                print(stop_msg)
                logging.info(stop_msg)
                break

        writer.close()
        print(f"--- Hoan tat {model_name} ---")
        logging.info(f"--- Hoan tat {model_name} ---\n")
        return

    # --- Pretrained models: 2-phase fine-tuning ---
    model = build_model(model_name, num_classes=2, is_feature_extraction=True, device=device)

    if opt_name == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)
    elif opt_name == "AdamW":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=os.path.join(workspace_dir, "runs", f"{model_name}_finetune"))

    # Phase 1: Warm-up — chỉ train head, backbone đóng băng
    for epoch in range(3):
        train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, dataloaders["val"], criterion, device)
        msg = (
            f"FINE-TUNE {model_name} | Epoch {epoch} (Warm-up): "
            f"Loss(Train/Val): {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc(Train/Val): {train_acc:.4f}/{val_acc:.4f}"
        )
        print(msg)
        logging.info(msg)

    # Phase 2: Mở toàn bộ tham số, train với discriminative LR
    for param in model.parameters():
        param.requires_grad = True

    param_groups = get_parameters_for_discriminative_lr(model, base_lr)

    if opt_name == "Adam":
        optimizer = optim.Adam(param_groups)
    elif opt_name == "AdamW":
        optimizer = optim.AdamW(param_groups)
    else:
        optimizer = optim.SGD(param_groups, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    model_save_path = os.path.join(workspace_dir, "models", f"FINETUNED_best_{model_name}.pth")
    early_stopping = EarlyStopping(patience=6, path=model_save_path)

    for epoch in range(3, 15):
        train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, dataloaders["val"], criterion, device)
        scheduler.step(val_loss)

        writer.add_scalar("FT_Loss/train", train_loss, epoch)
        writer.add_scalar("FT_Loss/val", val_loss, epoch)

        msg = (
            f"FINE-TUNE {model_name} | Epoch {epoch}: "
            f"Loss(Train/Val): {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc(Train/Val): {train_acc:.4f}/{val_acc:.4f}"
        )
        print(msg)
        logging.info(msg)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            stop_msg = f"   Early stopping triggered at epoch {epoch}"
            print(stop_msg)
            logging.info(stop_msg)
            break

    writer.close()
    print(f"--- Hoan tat {model_name} ---")
    logging.info(f"--- Hoan tat {model_name} ---\n")

# ════════════════════════════════════════════════════════════════
# FASTER R-CNN — FINE TUNING MODULE
# ════════════════════════════════════════════════════════════════



def frcnn_fine_tune_model(best_params: dict, img_dir: str, label_dir: str,
                          workspace_dir: str, device, num_epoch,
                          num_classes: int = 3, score_thresh: float = 0.5,
                          logger=None):
    """
    Thực hiện fine-tuning mô hình Faster R-CNN cho bài toán phát hiện đối tượng.

    Quá trình huấn luyện được chia thành hai giai đoạn. Giai đoạn đầu chỉ
    huấn luyện phần head trong khi backbone được đóng băng để mô hình thích nghi
    với dữ liệu mới. Giai đoạn thứ hai mở khóa toàn bộ tham số, áp dụng
    Discriminative Learning Rate và đánh giá hiệu năng bằng các chỉ số mAP.
    Trong quá trình huấn luyện, hàm lưu lại lịch sử huấn luyện, ghi log và
    tự động lưu mô hình tốt nhất thông qua Early Stopping.

    Parameters:
        best_params : dict
            Bộ siêu tham số tối ưu sử dụng cho quá trình huấn luyện.
        img_dir : str
            Thư mục chứa ảnh của tập dữ liệu.
        label_dir : str
            Thư mục chứa các file nhãn tương ứng.
        workspace_dir : str
            Thư mục lưu mô hình và log huấn luyện.
        device : torch.device
            Thiết bị dùng để huấn luyện mô hình.
        total_epoch : int
            Tổng số epoch tối đa của toàn bộ quá trình fine-tuning.
        num_classes : int, optional
            Tổng số lớp của mô hình, bao gồm cả lớp background.
        score_thresh : float, optional
            Ngưỡng confidence dùng khi đánh giá dự đoán.
        logger : logging.Logger, optional
            Logger dùng để ghi lại thông tin huấn luyện.

    Returns:
        tuple
            Bao gồm lịch sử huấn luyện và giá trị mAP@0.5 tốt nhất đạt được
            trên tập validation.
    """

    import torch
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    from data import frcnn_get_dataloaders
    from models import frcnn_build_model, frcnn_set_parameter_requires_grad, frcnn_get_discriminative_params
    from training import FrcnnEarlyStopping, frcnn_train_one_epoch, frcnn_evaluate_map, frcnn_evaluate_loss

    EVAL_THRESH = 0.05   # dùng nội bộ để tính mAP khi train
    banner = "#" * 40
    msg = f"\n{banner}\n### FINE-TUNING FASTER R-CNN\n{banner}"
    print(msg)
    if logger:
        logger.info(msg)

    base_lr    = best_params["lr"]
    opt_name   = best_params.get("optimizer", "SGD")
    batch_size = best_params.get("batch_size", 4)

    dataloaders, dataset_sizes, _ = frcnn_get_dataloaders(
        img_dir=img_dir, label_dir=label_dir, batch_size=batch_size
    )

    msg = (
        f"  Hyperparams: lr={base_lr:.6f}  optimizer={opt_name}  batch_size={batch_size}\n"
        f"  Dataset sizes: {dataset_sizes}"
    )
    print(msg)
    if logger:
        logger.info(msg)

    # ════════════════════════════════════════
    # PHASE 1 — WARM-UP (freeze backbone)
    # ════════════════════════════════════════
    phase1_msg = "\n[Phase 1] Warm-up: chỉ train head (backbone frozen)..."
    print(phase1_msg)
    if logger:
        logger.info(phase1_msg)

    model = frcnn_build_model(num_classes=num_classes, is_feature_extraction=True, device=device)
    params_head = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "Adam":
        optimizer = optim.Adam(params_head, lr=base_lr)
    elif opt_name == "AdamW":
        optimizer = optim.AdamW(params_head, lr=base_lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(params_head, lr=base_lr, momentum=0.9, weight_decay=5e-4)

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, 9):
        train_loss, _ = frcnn_train_one_epoch(
            model, dataloaders["train"], optimizer, scaler,
            device, epoch, writer=None, logger=logger,
        )
        val_result = frcnn_evaluate_map(model, dataloaders["val"], device, EVAL_THRESH)
        map50      = val_result["map_50"].item()
        msg = f"  [Warm-up] Epoch {epoch}: loss={train_loss:.4f}  mAP@0.5={map50:.4f}"
        print(msg)
        if logger:
            logger.info(msg)
        warmup_save_path = os.path.join(workspace_dir, "models", "frcnn_warmup.pth")
        torch.save(model.state_dict(), warmup_save_path)
        print(f"  [Warm-up] Đã lưu checkpoint: {warmup_save_path}")

    # ════════════════════════════════════════
    # PHASE 2 — FULL FINE-TUNE (unfreeze all)
    # ════════════════════════════════════════
    phase2_msg = "\n[Phase 2] Full fine-tune: unfreeze toàn bộ + Discriminative LR..."
    print(phase2_msg)
    if logger:
        logger.info(phase2_msg)

    frcnn_set_parameter_requires_grad(model, freeze=False)
    param_groups = frcnn_get_discriminative_params(model, base_lr * 0.1)

    if opt_name == "Adam":
        optimizer = optim.Adam(param_groups)
    elif opt_name == "AdamW":
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

    model_save_path = os.path.join(workspace_dir, "models", "FINETUNED_best_frcnn.pth")
    os.makedirs(os.path.join(workspace_dir, "models"), exist_ok=True)
    early_stop = FrcnnEarlyStopping(patience=10, min_delta=0.001, path=model_save_path)

    writer = SummaryWriter(log_dir=os.path.join(workspace_dir, "runs", "frcnn_finetune"))

    history = {
        "train_loss": [], "val_loss":   [],
        "val_map50":  [], "val_map":    [],
        "val_ap_cho": [], "val_ap_meo": [],
    }

    for epoch in range(9, num_epoch + 1):
        t0 = _time.time()

        train_loss, _ = frcnn_train_one_epoch(
            model, dataloaders["train"], optimizer, scaler,
            device, epoch, writer=writer, logger=logger,
        )
        val_result = frcnn_evaluate_map(
            model, dataloaders["val"], device, EVAL_THRESH,
            epoch=epoch, writer=writer, prefix="val",
        )
        val_loss = frcnn_evaluate_loss(model, dataloaders["val"], device)

        map50    = val_result["map_50"].item()
        map_5095 = val_result["map"].item()
        ap_cho   = val_result["map_per_class"][0].item() if "map_per_class" in val_result else -1.0
        ap_meo   = val_result["map_per_class"][1].item() if "map_per_class" in val_result else -1.0

        scheduler.step(map50)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map50"].append(map50)
        history["val_map"].append(map_5095)
        history["val_ap_cho"].append(ap_cho)
        history["val_ap_meo"].append(ap_meo)

        elapsed = _time.time() - t0
        msg = (
            f"  [FT] Epoch {epoch}: train loss={train_loss:.4f}  val loss={val_loss:.4f}  "
            f"mAP@0.5={map50:.4f}  mAP@0.5:0.95={map_5095:.4f}  "
            f"AP_cho={ap_cho:.4f}  AP_meo={ap_meo:.4f}  time={elapsed:.1f}s"
        )
        print(msg)
        if logger:
            logger.info(msg)

        saved = early_stop(map50, model)
        if saved:
            msg = f"  --- Lưu model tốt nhất: mAP@0.5 = {early_stop.best_map:.4f} ---"
        else:
            msg = f"  [Early Stopping] Chưa tiến bộ: {early_stop.counter}/{early_stop.patience}"
        print(msg)
        if logger:
            logger.info(msg)

        if early_stop.early_stop:
            msg = f"\n[Early Stopping] Dừng tại Epoch {epoch}. Best = {early_stop.best_map:.4f}"
            print(msg)
            if logger:
                logger.info(msg)
            break

    writer.close()
    msg = f"\n--- Fine-tuning hoàn tất. Best mAP@0.5 = {early_stop.best_map:.4f} ---"
    print(msg)
    if logger:
        logger.info(msg)

    return history, early_stop.best_map


# YOLO ---------------------------------------------------------------------------------------

def yolo_finetuning(data_dir, workspace_dir, device=0, 
                    optuna_study=None, manual_params=None,
                    warm_up_epochs = 3, fine_tune_epochs = 12):

    """
    Thực hiện fine-tuning mô hình YOLOv8 theo chiến lược huấn luyện hai giai đoạn.

    Hàm hỗ trợ sử dụng bộ siêu tham số từ Optuna hoặc cấu hình thủ công.
    Ở giai đoạn đầu, backbone được đóng băng để huấn luyện phần head nhằm
    giúp mô hình thích nghi với tập dữ liệu mới. Ở giai đoạn thứ hai, toàn bộ
    mạng được mở khóa và tiếp tục huấn luyện với learning rate nhỏ hơn để
    tinh chỉnh các đặc trưng đã học. Trong quá trình huấn luyện, mô hình được
    lưu tự động và sử dụng nhiều kỹ thuật tăng cường dữ liệu nhằm cải thiện
    khả năng tổng quát hóa.

    Parameters:
        data_dir : str
            Thư mục chứa dữ liệu và file cấu hình data.yaml.
        workspace_dir : str
            Thư mục lưu kết quả huấn luyện.
        device : int | str, optional
            Thiết bị sử dụng để huấn luyện mô hình.
        optuna_study : optuna.study.Study, optional
            Đối tượng Optuna chứa bộ siêu tham số tối ưu.
        manual_params : dict, optional
            Bộ siêu tham số được cấu hình thủ công.
        warm_up_epochs : int, optional
            Số epoch sử dụng cho giai đoạn warm-up.
        fine_tune_epochs : int, optional
            Số epoch sử dụng cho giai đoạn fine-tuning toàn bộ mô hình.

    Returns:
        str
            Đường dẫn tới file trọng số tốt nhất của mô hình sau khi hoàn tất
            quá trình huấn luyện.
    """
    project_dir = os.path.join(workspace_dir, "runs")
    logging.info(f"\n{'-'*10} KHỔI ĐỘNG HÀM FINE-TUNING YOLOv8s HAI GIAI ĐOẠN{'-'*10}")

    # Khởi tạo mặc định siêu tham số ban đầu
    base_lr = 0.0025 
    opt_name = "AdamW"

    if optuna_study is not None:
        try:
            base_lr = optuna_study.best_params['lr']
            opt_name = optuna_study.best_params['optimizer']
            logging.info(f"Đã truyền bộ siêu tham số tối ưu từ Optuna Study.")
        except (AttributeError, KeyError):
            logging.warning(f"Đối tượng Optuna study không hợp lệ hoặc thiếu tham số. Dùng cấu hình mặc định.")
    elif manual_params is not None:
        base_lr = manual_params.get("lr", base_lr)
        opt_name = manual_params.get("optimizer", opt_name)
        logging.info(f"Đã nhận bộ tham số cấu hình thủ công (Manual).")
    else:
        logging.info(f"Không có tham số đầu vào. Sử dụng cấu hình mặc định hệ thống.")

    logging.info(f"Tốc độ học cơ sở (base_lr): {base_lr:.6f}")
    logging.info(f"Thuật toán tối ưu (optimizer): {opt_name}\n")

    # GIAI ĐOẠN 1: WARM-UP head (đóng băng BACKBONE - 10 layers đâu): thực hiện {warm_up_epochs} epochs
    logging.info(f"{'-'*10} STAGE 1: Chạy {warm_up_epochs} Epoch đầu | Đóng băng hoàn toàn Backbone (freeze=10) {'-'*10}")
    logging.info("YOLOv8 Stage 1 (Warm-up Head) started")

    model_stage1 = YOLO("yolov8s.pt")

    model_stage1.train(
        data=os.path.join(data_dir, 'data.yaml'),
        epochs=warm_up_epochs,                        
        batch=16,                        
        imgsz=640,  # Chuẩn hóa tất cả các ảnh về cùng một size trong 3 ba quá trình train, val, test
        device=device,
        optimizer=opt_name,
        lr0=base_lr,

        freeze=10,                       # Khóa từ Layer 0 đến Layer 9 (Backbone)
        fraction=1.0,                    # Chạy với bộ dữ liệu hoàn chỉnh

        # Thực hiện tăng cường dữ liệu (Data Augmentation):
        degrees=15.0,
        flipud=0.0,
        fliplr=0.5,
        scale=0.5,
        mosaic=1.0,

        project=project_dir,
        name="yolov8_stage1_warmup",
        save=True,
        plots=True,
        verbose=True,
        exist_ok=True
    )

    # Xác định file trọng số tốt nhất sau Stage 1
    stage1_best_weights = os.path.join(project_dir, "yolov8_stage1_warmup", "weights", "best.pt")

    if not os.path.exists(stage1_best_weights):
        raise FileNotFoundError(f"Không tìm thấy file trọng số Stage 1 tại {stage1_best_weights}. Vui lòng kiểm tra lại quá trình train.")

    # GIAI ĐOẠN 2: FINE-TUNE toàn bộ mạng neural network - thực hiện với {fine_tune_epochs} epochs
    logging.info(f"\n{'-'*10}STAGE 2: Chạy {fine_tune_epochs} Epoch tiếp theo | Mở khóa toàn mạng (freeze=0){'-'*10}")
    logging.info("YOLOv8 Stage 2 (Full Fine-tuning) started")

    # Tải lại mô hình kế thừa từ trọng số tốt nhất vừa đạt được ở Stage 1
    model_stage2 = YOLO(stage1_best_weights)

    # Cập nhật lại learning rate cho quá trình fine-tuning:
    finetune_lr = base_lr / 10.0
    logging.info(f"Tốc độ học tinh chỉnh mới cho Giai đoạn 2: {finetune_lr:.6f}")

    model_stage2.train(
        data=os.path.join(data_dir, 'data.yaml'),
        epochs=fine_tune_epochs,                      
        batch=16,
        imgsz=640,
        device=device,
        optimizer=opt_name,
        lr0=finetune_lr,                 

        # Chiến lược mở khóa:
        freeze=0,                        # Giải phóng hoàn toàn mạng, cho tinh chỉnh toàn bộ 130 tầng
        fraction=1.0,                    # Chạy trên 100% dữ liệu chính thức
        patience=5,                      # Tự động Early Stopping nếu Val Loss không giảm sau 5 epoch

        # Giữ nguyên cấu hình Augmentation đồng nhất
        degrees=15.0,
        flipud=0.0,
        fliplr=0.5,
        scale=0.5,
        mosaic=1.0,

        project=project_dir,
        name="yolov8_stage2_finetuned",
        save=True,
        plots=True,
        verbose=True,
        exist_ok=True
    )

    final_weights_path = os.path.join(project_dir, 'yolov8_stage2_finetuned', 'weights', 'best.pt')
    logging.info(f"\n{'-'*10}Hoàn tất quá trình fine-tuning!!!")
    logging.info(f"Trọng số tối ưu cuối cùng lưu tại: {final_weights_path}\n{'-'*10}")
    logging.info("YOLOv8 functional two-stage training workflow finished.")

    return final_weights_path
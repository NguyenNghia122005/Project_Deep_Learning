import logging
import os
import random

import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from sklearn import svm
from sklearn.metrics import accuracy_score
from data import get_dataloaders
from models import build_model
from training import EarlyStopping, train_one_epoch, validate_one_epoch
from ultralytics import YOLO
import sys
from ultralytics.utils import LOGGER as YOLO_LOGGER 


def objective(trial, model_type, X_train=None, y_train=None, X_val=None, y_val=None,
              model_name=None, data_dir=None, workspace_dir=None, device=None):
    
    """
    Hàm mục tiêu của Optuna dùng để tìm kiếm siêu tham số tối ưu.

    Hàm hỗ trợ hai loại mô hình khác nhau gồm SVM và Deep Learning.
    Đối với SVM, hàm thực hiện huấn luyện trên tập train và đánh giá
    độ chính xác trên tập validation. Đối với các mô hình Deep Learning,
    hàm thực hiện huấn luyện trên một tập con của dữ liệu nhằm giảm thời
    gian tìm kiếm siêu tham số, đồng thời sử dụng Early Stopping và
    Optuna Pruning để loại bỏ các cấu hình kém hiệu quả.

    Parameters:
        trial : optuna.Trial
            Đối tượng Trial do Optuna cung cấp.
        model_type : str
            Loại mô hình cần tối ưu, gồm "svm" hoặc "deep".
        X_train : np.ndarray, optional
            Dữ liệu huấn luyện cho SVM.
        y_train : np.ndarray, optional
            Nhãn huấn luyện cho SVM.
        X_val : np.ndarray, optional
            Dữ liệu validation cho SVM.
        y_val : np.ndarray, optional
            Nhãn validation cho SVM.
        model_name : str, optional
            Tên mô hình Deep Learning cần tối ưu.
        data_dir : str, optional
            Thư mục chứa dữ liệu huấn luyện.
        workspace_dir : str, optional
            Thư mục lưu kết quả và mô hình.
        device : torch.device, optional
            Thiết bị thực thi mô hình.

    Returns:
        float
            Giá trị mục tiêu dùng để tối ưu hóa. Đối với SVM là Accuracy,
            đối với Deep Learning là Validation Loss tốt nhất.
    """

    # ------------------------------------------------------------------ #
    #  NHÁNH SVM                                                           #
    # ------------------------------------------------------------------ #
    if model_type == "svm":
        c_val      = trial.suggest_float("C", 1e-3, 10.0, log=True)
        kernel_val = trial.suggest_categorical("kernel", ["linear", "rbf"])

        model = svm.SVC(C=c_val, kernel=kernel_val)
        model.fit(X_train, y_train)

        preds    = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        return accuracy   # Optuna maximize accuracy cho SVM

    # ------------------------------------------------------------------ #
    #  NHÁNH DEEP LEARNING                                                 #
    # ------------------------------------------------------------------ #
    elif model_type == "deep":
        logging.info(f"\n{'=' * 20} START TRIAL {trial.number}: {model_name.upper()} {'=' * 20}")

        lr             = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size     = 128
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])

        # --- Tạo subset 20% dữ liệu train để tìm kiếm nhanh hơn ---
        dataloaders, _, _, _ = get_dataloaders(data_dir, batch_size=batch_size)
        train_dataset = dataloaders["train"].dataset
        indices       = list(range(len(train_dataset)))
        random.shuffle(indices)
        subset_indices = indices[: int(0.20 * len(train_dataset))]

        optuna_train_loader = DataLoader(
            Subset(train_dataset, subset_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        # --- Khởi tạo model, optimizer, criterion ---
        is_fe  = model_name != "custom"
        model  = build_model(model_name, num_classes=2, is_feature_extraction=is_fe, device=device)
        params = [p for p in model.parameters() if p.requires_grad]

        if optimizer_name == "Adam":
            optimizer = optim.Adam(params, lr=lr)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(params, lr=lr)
        else:
            optimizer = optim.SGD(params, lr=lr, momentum=0.9)

        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

        # --- Early stopping & logging ---
        model_save_path = os.path.join(
            workspace_dir, "models", f"best_{model_name}_trial_{trial.number}.pth"
        )
        early_stopping = EarlyStopping(patience=4, path=model_save_path)
        writer = SummaryWriter(
            log_dir=os.path.join(workspace_dir, "runs", model_name, f"trial_{trial.number}")
        )

        # --- Training loop ---
        for epoch in range(20):
            train_loss, train_acc = train_one_epoch(model, optuna_train_loader, criterion, optimizer, device)
            val_loss,   val_acc   = validate_one_epoch(model, dataloaders["val"], criterion, device)
            scheduler.step(val_loss)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val",   val_loss,   epoch)
            writer.add_scalar("Acc/train",  train_acc,  epoch)
            writer.add_scalar("Acc/val",    val_acc,    epoch)

            logging.info(
                f"[Trial {trial.number}] Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            trial.report(val_loss, epoch)
            if trial.should_prune():
                logging.info(f"Trial {trial.number} pruned at epoch {epoch}")
                writer.close()
                raise optuna.TrialPruned()

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break

        logging.info(f"Trial {trial.number} finished. Best Val Loss: {early_stopping.best_loss:.4f}")
        writer.close()
        return early_stopping.best_loss   # Optuna minimize val_loss cho deep model

    else:
        raise ValueError(f"model_type không hợp lệ: '{model_type}'. Chọn 'svm' hoặc 'deep'.")


# ════════════════════════════════════════════════════════════════
# FASTER R-CNN — TUNING MODULE
# ════════════════════════════════════════════════════════════════



def frcnn_objective(trial, img_dir: str, label_dir: str,
                    workspace_dir: str, device, logger=None) -> float:
    """
    Hàm mục tiêu của Optuna cho mô hình Faster R-CNN.

    Hàm tạo tập dữ liệu huấn luyện và validation, xây dựng mô hình
    Faster R-CNN, lựa chọn optimizer theo tham số được đề xuất bởi
    Optuna và thực hiện huấn luyện trong một số epoch ngắn. Hiệu năng
    của mô hình được đánh giá bằng chỉ số mAP@0.5 trên tập validation,
    sau đó được sử dụng làm mục tiêu tối ưu hóa.

    Parameters:
        trial : optuna.Trial
            Đối tượng Trial do Optuna cung cấp.
        img_dir : str
            Thư mục chứa ảnh của dataset.
        label_dir : str
            Thư mục chứa nhãn tương ứng.
        workspace_dir : str
            Thư mục lưu kết quả huấn luyện.
        device : torch.device
            Thiết bị thực thi mô hình.
        logger : logging.Logger, optional
            Logger dùng để ghi lại quá trình tuning.

    Returns:
        float
            Giá trị mAP@0.5 tốt nhất đạt được trong quá trình huấn luyện.
    """
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset

    from data import FrcnnDataset, frcnn_collate_fn
    from models import frcnn_build_model
    from training import frcnn_train_one_epoch, frcnn_evaluate_map

    lr             = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    batch_size     = trial.suggest_categorical("batch_size", [2, 4])

    log_msg = (
        f"\n{'='*20} TRIAL {trial.number} {'='*20}\n"
        f"  lr={lr:.6f}  optimizer={optimizer_name}  batch_size={batch_size}"
    )
    print(log_msg)
    if logger:
        logger.info(log_msg)

    train_full = FrcnnDataset(
        img_dir=os.path.join(img_dir, "train"),
        label_dir=os.path.join(label_dir, "train"),
    )
    val_dataset = FrcnnDataset(
        img_dir=os.path.join(img_dir, "val"),
        label_dir=os.path.join(label_dir, "val"),
    )

    n_subset     = int(0.20 * len(train_full))
    indices      = random.sample(range(len(train_full)), n_subset)
    train_subset = Subset(train_full, indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=frcnn_collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=frcnn_collate_fn,
    )

    model = frcnn_build_model(num_classes=3, is_feature_extraction=True, device=device)
    params_to_update = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "SGD":
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(params_to_update, lr=lr)
    else:
        optimizer = optim.AdamW(params_to_update, lr=lr, weight_decay=1e-4)

    scaler     = torch.amp.GradScaler("cuda")
    best_map50 = 0.0

    for epoch in range(1, 21):
        frcnn_train_one_epoch(
            model, train_loader, optimizer, scaler,
            device, epoch, writer=None, logger=logger,
        )
        val_result = frcnn_evaluate_map(model, val_loader, device, score_thresh=0.5)
        map50      = val_result["map_50"].item()
        best_map50 = max(best_map50, map50)

        log_msg = (
            f"  [Trial {trial.number}] Epoch {epoch}: "
            f"mAP@0.5={map50:.4f}"
        )
        print(log_msg)
        if logger:
            logger.info(log_msg)

        trial.report(map50, epoch)
        if trial.should_prune():
            log_msg = f"  Trial {trial.number} bị pruned tại epoch {epoch}."
            print(log_msg)
            if logger:
                logger.info(log_msg)
            raise optuna.TrialPruned()

    log_msg = f"  Trial {trial.number} hoàn tất. Best mAP@0.5 = {best_map50:.4f}"
    print(log_msg)
    if logger:
        logger.info(log_msg)

    return best_map50


def frcnn_run_optuna_study(img_dir: str, label_dir: str, workspace_dir: str,
                           device, n_trials: int = 10, logger=None) -> dict:
    """
    Thực hiện quá trình tìm kiếm siêu tham số cho Faster R-CNN bằng Optuna.

    Hàm khởi tạo Study, cấu hình cơ chế Pruning nhằm loại bỏ sớm các
    trial có kết quả kém và thực hiện tối ưu hóa thông qua hàm
    frcnn_objective. Sau khi hoàn tất, hàm trả về bộ siêu tham số
    tốt nhất tìm được.

    Parameters:
        img_dir : str
            Thư mục chứa ảnh của dataset.
        label_dir : str
            Thư mục chứa nhãn tương ứng.
        workspace_dir : str
            Thư mục lưu cơ sở dữ liệu Optuna và các kết quả huấn luyện.
        device : torch.device
            Thiết bị thực thi mô hình.
        n_trials : int, optional
            Số lượng trial tối đa cần thực hiện.
        logger : logging.Logger, optional
            Logger dùng để ghi lại quá trình tối ưu.

    Returns:
        dict
            Bộ siêu tham số tối ưu được tìm thấy bởi Optuna.
    """
    os.makedirs(os.path.join(workspace_dir, "models"), exist_ok=True)

    storage_path = f"sqlite:///{os.path.join(workspace_dir, 'optuna_frcnn.db')}"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)

    study = optuna.create_study(
        study_name     = "frcnn_tuning",
        storage        = storage_path,
        direction      = "maximize",   # maximize mAP@0.5 (khác classification: minimize)
        pruner         = pruner,
        load_if_exists = True,
    )

    study.optimize(
        lambda trial: frcnn_objective(
            trial, img_dir, label_dir, workspace_dir, device, logger
        ),
        n_trials = n_trials,
        timeout  = None,
    )

    best = study.best_trial
    msg = (
        f"\n{'#'*40}\n"
        f"Optuna hoàn tất! Best Trial #{best.number}\n"
        f"  Best mAP@0.5 = {best.value:.4f}\n"
        f"  Best params  = {best.params}\n"
        f"{'#'*40}"
    )
    print(msg)
    if logger:
        logger.info(msg)

    return best.params


# ════════════════════════════════════════════════════════════════
# YOLO
# ================================================================

def yolo_objective(trial, data_dir, device, n_epochs = 3, fraction = 0.20):
    """
    Hàm mục tiêu của Optuna dành cho mô hình YOLOv8.

    Hàm thực hiện huấn luyện YOLOv8 trên một phần dữ liệu trong số epoch
    ngắn nhằm đánh giá nhanh hiệu quả của các bộ siêu tham số khác nhau.
    Trong quá trình huấn luyện, callback được sử dụng để báo cáo chỉ số
    mAP@0.5 cho Optuna và kích hoạt cơ chế Pruning nếu trial hiện tại có
    kết quả thấp hơn đáng kể so với các trial trước đó.

    Parameters:
        trial : optuna.Trial
            Đối tượng Trial do Optuna cung cấp.
        data_dir : str
            Thư mục chứa dữ liệu và file cấu hình data.yaml.
        device : int | str
            Thiết bị sử dụng để huấn luyện mô hình.
        n_epochs : int, optional
            Số epoch sử dụng cho mỗi trial.
        fraction : float, optional
            Tỷ lệ dữ liệu huấn luyện được sử dụng trong quá trình tuning.

    Returns:
        float
            Giá trị mAP@0.5 đạt được sau khi hoàn tất trial.
    """
    logging.info(f"{'-' * 10} START TRIAL {trial.number}: YOLOv8s {'-' * 10}")

    # Khởi tạo không gian tham số:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    batch_size = 16

    # Khởi tạo mô hình pre-trained cơ bản:
    model = YOLO("yolov8s.pt")

    # Hàm Callback: Chỉ lấy mAP50 báo cáo cho Optuna để thực hiện cắt tỉa (Pruning)
    def optuna_pruning_callback(trainer):
        val_map50 = 0.0
        if hasattr(trainer, 'validator') and hasattr(trainer.validator, 'metrics'):
            val_map50 = float(trainer.validator.metrics.box.map50)

        # Báo cáo tiến độ về cho Optuna
        trial.report(val_map50, step=trainer.epoch)
        
        # Nếu trial quá tệ so với các trial trước, chủ động dừng sớm (Prune) để tiết kiệm thời gian
        if trial.should_prune():
            logging.info(f"---[Trial {trial.number}] kết thúc tại epoch {trainer.epoch + 1} do kết quả thấp.")
            raise optuna.TrialPruned()

    # Đăng ký callback báo cáo cho Optuna
    model.add_callback("on_fit_epoch_end", optuna_pruning_callback)

    # Kích hoạt vòng lặp huấn luyện của YOLOv8
    try:
        results = model.train(
            data=os.path.join(data_dir, 'data.yaml'),
            epochs=n_epochs,                         # Chạy ngắn 3 epoch để check xu hướng
            batch=batch_size,
            imgsz=640,
            lr0=lr,
            optimizer=optimizer_name,
            freeze=10,                               # Đóng băng 10 tầng Backbone
            fraction=fraction,                       # Lấy một phần dữ liệu train để tuning
            device=device,
            
            # BẬT LẠI LOG MẶC ĐỊNH CỦA YOLO ĐỂ XEM THỜI GIAN VÀ CÁC CHỈ SỐ KHÁC
            verbose=True,                            # Hiện log chi tiết từng epoch của YOLO (có kèm thời gian)
            plots=True,                              
            save=True                               
        )

        # Trả về chỉ số chính xác cao nhất của Trial làm mục tiêu tối ưu cho Optuna
        return results.box.map50

    except optuna.TrialPruned:
        return 0.0



def yolo_setup_custom_logging(workspace_dir: str, log_filename: str = "log_yolo_optuna_process.txt"):
    """
    Công dụng:
        Cấu hình hệ thống log đồng bộ cho cả Ultralytics YOLO và Optuna Hyperparameter Tuning.
        Tự động bắt toàn bộ gói tin log, in ra màn hình và ghi vào file txt được chỉ định.

    Tham số truyền vào:
        - workspace_dir (str): Đường dẫn đến thư mục làm việc chính.
        - log_filename (str): Tên file log muốn lưu (ví dụ: 'optuna_tuning_run1.txt').

    Kết quả trả về (Output):
        - file_handler (logging.FileHandler): Đối tượng handler quản lý file log để dùng lại nếu cần.
    """
    log_file_path = os.path.join(workspace_dir, log_filename)

    # Tự động kiểm tra và tạo thư mục chứa log nếu chưa tồn tại
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Tạo Handler ghi file
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    file_handler.setLevel(logging.INFO)

    # Xóa sạch các FileHandler cũ bám trên hệ thống để tránh ghi lặp log
    for logger in [logging.root, YOLO_LOGGER, optuna.logging.get_logger("optuna")]:
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

    # Cấu hình root logger mặc định của Python
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, logging.StreamHandler(sys.stdout)]
    )

    # Ép hệ thống Core của YOLOv8 ghi log vào file này
    YOLO_LOGGER.addHandler(file_handler)

    # Ép hệ thống Core của Optuna ghi log vào file này
    optuna_logger = optuna.logging.get_logger("optuna")
    optuna_logger.addHandler(file_handler)
    # Bật chế độ hiển thị log mặc định của Optuna (INFO) để nó in ra thông tin từng trial
    optuna.logging.set_verbosity(optuna.logging.INFO)

    logging.info(f"Set up logging thành công. File log lưu tại: {log_file_path}")
    return file_handler
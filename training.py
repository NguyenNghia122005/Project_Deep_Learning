import torch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, path="best_model.pth"):
        """
        Khởi tạo cơ chế Early Stopping cho bài toán phân loại ảnh.

        Đối tượng này theo dõi giá trị validation loss trong quá trình huấn luyện.
        Khi validation loss không còn cải thiện sau một số epoch liên tiếp, quá trình
        huấn luyện sẽ được dừng sớm nhằm tránh hiện tượng overfitting và tiết kiệm
        thời gian huấn luyện.

        Parameters:
            patience : int, optional
                Số epoch tối đa được phép không cải thiện trước khi dừng huấn luyện.
            min_delta : float, optional
                Mức cải thiện tối thiểu của validation loss để được xem là tiến bộ.
            path : str, optional
                Đường dẫn lưu mô hình tốt nhất.

        Returns:
            None
                Hàm khởi tạo đối tượng EarlyStopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        """
        Cập nhật trạng thái Early Stopping sau mỗi epoch.

        Hàm so sánh validation loss hiện tại với giá trị tốt nhất đã ghi nhận.
        Nếu mô hình cải thiện, trọng số sẽ được lưu lại và bộ đếm được đặt về 0.
        Ngược lại, bộ đếm sẽ tăng lên và có thể kích hoạt dừng sớm nếu vượt quá
        ngưỡng patience.

        Parameters:
            val_loss : float
                Giá trị validation loss của epoch hiện tại.
            model : torch.nn.Module
                Mô hình cần lưu khi đạt kết quả tốt nhất.

        Returns:
            None
                Hàm cập nhật trạng thái EarlyStopping và lưu mô hình khi cần.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def _create_scaler(device):
    """
    Khởi tạo GradScaler phục vụ huấn luyện hỗn hợp độ chính xác (Mixed Precision).

    Hàm tự động kích hoạt hoặc vô hiệu hóa GradScaler dựa trên thiết bị đang sử
    dụng. Khi chạy trên GPU, GradScaler giúp tăng tốc huấn luyện và giảm lượng
    bộ nhớ tiêu thụ.

    Parameters:
        device : torch.device
            Thiết bị thực thi mô hình.

    Returns:
        torch.amp.GradScaler
            Đối tượng GradScaler được cấu hình phù hợp với thiết bị.
    """
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """
    Huấn luyện mô hình trong một epoch.

    Hàm thực hiện một vòng lặp huấn luyện hoàn chỉnh trên toàn bộ tập train.
    Trong quá trình huấn luyện, dữ liệu được chuyển tới thiết bị tính toán,
    thực hiện lan truyền xuôi, tính loss, lan truyền ngược và cập nhật trọng số.
    Hàm cũng hỗ trợ Mixed Precision Training thông qua GradScaler.

    Parameters:
        model : torch.nn.Module
            Mô hình cần huấn luyện.
        dataloader : DataLoader
            DataLoader của tập huấn luyện.
        criterion : torch.nn.Module
            Hàm mất mát sử dụng trong quá trình huấn luyện.
        optimizer : torch.optim.Optimizer
            Optimizer dùng để cập nhật trọng số.
        device : torch.device
            Thiết bị thực thi mô hình.
        scaler : torch.amp.GradScaler, optional
            Đối tượng hỗ trợ Mixed Precision Training.

    Returns:
        tuple
            Bao gồm train loss trung bình và train accuracy của epoch.
    """
    model.train()
    running_loss, running_corrects = 0.0, 0

    if scaler is None:
        scaler = _create_scaler(device)

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Đánh giá mô hình trên tập validation trong một epoch.

    Hàm chuyển mô hình sang chế độ đánh giá, thực hiện suy luận trên toàn bộ
    tập validation và tính toán loss cùng accuracy mà không cập nhật trọng số.

    Parameters:
        model : torch.nn.Module
            Mô hình cần đánh giá.
        dataloader : DataLoader
            DataLoader của tập validation.
        criterion : torch.nn.Module
            Hàm mất mát sử dụng để đánh giá.
        device : torch.device
            Thiết bị thực thi mô hình.

    Returns:
        tuple
            Bao gồm validation loss trung bình và validation accuracy.
    """
    model.eval()
    running_loss, running_corrects = 0.0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()


# ════════════════════════════════════════════════════════════════
# FASTER R-CNN — TRAINING MODULE
# ════════════════════════════════════════════════════════════════


import logging
import time

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# ─────────────────────────────────────────────
#  Early Stopping (theo mAP@0.5 — càng cao càng tốt)
# ─────────────────────────────────────────────

class FrcnnEarlyStopping:

    def __init__(self, patience: int = 3, min_delta: float = 0.001, path: str = "best_frcnn.pth"):
        """
        Khởi tạo cơ chế Early Stopping cho mô hình Faster R-CNN.

        Khác với bài toán phân loại sử dụng validation loss, Faster R-CNN sử dụng
        chỉ số mAP@0.5 để đánh giá hiệu năng. Quá trình huấn luyện sẽ được dừng
        nếu mAP không cải thiện sau một số epoch liên tiếp.

        Parameters:
            patience : int, optional
                Số epoch tối đa được phép không cải thiện.
            min_delta : float, optional
                Mức cải thiện tối thiểu của mAP để được xem là tiến bộ.
            path : str, optional
                Đường dẫn lưu trọng số tốt nhất.

        Returns:
            None
                Hàm khởi tạo đối tượng FrcnnEarlyStopping.
        """
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.counter    = 0
        self.best_map   = 0.0
        self.early_stop = False

    def __call__(self, map50: float, model):
        """
        Cập nhật trạng thái Early Stopping dựa trên mAP@0.5.

        Nếu mAP@0.5 của epoch hiện tại cao hơn giá trị tốt nhất trước đó,
        trọng số mô hình sẽ được lưu lại và bộ đếm được đặt lại. Nếu không có
        sự cải thiện đủ lớn, bộ đếm sẽ tăng lên và có thể kích hoạt dừng sớm.

        Parameters:
            map50 : float
                Giá trị mAP@0.5 hiện tại.
            model : torchvision.models.detection.FasterRCNN
                Mô hình cần lưu khi đạt kết quả tốt nhất.

        Returns:
            bool
                True nếu mô hình mới được lưu, False nếu không có cải thiện.
        """
        import torch
        if map50 > self.best_map + self.min_delta:
            self.best_map = map50
            self.counter  = 0
            torch.save(model.state_dict(), self.path)
            return True   # đã lưu best
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # không lưu


# ─────────────────────────────────────────────
#  Train 1 epoch — FRCNN
# ─────────────────────────────────────────────

def frcnn_train_one_epoch(model, loader, optimizer, scaler, device, epoch: int,
                          writer: SummaryWriter = None, logger: logging.Logger = None):
    """
    Huấn luyện mô hình Faster R-CNN trong một epoch.

    Hàm thực hiện lan truyền xuôi, tính toán các thành phần loss của Faster R-CNN,
    lan truyền ngược và cập nhật trọng số. Các giá trị loss được ghi nhận để
    phục vụ theo dõi quá trình huấn luyện và hiển thị trên TensorBoard.

    Parameters:
        model : torchvision.models.detection.FasterRCNN
            Mô hình Faster R-CNN cần huấn luyện.
        loader : DataLoader
            DataLoader của tập huấn luyện.
        optimizer : torch.optim.Optimizer
            Optimizer dùng để cập nhật trọng số.
        scaler : torch.amp.GradScaler
            Đối tượng hỗ trợ Mixed Precision Training.
        device : torch.device
            Thiết bị thực thi mô hình.
        epoch : int
            Epoch hiện tại.
        writer : SummaryWriter, optional
            Đối tượng TensorBoard dùng để ghi log.
        logger : logging.Logger, optional
            Logger dùng để ghi lại quá trình huấn luyện.

    Returns:
        tuple
            Bao gồm tổng loss trung bình và các thành phần loss chi tiết.
    """

    import torch

    model.train()
    total_loss    = 0.0
    loss_dict_sum = {}
    n_actual      = 0
    n_total       = len(loader)

    data_iter = iter(loader)

    while True:
        try:
            images, targets = next(data_iter)
        except StopIteration:
            break
        except Exception:
            msg = "[CẢNH BÁO] Phát hiện batch lỗi trong Train. Bỏ qua..."
            print(msg)
            if logger:
                logger.warning(msg)
            continue

        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())

        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + v.item()
        n_actual += 1

        if n_actual % max(1, n_total // 5) == 0:
            detail = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
            msg = f"  Epoch {epoch} [{n_actual}/{n_total}]  loss={losses.item():.4f}  {detail}"
            print(msg)
            if logger:
                logger.info(msg)

    if n_actual == 0:
        return 0.0, {}

    avg_loss    = total_loss / n_actual
    avg_details = {k: v / n_actual for k, v in loss_dict_sum.items()}

    if writer:
        writer.add_scalar("Loss/train_total", avg_loss, epoch)
        for k, v in avg_details.items():
            writer.add_scalar(f"Loss_detail/{k}", v, epoch)

    return avg_loss, avg_details


# ─────────────────────────────────────────────
#  Evaluate mAP
# ─────────────────────────────────────────────

@torch.no_grad()
def frcnn_evaluate_map(model, loader, device, score_thresh: float = 0.5,
                 epoch: int = None, writer: SummaryWriter = None,
                 prefix: str = "val"):
    """
    Đánh giá mô hình Faster R-CNN bằng chỉ số Mean Average Precision (mAP).

    Hàm thực hiện suy luận trên tập dữ liệu đánh giá, lọc các dự đoán theo
    ngưỡng confidence và sử dụng MeanAveragePrecision của TorchMetrics để
    tính toán các chỉ số mAP phục vụ đánh giá hiệu năng phát hiện đối tượng.

    Parameters:
        model : torchvision.models.detection.FasterRCNN
            Mô hình cần đánh giá.
        loader : DataLoader
            DataLoader của tập đánh giá.
        device : torch.device
            Thiết bị thực thi mô hình.
        score_thresh : float, optional
            Ngưỡng confidence để giữ lại dự đoán.
        epoch : int, optional
            Epoch hiện tại.
        writer : SummaryWriter, optional
            Đối tượng TensorBoard dùng để ghi log.
        prefix : str, optional
            Tiền tố dùng khi lưu metric vào TensorBoard.

    Returns:
        dict
            Từ điển chứa các chỉ số mAP được tính toán bởi TorchMetrics.
    """

    import torch

    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    ).to(device)

    data_iter = iter(loader)

    while True:
        try:
            images, targets = next(data_iter)
        except StopIteration:
            break
        except Exception:
            print("[CẢNH BÁO] Batch lỗi trong Evaluate. Bỏ qua...")
            continue

        images = [img.to(device) for img in images]

        with torch.amp.autocast("cuda"):
            outputs = model(images)

        preds, gts = [], []
        for out, tgt in zip(outputs, targets):
            mask = out["scores"] >= score_thresh
            preds.append({
                "boxes":  out["boxes"][mask].cpu().float(),
                "scores": out["scores"][mask].cpu().float(),
                "labels": out["labels"][mask].cpu().int(),
            })
            gts.append({
                "boxes":  tgt["boxes"].cpu().float(),
                "labels": tgt["labels"].cpu().int(),
            })
        metric.update(preds, gts)

    result = metric.compute()

    if writer and epoch is not None:
        writer.add_scalar(f"mAP/{prefix}_map50",   result["map_50"].item(), epoch)
        writer.add_scalar(f"mAP/{prefix}_map5095", result["map"].item(),    epoch)
        if "map_per_class" in result:
            for i, ap in enumerate(result["map_per_class"]):
                writer.add_scalar(f"AP_class/{prefix}_class{i+1}", ap.item(), epoch)

    return result


# ─────────────────────────────────────────────
#  Val Loss — FRCNN
# ─────────────────────────────────────────────

@torch.no_grad()
def frcnn_evaluate_loss(model, loader, device):
    """
    Tính validation loss của mô hình Faster R-CNN.

    Hàm sử dụng tập validation để tính toán tổng loss trung bình của mô hình
    mà không cập nhật trọng số. Giá trị này được dùng để theo dõi độ ổn định
    của quá trình huấn luyện.

    Parameters:
        model : torchvision.models.detection.FasterRCNN
            Mô hình cần đánh giá.
        loader : DataLoader
            DataLoader của tập validation.
        device : torch.device
            Thiết bị thực thi mô hình.

    Returns:
        float
            Validation loss trung bình trên toàn bộ tập dữ liệu.
    """
    import torch

    was_training = model.training
    model.train()
    total_loss = 0.0
    batches    = 0

    try:
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                loss_dict = model(images, targets)
                loss      = sum(loss_dict.values())

            total_loss += loss.item()
            batches    += 1
    finally:
        if not was_training:
            model.eval()

    return total_loss / max(1, batches)


# ─────────────────────────────────────────────
#  Main Training Loop — FRCNN
# ─────────────────────────────────────────────

def frcnn_run_training(model, dataloaders, optimizer, scheduler,
                       device, num_epochs: int, score_thresh: float,
                       workspace_dir: str, model_name: str = "frcnn",
                       patience: int = 3, min_delta: float = 0.001,
                       logger: logging.Logger = None):
    """
    Thực hiện toàn bộ quá trình huấn luyện mô hình Faster R-CNN.

    Hàm điều phối toàn bộ pipeline huấn luyện bao gồm train, đánh giá,
    ghi TensorBoard, điều chỉnh learning rate, lưu mô hình tốt nhất và
    kích hoạt Early Stopping. Trong mỗi epoch, các chỉ số loss, mAP@0.5,
    mAP@0.5:0.95 và AP của từng lớp được ghi nhận để phục vụ phân tích kết quả.

    Parameters:
        model : torchvision.models.detection.FasterRCNN
            Mô hình cần huấn luyện.
        dataloaders : dict
            Từ điển chứa DataLoader của tập train và validation.
        optimizer : torch.optim.Optimizer
            Optimizer dùng để cập nhật trọng số.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Scheduler điều chỉnh learning rate.
        device : torch.device
            Thiết bị thực thi mô hình.
        num_epochs : int
            Số epoch tối đa của quá trình huấn luyện.
        score_thresh : float
            Ngưỡng confidence dùng khi đánh giá.
        workspace_dir : str
            Thư mục lưu mô hình và log.
        model_name : str, optional
            Tên mô hình dùng khi lưu kết quả.
        patience : int, optional
            Số epoch cho Early Stopping.
        min_delta : float, optional
            Mức cải thiện tối thiểu để được xem là tiến bộ.
        logger : logging.Logger, optional
            Logger dùng để ghi log.

    Returns:
        tuple
            Bao gồm lịch sử huấn luyện và giá trị mAP@0.5 tốt nhất đạt được.
    """
    import os
    import torch

    model_save_path = os.path.join(workspace_dir, "models", f"best_{model_name}.pth")
    os.makedirs(os.path.join(workspace_dir, "models"), exist_ok=True)

    writer     = SummaryWriter(log_dir=os.path.join(workspace_dir, "runs", model_name))
    early_stop = FrcnnEarlyStopping(patience=patience, min_delta=min_delta, path=model_save_path)
    scaler     = torch.amp.GradScaler("cuda")

    history = {
        "train_loss": [], "val_loss":   [],
        "val_map50":  [], "val_map":    [],
        "val_ap_cho": [], "val_ap_meo": [],
    }

    banner = "=" * 60
    msg = f"\n{banner}\nBắt đầu Training — {num_epochs} epochs | Device: {device}\n{banner}"
    print(msg)
    if logger:
        logger.info(msg)

    for epoch in range(1, num_epochs + 1):
        t0  = time.time()
        lr  = scheduler.get_last_lr()
        msg = f"\n--- Epoch {epoch}/{num_epochs}  LR={lr} ---"
        print(msg)
        if logger:
            logger.info(msg)

        avg_loss, loss_details = frcnn_train_one_epoch(
            model, dataloaders["train"], optimizer, scaler,
            device, epoch, writer=writer, logger=logger,
        )
        scheduler.step()

        val_result = frcnn_evaluate_map(
            model, dataloaders["val"], device, score_thresh,
            epoch=epoch, writer=writer, prefix="val",
        )
        val_loss = frcnn_evaluate_loss(model, dataloaders["val"], device)

        map50    = val_result["map_50"].item()
        map_5095 = val_result["map"].item()
        ap_cho   = val_result["map_per_class"][0].item() if "map_per_class" in val_result else -1.0
        ap_meo   = val_result["map_per_class"][1].item() if "map_per_class" in val_result else -1.0

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_map50"].append(map50)
        history["val_map"].append(map_5095)
        history["val_ap_cho"].append(ap_cho)
        history["val_ap_meo"].append(ap_meo)

        writer.add_scalar("Loss/val", val_loss, epoch)

        elapsed = time.time() - t0
        summary = (
            f"  - DONE - Epoch {epoch}  Train loss={avg_loss:.4f}  Val loss={val_loss:.4f}  "
            f"mAP@0.5={map50:.4f}  mAP@0.5:0.95={map_5095:.4f}  "
            f"AP_cho={ap_cho:.4f}  AP_meo={ap_meo:.4f}  time={elapsed:.1f}s"
        )
        print(summary)
        if logger:
            logger.info(summary)

        saved = early_stop(map50, model)
        if saved:
            msg = f"  ---- Lưu model tốt nhất: mAP@0.5 = {early_stop.best_map:.4f} → {model_save_path} ----"
        else:
            msg = f"  - [Early Stopping] Chưa tiến bộ: {early_stop.counter}/{patience}"
        print(msg)
        if logger:
            logger.info(msg)

        if early_stop.early_stop:
            msg = (
                f"\n[THÔNG BÁO] Kích hoạt Early Stopping tại Epoch {epoch}. "
                f"Best mAP@0.5 = {early_stop.best_map:.4f}"
            )
            print(msg)
            if logger:
                logger.info(msg)
            break

    writer.close()
    msg = "\nTraining hoàn tất!"
    print(msg)
    if logger:
        logger.info(msg)

    return history, early_stop.best_map

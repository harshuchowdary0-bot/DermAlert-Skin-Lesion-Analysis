"""Fine-tune EfficientNet on HAM10000 benign vs malignant, export to ONNX.

Replaces the previous 26-feature + RBF-SVM pipeline. That pipeline was not
under-tuned, it was feature-limited: over a grid of scalers and C/gamma it
plateaued at 82-83%, a 593-slot handcrafted vector (448px, tiled multi-angle
GLCM, HSV/Lab histograms) reached only 84.6%, and frozen ResNet50+EfficientNet
embeddings fed to the same SVM reached 83.5%. Its learning curve was flat --
five times the data bought 3.4 points -- so the ceiling was the features.

Protocol, chosen so the number is honest rather than flattering:
  * one image per lesion (download_data.py already does this); HAM10000 ships
    10015 images of 7470 lesions and letting two views of one lesion straddle
    a split reports memorisation as accuracy
  * test  = 15% of the class-balanced 2338, seed 42 -- the SAME split every
    earlier number in this repo used, so the comparison is like for like
  * val   = a separate balanced 400 images, used to pick the epoch
  * train = everything else (6719 images) with a class-balanced sampler;
    training on all of it rather than the balanced 2338 subset is worth
    +2.6 test points
  * the test set is scored once, at the epoch validation chose

Training needs requirements-train.txt (torch). Serving needs only
requirements.txt (onnxruntime), which is why this exports ONNX.
"""
import os
import numpy as np
import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, recall_score,
                             roc_auc_score, confusion_matrix, classification_report)

import features

DATASET_DIR = "dataset"
CATEGORIES = ["benign", "malignant_inflamed"]
SEED = 42
EPOCHS = int(os.environ.get("EPOCHS", 8))
BATCH = 32
VAL_PER_CLASS = 200

BACKBONES = {
    "b0": (tv.models.efficientnet_b0, tv.models.EfficientNet_B0_Weights.IMAGENET1K_V1, 224),
    "b3": (tv.models.efficientnet_b3, tv.models.EfficientNet_B3_Weights.IMAGENET1K_V1, 300),
}
BACKBONE = os.environ.get("BACKBONE", "b0")
CTOR, WEIGHTS, SIZE = BACKBONES[BACKBONE]

dev = "mps" if torch.backends.mps.is_available() else "cpu"


def load_paths():
    paths, labels = [], []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATASET_DIR, category)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(features.VALID_EXTENSIONS):
                paths.append(os.path.join(folder, name))
                labels.append(label)
    return np.array(paths), np.array(labels)


def make_splits(y):
    """test/val/train indices. test matches the repo's historical split."""
    n = int(np.bincount(y).min())
    rng = np.random.RandomState(SEED)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    balanced = np.concatenate([i0[np.sort(rng.choice(len(i0), n, replace=False))], i1])
    _, test = train_test_split(balanced, test_size=0.15, random_state=SEED,
                               stratify=y[balanced])

    pool = np.setdiff1d(np.arange(len(y)), test)
    rng2 = np.random.RandomState(7)
    val = np.concatenate([rng2.choice(pool[y[pool] == c], VAL_PER_CLASS, replace=False)
                          for c in (0, 1)])
    return np.setdiff1d(pool, val), val, test


TRAIN_TF = v2.Compose([
    v2.RandomResizedCrop(SIZE, scale=(0.7, 1.0), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(30),
    v2.ColorJitter(0.2, 0.2, 0.2, 0.05),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(features.MEAN, features.STD),
])


class Lesions(Dataset):
    """Decodes and shaves on the fly. Holds its own arrays so DataLoader
    workers survive macOS spawn, which re-imports the module."""

    def __init__(self, paths, labels, ids, train):
        self.paths, self.labels, self.ids, self.train = paths, labels, ids, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import cv2
        cv2.setNumThreads(0)  # workers deadlock if OpenCV also forks threads
        j = self.ids[i]
        # Both paths prepare at the same source size; training then takes a
        # random crop and evaluation the centre crop app.py will send, so the
        # network never sees a field of view it did not train on.
        img = cv2.imread(self.paths[j])
        _, _, shaved = features.preprocess(img, features.source_size(SIZE))
        if self.train:
            px = torch.from_numpy(cv2.cvtColor(shaved, cv2.COLOR_BGR2RGB).copy())
            return TRAIN_TF(px.permute(2, 0, 1)), int(self.labels[j])
        return torch.from_numpy(features.model_input(shaved, SIZE)[0]), int(self.labels[j])


@torch.no_grad()
def infer(net, loader):
    """Logits with 4-view flip TTA, plus the labels in loader order."""
    out, ys = [], []
    for xb, yb in loader:
        xb = xb.to(dev)
        logits = sum(net(t) for t in (xb, torch.flip(xb, [3]),
                                      torch.flip(xb, [2]), torch.flip(xb, [2, 3]))) / 4
        out.append(logits.cpu())
        ys.append(yb)
    return torch.cat(out), torch.cat(ys).numpy()


def main():
    paths, y = load_paths()
    if len(paths) < 50:
        print("Error: dataset/ is empty. Run 'python download_data.py' first.")
        return
    print(f"{len(paths)} lesions, per class {np.bincount(y)}")

    itr, ival, ite = make_splits(y)

    # Optional extra training lesions (EXTRA=isic_extra.csv, columns path,label).
    # HAM10000 supplies only 794 unique malignant lesions to train on and both
    # backbones overfit them by epoch 6; ISIC-2019 adds 2403 more. These are
    # appended to the training index only -- validation and test stay pure
    # HAM10000, so the number remains comparable to every earlier run.
    extra = os.environ.get("EXTRA", "")
    if extra:
        import csv as _csv
        with open(extra, newline="") as fh:
            rows = [(r["path"], int(r["label"])) for r in _csv.DictReader(fh)
                    if os.path.exists(r["path"])]
        base = len(paths)
        paths = np.concatenate([paths, np.array([p for p, _ in rows])])
        y = np.concatenate([y, np.array([l for _, l in rows])])
        itr = np.concatenate([itr, np.arange(base, len(paths))])
        print(f"+{len(rows)} extra training lesions from {extra}")

    print(f"train {len(itr)} {np.bincount(y[itr])}  "
          f"val {len(ival)} {np.bincount(y[ival])}  "
          f"test {len(ite)} {np.bincount(y[ite])}")
    print(f"backbone efficientnet_{BACKBONE} @ {SIZE}px on {dev}, {EPOCHS} epochs\n")

    counts = np.bincount(y[itr])
    weights = (1.0 / counts)[y[itr]]
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(itr), replacement=True)
    nw = max(0, min(6, (os.cpu_count() or 2) - 2))
    tl = DataLoader(Lesions(paths, y, itr, True), batch_size=BATCH, sampler=sampler,
                    num_workers=nw, persistent_workers=nw > 0)
    vl = DataLoader(Lesions(paths, y, ival, False), batch_size=BATCH, num_workers=nw)
    xl = DataLoader(Lesions(paths, y, ite, False), batch_size=BATCH, num_workers=nw)

    net = CTOR(weights=WEIGHTS)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 2)
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val, best_state = 0.0, None
    for ep in range(EPOCHS):
        net.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            lossf(net(xb), yb).backward()
            opt.step()
        sched.step()

        net.eval()
        logits, yv = infer(net, vl)
        acc = accuracy_score(yv, logits.argmax(1).numpy())
        flag = ""
        if acc > best_val:
            best_val = acc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            flag = "  <- best"
            # Checkpoint every improvement, not just at the end: an interrupted
            # run should still leave a usable model behind.
            torch.save({"backbone": BACKBONE, "size": SIZE, "state_dict": best_state},
                       "model.pt")
        print(f"epoch {ep + 1:>2}/{EPOCHS}  val {acc * 100:.2f}%{flag}", flush=True)

    # Score the test set exactly once, with the weights validation chose.
    net.load_state_dict(best_state)
    net.eval()
    logits, yt = infer(net, xl)
    prob = torch.softmax(logits, 1).numpy()
    pred = prob.argmax(1)

    print(f"\nbest validation accuracy: {best_val * 100:.2f}%")
    print(f"HELD-OUT TEST accuracy:   {accuracy_score(yt, pred) * 100:.2f}%")
    print(f"balanced accuracy:        {balanced_accuracy_score(yt, pred) * 100:.2f}%")
    print(f"recall, malignant:        {recall_score(yt, pred) * 100:.2f}%")
    print(f"ROC AUC:                  {roc_auc_score(yt, prob[:, 1]):.4f}")
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(yt, pred))
    print()
    print(classification_report(yt, pred, target_names=CATEGORIES, digits=3))

    torch.save({"backbone": BACKBONE, "size": SIZE, "state_dict": best_state}, "model.pt")
    net.cpu().eval()
    # external_data=False keeps the weights inside model.onnx. The default
    # spills them into a sidecar model.onnx.data, which deploys fine right up
    # until someone commits only the 0.7 MB graph and the server loads a model
    # with no weights. opset 18 because the exporter cannot downconvert to 17.
    torch.onnx.export(
        net, torch.zeros(1, 3, SIZE, SIZE),
        "model.onnx",
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}},
        opset_version=18, external_data=False,
    )
    size_mb = os.path.getsize("model.onnx") / 1e6
    print(f"Saved model.pt and model.onnx ({size_mb:.1f} MB). "
          f"app.py serves the ONNX file, so deployment needs no torch.")


if __name__ == "__main__":
    main()

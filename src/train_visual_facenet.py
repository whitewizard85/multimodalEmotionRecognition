"""
Facial emotion classification via transfer learning from InceptionResnetV1
pretrained on VGGFace2 (face recognition), instead of ImageNet-pretrained
ResNet50 (train_visual_resnet50.py, which only reached ~56% validation
accuracy -- a modest improvement over the from-scratch custom CNN's 52%,
much smaller than the gain transfer learning gave audio).

Rationale: ImageNet pretraining teaches general object-category features
(distinguishing dogs from cars), not the fine-grained facial geometry and
texture that discriminates expressions. A backbone pretrained on face
RECOGNITION (VGGFace2: predicting identity across ~9,000 people) is forced
to learn much more facial-structure-relevant features, which is well
established in FER literature as transferring far better to expression
recognition than generic ImageNet features. This uses facenet-pytorch's
InceptionResnetV1(pretrained='vggface2') -- a real, actively-maintained,
verified package (confirmed by reading its source directly before writing
this script, not assumed from memory).

Same two-phase fine-tuning approach as the ResNet50 attempt:
  Phase 1: backbone frozen, train only the new classification head.
  Phase 2: unfreeze the final Inception-ResNet block (block8) plus the
           embedding projection layers, continue at a much lower LR.

Requires the torch/transformers venv (NOT venv_visual) -- same environment
train_text.py and train_audio_wav2vec2.py used. Install facenet-pytorch
first: pip install facenet-pytorch
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from facenet_pytorch import InceptionResnetV1

from common import EMOTIONS, NUM_CLASSES, RANDOM_SEED

IMG_SIZE = 160  # InceptionResnetV1's expected input size


def fixed_image_standardization(image_tensor):
    # Exact preprocessing facenet-pytorch's own pretrained weights expect
    # (verified from facenet_pytorch/models/mtcnn.py source directly)
    return (image_tensor - 127.5) / 128.0


class FaceEmotionDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.samples = []
        for idx, emotion in enumerate(EMOTIONS):
            d = os.path.join(root_dir, emotion)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                self.samples.append((os.path.join(d, fname), idx))

        aug = [
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.15, 0.15), shear=15, scale=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
        ] if augment else []
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            *aug,
            transforms.ToTensor(),  # -> [0, 1] float
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img) * 255.0  # back to [0, 255] for fixed_image_standardization
        img = fixed_image_standardization(img)
        return img, label


class FaceEmotionModel(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained="vggface2", classify=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        emb = self.backbone(x)  # (batch, 512)
        return self.classifier(self.dropout(emb))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_top(self):
        # Unfreeze only the final Inception-ResNet block + embedding
        # projection layers -- keep earlier, more generic facial-feature
        # layers frozen, matching the ResNet50 script's partial-unfreeze approach
        for name, p in self.backbone.named_parameters():
            if any(name.startswith(prefix) for prefix in ["block8", "last_linear", "last_bn"]):
                p.requires_grad = True

    def sync_bn_eval_mode(self):
        """Set each BatchNorm layer in the backbone to eval() if ALL of its
        own parameters are frozen (requires_grad=False), else train(). This
        must be called every epoch AFTER model.train(), since train() would
        otherwise recursively re-enable every BatchNorm layer -- including
        ones we deliberately froze -- letting their running statistics drift
        away from the pretrained values even though their weights never
        update. A single phase-level "backbone_frozen" flag isn't precise
        enough once phase 2 unfreezes only PART of the backbone (block8,
        last_bn) while the rest (block35, block17, reduction blocks, etc.)
        must stay in eval mode throughout -- this handles that correctly."""
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                frozen = not any(p.requires_grad for p in module.parameters())
                module.eval() if frozen else module.train()


def run_epoch(model, loader, optimizer, device, criterion, train: bool):
    if train:
        model.train()
        # See sync_bn_eval_mode's docstring: this must run every epoch after
        # model.train(), which would otherwise re-enable BatchNorm layers we
        # deliberately froze, letting their running stats drift.
        model.sync_bn_eval_mode()
    else:
        model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader.dataset), acc, all_preds, all_labels


def train_phase(model, train_loader, val_loader, device, lr, max_epochs, patience, phase_name):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc, patience_counter, best_state = 0.0, 0, None

    for epoch in range(max_epochs):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, optimizer, device, criterion, train=True)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, optimizer, device, criterion, train=False)
        print(f"[{phase_name}] Epoch {epoch+1}/{max_epochs} - "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{phase_name}] Early stopping (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="visual_facenet_model.pt")
    ap.add_argument("--phase1_epochs", type=int, default=20)
    ap.add_argument("--phase2_epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--skip_phase2", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = FaceEmotionDataset(os.path.join(args.data_dir, "train"), augment=True)
    val_dataset = FaceEmotionDataset(os.path.join(args.data_dir, "val"), augment=False)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = FaceEmotionModel().to(device)

    print("=" * 70)
    print("PHASE 1: training classification head, backbone frozen")
    print("=" * 70)
    model.freeze_backbone()
    model, phase1_acc = train_phase(model, train_loader, val_loader, device,
                                     lr=1e-3, max_epochs=args.phase1_epochs,
                                     patience=6, phase_name="Phase 1")
    print(f"Phase 1 best val accuracy: {phase1_acc:.4f}")

    if not args.skip_phase2:
        print("=" * 70)
        print("PHASE 2: fine-tuning final backbone block at a low learning rate")
        print("=" * 70)
        model.unfreeze_top()
        model, phase2_acc = train_phase(model, train_loader, val_loader, device,
                                         lr=1e-5, max_epochs=args.phase2_epochs,
                                         patience=6, phase_name="Phase 2")
        print(f"Phase 2 best val accuracy: {phase2_acc:.4f}")

    _, final_acc, preds, labels = run_epoch(model, val_loader, None, device, nn.CrossEntropyLoss(), train=False)
    print("\nFinal validation accuracy:", final_acc)
    print(classification_report(labels, preds, target_names=EMOTIONS))

    torch.save(model.state_dict(), args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

"""
=============================================================
  PLANT SEEDLING CLASSIFICATION - Tkinter GUI
  Based on: "Classification of Plant Seedlings Using Deep CNN Architectures"
  Audisankara College of Engineering and Technology
=============================================================

HOW TO USE:
  1. Make sure training is complete and 'plant_model.pth' is saved
     (The training script saves it automatically after completion)
  2. Run:  python plant_gui.py
  3. Click "Upload" to select a plant image
  4. Click "Classify Plant" to get prediction

REQUIREMENTS:
  pip install torch torchvision pillow matplotlib scikit-learn numpy seaborn
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, sys, threading
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:
    print("ERROR: Pillow not found. Run: pip install pillow")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import torch
    from torch import nn
    from torchvision import models, transforms
except ImportError:
    print("ERROR: PyTorch not found. Run: pip install torch torchvision")
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
APP_TITLE  = "PLANT SEEDLING CLASSIFICATION"
MODEL_PATH = "plant_model.pth"

CLASS_NAMES = [
    "Black-grass",
    "Charlock",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Fat Hen",
    "Loose Silky-bent",
    "Maize",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Sugar beet",
]

PLANT_TYPE = {
    "Black-grass"              : ("Weed",  "#8B4513"),
    "Charlock"                 : ("Weed",  "#8B4513"),
    "Cleavers"                 : ("Weed",  "#8B4513"),
    "Common Chickweed"         : ("Weed",  "#8B4513"),
    "Common wheat"             : ("Crop",  "#1a6b1a"),
    "Fat Hen"                  : ("Weed",  "#8B4513"),
    "Loose Silky-bent"         : ("Weed",  "#8B4513"),
    "Maize"                    : ("Crop",  "#1a6b1a"),
    "Scentless Mayweed"        : ("Weed",  "#8B4513"),
    "Shepherds Purse"          : ("Weed",  "#8B4513"),
    "Small-flowered Cranesbill": ("Weed",  "#8B4513"),
    "Sugar beet"               : ("Crop",  "#1a6b1a"),
}

COLOR_BG        = "#f0f4f0"
COLOR_HDR_BG    = "#2d6a2d"
COLOR_HDR_FG    = "white"
COLOR_BAR_BG    = "#4a8c4a"
COLOR_GREEN_BTN = "#3a7d3a"
COLOR_RED_BTN   = "#b22222"
COLOR_BLUE_BTN  = "#1565c0"

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ═════════════════════════════════════════════════════════════════════════════
#  MODEL HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def build_resnet18(num_classes=12):
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(path):
    model = build_resnet18()
    state = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model

def predict_image(model, img_path):
    img  = Image.open(img_path).convert("RGB")
    inp  = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1)[0].numpy()
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════
class PlantApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x700")
        self.configure(bg=COLOR_BG)
        self.resizable(True, True)
        self.model    = None
        self.img_path = None
        self._photo   = None
        self._build_ui()
        self._load_model_async()

    # ─────────────────────────────────────────────────────────────────────────
    #  BUILD UI
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        self._build_menu_bar()
        self._build_body()
        self._build_status_bar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=COLOR_HDR_BG, height=72)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr,
                 text="🌱  " + APP_TITLE,
                 font=("Arial", 20, "bold"),
                 fg=COLOR_HDR_FG, bg=COLOR_HDR_BG
                 ).pack(side="left", padx=20, pady=16)
        tk.Label(hdr,
                 text="ResNet-18  |  Deep CNN  |  12 Plant Species",
                 font=("Arial", 10),
                 fg="#a8d5a2", bg=COLOR_HDR_BG
                 ).pack(side="right", padx=20)

    def _build_menu_bar(self):
        bar = tk.Frame(self, bg=COLOR_BAR_BG, height=42)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        s = dict(font=("Arial", 10, "bold"), relief="flat", cursor="hand2",
                 padx=20, pady=7)
        tk.Button(bar, text="Classify Plant",
                  bg="#6abf6a", fg="white",
                  command=self._upload_image, **s).pack(side="left", padx=10, pady=4)
        tk.Button(bar, text="Exit",
                  bg=COLOR_RED_BTN, fg="white",
                  command=self.destroy, **s).pack(side="left", padx=4, pady=4)

    def _build_body(self):
        body = tk.Frame(self, bg=COLOR_BG)
        body.pack(fill="both", expand=True, padx=14, pady=10)

        # ── LEFT: Image Panel ─────────────────────────────────────────────
        left = tk.LabelFrame(body, text="  Plant Image  ",
                             font=("Arial", 10, "bold"),
                             bg=COLOR_BG, fg=COLOR_HDR_BG,
                             relief="groove", bd=2)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.img_canvas = tk.Canvas(left, bg="#e8f5e9", width=370, height=350,
                                    highlightthickness=1,
                                    highlightbackground="#b2dfdb")
        self.img_canvas.pack(padx=10, pady=10, fill="both", expand=True)
        self._show_placeholder()

        btn_row = tk.Frame(left, bg=COLOR_BG)
        btn_row.pack(pady=8)
        s2 = dict(font=("Arial", 10, "bold"), relief="flat",
                  cursor="hand2", padx=14, pady=7)
        tk.Button(btn_row, text="📂  Upload",
                  bg=COLOR_GREEN_BTN, fg="white",
                  command=self._upload_image, **s2).pack(side="left", padx=5)
        tk.Button(btn_row, text="⚡  Predict",
                  bg=COLOR_BLUE_BTN, fg="white",
                  command=self._predict, **s2).pack(side="left", padx=5)
        tk.Button(btn_row, text="🔄  Clear",
                  bg="#757575", fg="white",
                  command=self._clear, **s2).pack(side="left", padx=5)

        # ── RIGHT: Result Panel ───────────────────────────────────────────
        right = tk.LabelFrame(body, text="  Prediction Result  ",
                              font=("Arial", 10, "bold"),
                              bg=COLOR_BG, fg=COLOR_HDR_BG,
                              relief="groove", bd=2, width=420)
        right.pack(side="right", fill="both", expand=True)

        self.result_var = tk.StringVar(value="Awaiting prediction…")
        tk.Label(right, textvariable=self.result_var,
                 font=("Arial", 16, "bold"),
                 bg=COLOR_BG, fg=COLOR_HDR_BG,
                 wraplength=380, justify="center"
                 ).pack(pady=(14, 2))

        self.conf_var = tk.StringVar()
        tk.Label(right, textvariable=self.conf_var,
                 font=("Arial", 11), bg=COLOR_BG, fg="#444").pack()

        self.type_var = tk.StringVar()
        tk.Label(right, textvariable=self.type_var,
                 font=("Arial", 12, "bold"),
                 bg=COLOR_BG, fg="#1a6b1a").pack(pady=2)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=14, pady=6)

        # Chart area
        self.chart_frame = tk.Frame(right, bg=COLOR_BG)
        self.chart_frame.pack(fill="both", expand=True, padx=4, pady=2)
        self._draw_empty_chart()

        # Bottom buttons
        vrow = tk.Frame(right, bg=COLOR_BG)
        vrow.pack(pady=8)
        s3 = dict(font=("Arial", 9, "bold"), relief="flat",
                  cursor="hand2", padx=10, pady=5)
        tk.Button(vrow, text="📊 Confusion Matrix",
                  bg="#7b1fa2", fg="white",
                  command=self._show_confusion_matrix, **s3).pack(side="left", padx=3)
        tk.Button(vrow, text="📈 Training Graphs",
                  bg="#0277bd", fg="white",
                  command=self._show_training_graphs, **s3).pack(side="left", padx=3)
        tk.Button(vrow, text="🌿 All Classes",
                  bg="#2e7d32", fg="white",
                  command=self._show_all_classes, **s3).pack(side="left", padx=3)

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="🔄  Loading model…")
        bar = tk.Frame(self, bg="#c8e6c9", height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self.status_var,
                 font=("Arial", 9), bg="#c8e6c9", fg="#1b5e20",
                 anchor="w").pack(fill="x", padx=10, pady=3)

    # ─────────────────────────────────────────────────────────────────────────
    #  CHART HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_empty_chart(self):
        for w in self.chart_frame.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(4.4, 2.9), dpi=78)
        fig.patch.set_facecolor(COLOR_BG)
        ax.set_facecolor(COLOR_BG)
        ax.set_title("Upload an image and click Predict", fontsize=8, color="#888")
        ax.axis("off")
        cv = FigureCanvasTkAgg(fig, master=self.chart_frame)
        cv.draw(); cv.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def _draw_prob_chart(self, probs):
        for w in self.chart_frame.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(4.4, 3.1), dpi=78)
        fig.patch.set_facecolor(COLOR_BG)
        ax.set_facecolor("#f9fbe7")
        colors = ["#43a047" if p == max(probs) else "#a5d6a7" for p in probs]
        short  = [c.replace("Small-flowered ", "SF-").replace("Scentless ", "Sc-") for c in CLASS_NAMES]
        bars   = ax.barh(short, probs, color=colors, edgecolor="white", height=0.62)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("Probability", fontsize=8)
        ax.set_title("Predicting class name", fontsize=9, fontweight="bold", color="#1b5e20")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        max_idx = int(np.argmax(probs))
        ax.get_yticklabels()[max_idx].set_color("#1b5e20")
        ax.get_yticklabels()[max_idx].set_fontweight("bold")
        for bar, p in zip(bars, probs):
            if p > 0.015:
                ax.text(p + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{p:.3f}", va="center", fontsize=6.5)
        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=self.chart_frame)
        cv.draw(); cv.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    #  MODEL LOADING
    # ─────────────────────────────────────────────────────────────────────────
    def _load_model_async(self):
        def _load():
            if not os.path.exists(MODEL_PATH):
                self.status_var.set(
                    f"⚠️  '{MODEL_PATH}' not found — finish training first, "
                    "then place plant_model.pth in the same folder as this file.")
                return
            try:
                self.model = load_model(MODEL_PATH)
                self.status_var.set("✅  Model loaded! Upload a plant image to classify.")
            except Exception as e:
                self.status_var.set(f"❌  Model error: {e}")
        threading.Thread(target=_load, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  ACTIONS
    # ─────────────────────────────────────────────────────────────────────────
    def _show_placeholder(self):
        self.img_canvas.delete("all")
        cx = self.img_canvas.winfo_width()  // 2 or 185
        cy = self.img_canvas.winfo_height() // 2 or 175
        self.img_canvas.create_text(
            cx, cy,
            text="📷\n\nNo image uploaded\nClick 'Upload' to begin",
            font=("Arial", 12), fill="#888888", justify="center")

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Plant Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("All files", "*.*")])
        if not path:
            return
        self.img_path = path
        self._display_image(path)
        self.result_var.set("Image loaded. Click ⚡ Predict to classify.")
        self.conf_var.set(""); self.type_var.set("")
        self._draw_empty_chart()
        self.status_var.set(f"📂  Loaded: {os.path.basename(path)}")

    def _display_image(self, path):
        img = Image.open(path).convert("RGB")
        w = self.img_canvas.winfo_width()  or 370
        h = self.img_canvas.winfo_height() or 350
        img.thumbnail((w - 12, h - 12), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(w // 2, h // 2,
                                     image=self._photo, anchor="center")

    def _predict(self):
        if self.model is None:
            messagebox.showwarning("Model Not Ready",
                                   f"'{MODEL_PATH}' is not loaded yet.\n"
                                   "Please wait for training to complete.")
            return
        if self.img_path is None:
            messagebox.showinfo("No Image", "Please upload a plant image first.")
            return
        self.status_var.set("⚡  Running prediction…")
        self.update()
        try:
            cls, conf, probs = predict_image(self.model, self.img_path)
            ptype, pcolor = PLANT_TYPE.get(cls, ("Unknown", "#333"))
            self.result_var.set(f"🌱  {cls}")
            self.conf_var.set(f"Confidence:  {conf * 100:.2f}%")
            self.type_var.set(f"Type: {ptype}")
            self.type_lbl_ref = self.type_var   # keep ref
            self._draw_prob_chart(probs)
            self.status_var.set(
                f"✅  Predicted: {cls}   |   Confidence: {conf*100:.2f}%   |   Type: {ptype}")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.status_var.set("❌  Prediction failed.")

    def _clear(self):
        self.img_path = None; self._photo = None
        self._show_placeholder()
        self.result_var.set("Awaiting prediction…")
        self.conf_var.set(""); self.type_var.set("")
        self._draw_empty_chart()
        self.status_var.set("🔄  Cleared. Ready.")

    # ─────────────────────────────────────────────────────────────────────────
    #  POP-UP WINDOWS
    # ─────────────────────────────────────────────────────────────────────────
    def _show_all_classes(self):
        win = tk.Toplevel(self)
        win.title("All 12 Plant Species")
        win.geometry("530x490")
        win.configure(bg=COLOR_BG)
        tk.Label(win, text="🌿  All 12 Plant Species",
                 font=("Arial", 14, "bold"),
                 bg=COLOR_HDR_BG, fg="white").pack(fill="x")
        tk.Label(win,
                 text="3 Crops  •  9 Weeds  •  Dataset: ~5,000 images  •  Kaggle V2 Plant Seedlings",
                 font=("Arial", 9), bg=COLOR_BG, fg="#555").pack(pady=5)

        frame = tk.Frame(win, bg=COLOR_BG)
        frame.pack(fill="both", expand=True, padx=14, pady=4)

        for c, (txt, w) in enumerate(zip(["#", "Class Name", "Type", "Category"],
                                         [3, 26, 7, 12])):
            tk.Label(frame, text=txt, font=("Arial", 10, "bold"),
                     bg="#4a8c4a", fg="white", width=w,
                     anchor="w", padx=6, pady=4
                     ).grid(row=0, column=c, sticky="nsew", padx=1, pady=1)

        for i, cls in enumerate(CLASS_NAMES, 1):
            bg = "#f1f8e9" if i % 2 == 0 else "#ffffff"
            ptype, pcolor = PLANT_TYPE.get(cls, ("?", "#000"))
            desc = "Crop plant" if ptype == "Crop" else "Weed plant"
            tk.Label(frame, text=str(i), font=("Arial", 10), bg=bg,
                     width=3, anchor="w", padx=6
                     ).grid(row=i, column=0, sticky="nsew", padx=1, pady=1)
            tk.Label(frame, text=cls, font=("Arial", 10), bg=bg,
                     width=26, anchor="w", padx=6
                     ).grid(row=i, column=1, sticky="nsew", padx=1, pady=1)
            tk.Label(frame, text=ptype, font=("Arial", 10, "bold"),
                     fg=pcolor, bg=bg, width=7, anchor="w", padx=6
                     ).grid(row=i, column=2, sticky="nsew", padx=1, pady=1)
            tk.Label(frame, text=desc, font=("Arial", 9), bg=bg,
                     width=12, anchor="w", padx=6
                     ).grid(row=i, column=3, sticky="nsew", padx=1, pady=1)

        tk.Label(win,
                 text="Source: https://www.kaggle.com/vbookshelf/v2-plant-seedlings-dataset",
                 font=("Arial", 8, "italic"), bg=COLOR_BG, fg="#888"
                 ).pack(pady=8)

    def _show_confusion_matrix(self):
        win = tk.Toplevel(self)
        win.title("Confusion Matrix")
        win.geometry("780x630")
        win.configure(bg=COLOR_BG)
        tk.Label(win, text="🧮  Validation Accuracy – Confusion Matrix",
                 font=("Arial", 13, "bold"),
                 bg=COLOR_HDR_BG, fg="white").pack(fill="x")
        tk.Label(win,
                 text="Rows = True class  |  Columns = Predicted class  |  Diagonal = Correct predictions",
                 font=("Arial", 9), bg=COLOR_BG, fg="#555").pack(pady=4)

        # Load saved file if exists
        if os.path.exists("confusion_matrix.png"):
            try:
                img = Image.open("confusion_matrix.png")
                img.thumbnail((740, 520), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                lbl   = tk.Label(win, image=photo, bg=COLOR_BG)
                lbl.image = photo
                lbl.pack(pady=6)
                return
            except Exception:
                pass

        # Generate sample heatmap
        tk.Label(win,
                 text="(confusion_matrix.png not found – showing illustrative sample)",
                 font=("Arial", 8, "italic"), bg=COLOR_BG, fg="#c62828").pack()

        np.random.seed(7)
        n  = len(CLASS_NAMES)
        cm = np.random.randint(0, 4, (n, n))
        for i in range(n):
            cm[i, i] = np.random.randint(20, 28)

        fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=85)
        fig.patch.set_facecolor(COLOR_BG)
        im = ax.imshow(cm, cmap="Greens")
        short = [c[:13] for c in CLASS_NAMES]
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_xlabel("Predicted Class", fontsize=9)
        ax.set_ylabel("True Class", fontsize=9)
        ax.set_title("Confusion Matrix (val accuracy)", fontsize=10, fontweight="bold")
        for i in range(n):
            for j in range(n):
                color = "white" if cm[i, j] > cm.max() * 0.55 else "black"
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=6, color=color)
        fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
        fig.tight_layout()

        f2 = tk.Frame(win, bg=COLOR_BG)
        f2.pack(fill="both", expand=True, padx=6, pady=4)
        cv = FigureCanvasTkAgg(fig, master=f2)
        cv.draw(); cv.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def _show_training_graphs(self):
        win = tk.Toplevel(self)
        win.title("Training Graphs")
        win.geometry("840x470")
        win.configure(bg=COLOR_BG)
        tk.Label(win, text="📈  Training & Validation – Loss and Accuracy",
                 font=("Arial", 13, "bold"),
                 bg=COLOR_HDR_BG, fg="white").pack(fill="x")

        # Load saved file if exists
        if os.path.exists("metrics.png"):
            try:
                img = Image.open("metrics.png")
                img.thumbnail((800, 400), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                lbl   = tk.Label(win, image=photo, bg=COLOR_BG)
                lbl.image = photo
                lbl.pack(pady=10)
                return
            except Exception:
                pass

        tk.Label(win,
                 text="(metrics.png not found – showing illustrative sample graph)",
                 font=("Arial", 8, "italic"), bg=COLOR_BG, fg="#c62828").pack(pady=3)

        epochs    = np.arange(1, 51)
        np.random.seed(1)
        t_loss = 2.05 * np.exp(-0.075 * epochs) + 0.14 + np.random.normal(0, 0.018, 50)
        v_loss = 2.12 * np.exp(-0.068 * epochs) + 0.17 + np.random.normal(0, 0.025, 50)
        t_acc  = 0.96 * (1 - np.exp(-0.082 * epochs)) + np.random.normal(0, 0.009, 50)
        v_acc  = 0.94 * (1 - np.exp(-0.076 * epochs)) + np.random.normal(0, 0.012, 50)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4), dpi=84)
        fig.patch.set_facecolor(COLOR_BG)

        ax1.plot(epochs, t_loss, color="navy",     marker="o", ms=2, lw=1.5, label="Training Loss")
        ax1.plot(epochs, v_loss, color="firebrick", marker="*", ms=3, lw=1.5, label="Validation Loss")
        ax1.set_title("Training & Validation Loss", fontweight="bold", fontsize=10)
        ax1.set_xlabel("Epochs"); ax1.set_ylabel("Loss")
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        ax1.set_facecolor("#f9fbe7")

        ax2.plot(epochs, t_acc, color="navy",     marker="o", ms=2, lw=1.5, label="Training Accuracy")
        ax2.plot(epochs, v_acc, color="firebrick", marker="*", ms=3, lw=1.5, label="Validation Accuracy")
        ax2.set_title("Training & Validation Accuracy", fontweight="bold", fontsize=10)
        ax2.set_xlabel("Epochs"); ax2.set_ylabel("Accuracy")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        ax2.set_facecolor("#f9fbe7")

        fig.tight_layout(pad=2)
        f2 = tk.Frame(win, bg=COLOR_BG)
        f2.pack(fill="both", expand=True, padx=8, pady=6)
        cv = FigureCanvasTkAgg(fig, master=f2)
        cv.draw(); cv.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = PlantApp()
    app.mainloop()

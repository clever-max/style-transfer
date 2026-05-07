"""
图形化风格迁移训练界面 (tkinter)
—— 支持选择内容图、多张风格图、调节参数、实时进度显示
"""
import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import run_style_transfer
from step_comparison import generate_step_comparison


class StyleTransferGUI:
    """神经风格迁移图形界面"""

    def __init__(self, root):
        self.root = root
        self.root.title("神经风格迁移 - Neural Style Transfer")
        self.root.geometry("1000x720")
        self.root.minsize(900, 650)
        self.root.resizable(True, True)

        # 数据
        self.content_path = None
        self.style_paths = []          # 已选风格图路径列表
        self.style_vars = {}           # {路径: tk.BooleanVar} 用于勾选框
        self.is_training = False

        # 预览图片缓存
        self._preview_tk = None

        self._build_ui()

    # ==================== 界面构建 ====================

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        left_panel = ttk.Frame(main_frame, width=420)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        left_panel.pack_propagate(False)

        self._build_content_section(left_panel)
        self._build_style_section(left_panel)
        self._build_params_section(left_panel)
        self._build_action_section(left_panel)

        # 右侧预览 + 日志面板
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_preview_section(right_panel)
        self._build_log_section(right_panel)

    def _label(self, parent, text, **kw):
        return ttk.Label(parent, text=text, **kw)

    def _heading(self, parent, text):
        lbl = ttk.Label(parent, text=text, font=("Microsoft YaHei", 11, "bold"))
        lbl.pack(anchor=tk.W, pady=(12, 4))
        return lbl

    def _sep(self, parent):
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

    # ---- 内容图区域 ----
    def _build_content_section(self, parent):
        frame = ttk.LabelFrame(parent, text="内容图像 (Content)", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="选择内容图像...", command=self._choose_content).pack(side=tk.LEFT)
        self._content_label = ttk.Label(btn_row, text="未选择", foreground="gray")
        self._content_label.pack(side=tk.LEFT, padx=10)

    # ---- 风格图区域 ----
    def _build_style_section(self, parent):
        frame = ttk.LabelFrame(parent, text="风格图像 (Style) — 可多选批量训练", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(btn_row, text="添加风格图像...", command=self._add_styles).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="移除选中", command=self._remove_style).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="清空列表", command=self._clear_styles).pack(side=tk.LEFT)

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._style_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                         height=6, selectmode=tk.EXTENDED,
                                         exportselection=False)
        scrollbar.config(command=self._style_listbox.yview)
        self._style_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._style_count_label = ttk.Label(frame, text="已选: 0 张", foreground="gray")
        self._style_count_label.pack(anchor=tk.W, pady=(2, 0))

        self._style_listbox.bind('<<ListboxSelect>>', self._on_style_select)

    # ---- 参数区域 ----
    def _build_params_section(self, parent):
        frame = ttk.LabelFrame(parent, text="训练参数", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        fields = [
            ("迭代步数:", "steps", "300"),
            ("内容权重 α:", "cw", "1"),
            ("风格权重 β:", "sw", "100"),
            ("TV 权重 γ:", "tv", "0.001"),
        ]

        self._param_entries = {}
        for i, (label, key, default) in enumerate(fields):
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=15).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(row, textvariable=var, width=18)
            entry.pack(side=tk.LEFT)
            self._param_entries[key] = var

    # ---- 操作按钮区域 ----
    def _build_action_section(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(8, 4))

        self._train_btn = ttk.Button(frame, text="▶ 开始训练",
                                     command=self._start_training)
        self._train_btn.pack(side=tk.LEFT, ipadx=20, ipady=4)

        self._stop_btn = ttk.Button(frame, text="■ 停止",
                                    command=self._stop_training, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=8)

        self._progress = ttk.Progressbar(parent, mode='determinate', length=400)
        self._progress.pack(fill=tk.X, pady=(6, 0))

        self._status_label = ttk.Label(parent, text="就绪", foreground="gray")
        self._status_label.pack(anchor=tk.W)

    # ---- 预览区域 ----
    def _build_preview_section(self, parent):
        frame = ttk.LabelFrame(parent, text="图像预览", padding=8)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        canvas_frame = ttk.Frame(frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self._preview_canvas = tk.Canvas(canvas_frame, bg="#f0f0f0",
                                         width=520, height=280,
                                         highlightthickness=0)
        self._preview_canvas.pack(fill=tk.BOTH, expand=True)
        self._preview_canvas.create_text(260, 140, text="选择图像后将显示预览",
                                         fill="gray", font=("Microsoft YaHei", 11))

    # ---- 日志区域 ----
    def _build_log_section(self, parent):
        frame = ttk.LabelFrame(parent, text="训练日志", padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        self._log_text = scrolledtext.ScrolledText(frame, height=10,
                                                    font=("Consolas", 9),
                                                    wrap=tk.WORD)
        self._log_text.pack(fill=tk.BOTH, expand=True)
        self._log_text.insert(tk.END, "欢迎使用神经风格迁移训练工具\n")
        self._log_text.insert(tk.END, "-" * 50 + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    # ==================== 交互逻辑 ====================

    def _log(self, message):
        """线程安全地向日志区追加文本"""
        self.root.after(0, self._log_impl, message)

    def _log_impl(self, message):
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, message + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _choose_content(self):
        path = filedialog.askopenfilename(
            title="选择内容图像",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")]
        )
        if path:
            self.content_path = path
            self._content_label.config(text=os.path.basename(path), foreground="black")
            self._show_preview(path)

    def _add_styles(self):
        paths = filedialog.askopenfilenames(
            title="选择风格图像（可多选）",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")]
        )
        for path in paths:
            if path not in self.style_paths:
                self.style_paths.append(path)
                self._style_listbox.insert(tk.END, os.path.basename(path))
        self._update_style_count()

    def _remove_style(self):
        selected = self._style_listbox.curselection()
        for idx in reversed(selected):
            del self.style_paths[idx]
            self._style_listbox.delete(idx)
        self._update_style_count()

    def _clear_styles(self):
        self.style_paths.clear()
        self._style_listbox.delete(0, tk.END)
        self._update_style_count()

    def _update_style_count(self):
        n = len(self.style_paths)
        self._style_count_label.config(text=f"已选: {n} 张")

    def _on_style_select(self, event=None):
        sel = self._style_listbox.curselection()
        if sel:
            path = self.style_paths[sel[0]]
            self._show_preview(path)

    def _show_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
            canvas_w = self._preview_canvas.winfo_width() or 520
            canvas_h = self._preview_canvas.winfo_height() or 280
            scale = min(canvas_w / img.width, canvas_h / img.height)
            new_w, new_h = int(img.width * scale), int(img.height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            self._preview_tk = ImageTk.PhotoImage(img)
            self._preview_canvas.delete("all")
            cx, cy = canvas_w // 2, canvas_h // 2
            self._preview_canvas.create_image(cx, cy,
                                              image=self._preview_tk, anchor=tk.CENTER)
            self._preview_canvas.create_text(cx, canvas_h - 15,
                                             text=os.path.basename(path),
                                             fill="white",
                                             font=("Microsoft YaHei", 9))
        except Exception as e:
            self._log(f"[预览错误] {e}")

    # ==================== 训练控制 ====================

    def _start_training(self):
        if not self.content_path:
            messagebox.showwarning("提示", "请先选择内容图像")
            return
        if not self.style_paths:
            messagebox.showwarning("提示", "请至少添加一张风格图像")
            return

        try:
            steps = int(self._param_entries["steps"].get())
            cw = float(self._param_entries["cw"].get())
            sw = float(self._param_entries["sw"].get())
            tv = float(self._param_entries["tv"].get())
        except ValueError:
            messagebox.showerror("参数错误", "请确保所有参数为有效数字")
            return

        self.is_training = True
        self._train_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._progress.config(value=0)

        self._log("=" * 50)
        self._log(f"开始批量训练 — 共 {len(self.style_paths)} 个风格")

        thread = threading.Thread(target=self._training_thread,
                                  args=(steps, cw, sw, tv, 50),
                                  daemon=True)
        thread.start()

    def _stop_training(self):
        self.is_training = False
        self._log("[用户中断] 将在当前风格训练完成后停止")

    def _training_thread(self, steps, cw, sw, tv, save_interval):
        total_styles = len(self.style_paths)

        for idx, style_path in enumerate(self.style_paths):
            if not self.is_training:
                self._log("[已停止] 训练被用户中断")
                break

            style_name = os.path.basename(style_path)
            self._log(f"\n▶ [{idx+1}/{total_styles}] 风格: {style_name}")
            self._set_status(f"训练中 [{idx+1}/{total_styles}]: {style_name}")

            last_logged_step = [-1]

            def progress_cb(step, total, c_loss, s_loss, tv_loss, total_loss, image_tensor):
                if not self.is_training:
                    return

                overall_progress = (idx * total + step) / (total_styles * total)
                self.root.after(0, self._set_progress,
                                int(overall_progress * 100))

                if step - last_logged_step[0] >= 20 or step == total:
                    last_logged_step[0] = step
                    self._log(f"  Step {step:4d}/{total} | "
                              f"Total={total_loss:.2e} "
                              f"C={c_loss:.2e} S={s_loss:.2e} TV={tv_loss:.2e}")

            try:
                exp_dir = run_style_transfer(
                    content_path=self.content_path,
                    style_path=style_path,
                    content_weight=cw,
                    style_weight=sw,
                    tv_weight=tv,
                    num_steps=steps,
                    save_interval=save_interval,
                    progress_callback=progress_cb,
                )

                # 生成逐步对比拼合图
                inter_dir = os.path.join(exp_dir, "intermediate")
                step_files = sorted(os.listdir(inter_dir)) if os.path.exists(inter_dir) else []
                generate_step_comparison(exp_dir, step_files)

                self._log(f"  ✓ 完成! 结果文件夹: {exp_dir}")

            except Exception as e:
                self._log(f"  ✗ 训练出错: {e}")

        self.root.after(0, self._training_done)

    def _set_progress(self, value):
        self._progress.config(value=min(value, 100))

    def _set_status(self, text):
        self._status_label.config(text=text)

    def _training_done(self):
        self._progress.config(value=100)
        self._set_status("训练完成 ✓")
        self._train_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._log("\n" + "=" * 50)
        self._log("全部训练完成!")
        self._log(f"结果保存在 experiments/ 目录下")
        messagebox.showinfo("完成", "所有风格迁移训练已完成!\n请查看 experiments/ 文件夹")


def main():
    root = tk.Tk()
    app = StyleTransferGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

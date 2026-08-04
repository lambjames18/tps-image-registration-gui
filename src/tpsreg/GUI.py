"""Tkinter view for the multimodal image registration application.

Defines the abstract :class:`ViewInterface` the presenter talks to, plus the
concrete Tk implementation and its auxiliary viewer windows.
"""

import argparse
import logging
import os
import sys
import tkinter as tk
from abc import ABC, abstractmethod
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageTk

from tpsreg import __version__, overlays, validation
from tpsreg.presenter import ApplicationPresenter, CropMode, DataFormat, TransformType
from tpsreg.resources_util import apply_window_icon, theme_path
from tpsreg.theme import apply_to_window, get_palette, palette_of

logger = logging.getLogger(__name__)

#: Labels for the smoothing selector. Plain words rather than "regularization"
#: because the control has to mean something at a glance.
SMOOTHING_OFF = "Off"
SMOOTHING_AUTO = "Automatic"
SMOOTHING_MANUAL = "Manual..."


class ViewInterface(ABC):
    """Abstract base class for view implementations."""

    @abstractmethod
    def on_data_loaded(self) -> None:
        """Called when image data has been loaded."""
        pass

    @abstractmethod
    def on_points_changed(self) -> None:
        """Called when control points have changed."""
        pass

    @abstractmethod
    def on_display_update_needed(self) -> None:
        """Called when display needs to be updated."""
        pass

    @abstractmethod
    def on_error(self, message: str) -> None:
        """Called when an error occurs."""
        pass

    @abstractmethod
    def on_project_loaded(self) -> None:
        """Called when a project has been loaded."""
        pass

    @abstractmethod
    def on_request_corresponding_point(self, target: str) -> None:
        """Called when a corresponding point is needed."""
        pass

    @abstractmethod
    def on_project_reset(self) -> None:
        """Called when a new project is created."""
        pass

    @abstractmethod
    def on_show_matched_points(
        self,
        src_img: np.ndarray,
        dst_img: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> None:
        """Called to show matched points visualization."""
        pass


class ModernDistortionCorrectionView(tk.Tk, ViewInterface):
    """Modern implementation of the distortion correction GUI using MVP pattern."""

    def __init__(self):
        super().__init__()

        # Create presenter
        self.presenter = ApplicationPresenter()
        self.presenter.set_view(self)

        # UI state
        self.current_src_zoom = 100  # percentage
        self.current_dst_zoom = 100  # percentage
        self.show_points = True
        self.awaiting_corresponding_point = None

        #: In-progress marker drag, or None. See _on_canvas_press.
        self._drag: dict | None = None

        #: Directory the last file dialog used, so the next one starts there.
        self._last_directory: Path | None = None

        #: Result of the last quality check, or None. Discarded whenever the
        #: points change, because a report about a different point set is
        #: worse than no report.
        self.point_quality = None

        # Setup UI
        self._setup_window()
        self._setup_logging()
        self._create_menu()
        self._create_main_layout()
        self._create_controls()
        self._bind_events()

        apply_window_icon(self)

        logger.info("View initialized")

    def _style_call(self, style="dark"):
        """Apply the packaged Azure theme, falling back to a built-in theme.

        The colours are applied either way, so a theme that refuses to load
        degrades the look rather than preventing the application from starting.
        """
        # get_palette raises on an unknown style, which is the check that used
        # to live here.
        self.palette = get_palette(style)

        # Short aliases kept for the existing widget construction below.
        self.bg = self.palette.background
        self.fg = self.palette.foreground
        self.hl = self.palette.accent
        self.hl2 = self.palette.success

        s = ttk.Style(self)
        self.theme_name = self._load_azure_theme(s, style)

        s.configure("TFrame", background=self.bg)
        s.configure("TLabel", background=self.bg, foreground=self.fg)
        s.configure("TCheckbutton", background=self.bg, foreground=self.fg)
        s.configure(
            "TLabelframe",
            background=self.bg,
            foreground=self.fg,
            highlightcolor=self.hl,
            highlightbackground=self.hl,
        )
        s.configure(
            "TLabelframe.Label",
            background=self.bg,
            foreground=self.fg,
            highlightcolor=self.hl,
            highlightbackground=self.hl,
        )
        s.configure("TEntry", fieldbackground=self.bg, foreground=self.fg)

    def _load_azure_theme(self, style_obj: ttk.Style, style: str) -> str:
        """Source and activate the packaged Azure theme.

        Returns
        -------
        str
            The ttk theme actually in use, which is the built-in fallback when
            the packaged theme could not be loaded.

        Notes
        -----
        Tk 9 is the reason this is defensive. The theme's ``package require``
        line has to be unbounded for Tcl to satisfy it against a 9.x
        interpreter, and a stale copy of the theme (or a future incompatible
        Tk) would otherwise raise straight out of ``__init__`` and stop the
        application from opening at all.
        """
        theme = f"azure-{style}"
        try:
            self.tk.call("source", str(theme_path(style)))
            style_obj.theme_use(theme)
            logger.debug("Applied packaged theme %s", theme)
            return theme
        except (tk.TclError, FileNotFoundError, ValueError) as exc:
            available = set(style_obj.theme_names())
            # clam honours the colour configuration below far better than the
            # platform-native themes, so prefer it when present.
            for candidate in ("clam", "default"):
                if candidate in available:
                    style_obj.theme_use(candidate)
                    logger.warning(
                        "Could not load the %s theme (%s); falling back to '%s'. "
                        "The application is fully functional, only its "
                        "appearance changes.",
                        theme,
                        exc,
                        candidate,
                    )
                    return candidate

            logger.warning(
                "Could not load the %s theme (%s) and no fallback theme is "
                "available; using the Tk default.",
                theme,
                exc,
            )
            return style_obj.theme_use()

    def _setup_window(self):
        """Setup main window properties."""
        self.title("Multimodal Data Alignment Tool")
        self.geometry("1400x900")

        # Configure grid weights
        self.grid_rowconfigure(0, weight=0)  # Menu area
        self.grid_rowconfigure(1, weight=1)  # Main content
        self.grid_rowconfigure(2, weight=0)  # Status bar
        self.grid_columnconfigure(0, weight=1)

        # Set style
        self._style_call("dark")
        self.configure(background=self.bg)

    def _setup_logging(self):
        """Setup logging display."""
        # Create status bar for logging
        self.status_frame = ttk.Frame(self)
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=2)

        # Left section: Cursor position and point counts
        left_info_frame = ttk.Frame(self.status_frame)
        left_info_frame.pack(side="left", padx=5)

        self.cursor_label = ttk.Label(left_info_frame, text="Cursor: --, --", width=18)
        self.cursor_label.pack(side="left", padx=(0, 10))

        self.points_label = ttk.Label(left_info_frame, text="Points: 0 / 0", width=15)
        self.points_label.pack(side="left", padx=(0, 5))

        # Center section: Status message
        self.status_label = ttk.Label(self.status_frame, text="Ready", anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True, padx=5)

        # Right section: Progress bar
        self.progress_bar = ttk.Progressbar(
            self.status_frame,
            style="Niklas.Horizontal.TProgressbar",
            mode="indeterminate",
            length=200,
        )
        self.progress_bar.pack(side="right", padx=5)

    def _create_menu(self):
        """Create application menu."""
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)

        # File menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New project", command=self._on_new_project)
        file_menu.add_command(label="Open project", command=self._on_open_project)
        file_menu.add_command(
            label="Save project", command=self._on_save_project, accelerator="Ctrl+S"
        )
        file_menu.add_command(label="Save project as", command=self._on_save_project_as)
        file_menu.add_separator()
        file_menu.add_command(label="Open source image", command=self._on_open_source)
        file_menu.add_command(
            label="Open destination image", command=self._on_open_destination
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Load source points", command=self._on_load_source_points
        )
        file_menu.add_command(
            label="Load destination points", command=self._on_load_destination_points
        )
        file_menu.add_command(label="Save points", command=self._on_save_points)
        file_menu.add_separator()
        file_menu.add_command(
            label="Export transform", command=self._on_export_transform
        )
        file_menu.add_command(
            label="Export corrected data", command=self._on_export_corrected
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Edit menu
        self.edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Edit", menu=self.edit_menu)
        self.edit_menu.add_command(
            label="Undo", command=self._on_undo, accelerator="Ctrl+Z"
        )
        self.edit_menu.add_command(
            label="Redo", command=self._on_redo, accelerator="Ctrl+Y"
        )
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Clear points", command=self._on_clear_points)
        self.edit_menu.add_separator()
        self.edit_menu.add_command(
            label="Set resolution", command=self._on_set_resolution
        )
        self._refresh_edit_menu()

        # View menu
        view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(
            label="Hide points",
            variable=tk.BooleanVar(value=False),
            command=self._on_toggle_points,
        )
        view_menu.add_separator()
        view_menu.add_command(label="View corrected image", command=self._on_apply)
        view_menu.add_command(
            label="View corrected image stack",
            command=lambda: self._on_apply(True),
        )
        view_menu.add_separator()
        view_menu.add_command(
            label="View matched points", command=self._on_view_matched_points
        )
        view_menu.add_command(
            label="Check registration quality", command=self._on_check_quality
        )
        view_menu.add_separator()
        view_menu.add_command(
            label="Zoom in", command=self._on_zoom_in, accelerator="Ctrl++"
        )
        view_menu.add_command(
            label="Zoom out", command=self._on_zoom_out, accelerator="Ctrl+-"
        )
        view_menu.add_command(
            label="Zoom 100%", command=self._on_zoom_reset, accelerator="Ctrl+0"
        )

        # Tools menu
        tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Auto point detection", menu=tools_menu)
        tools_menu.add_command(
            label="MatchAnything",
            command=lambda: self._on_auto_detect_points("matchanything"),
        )
        tools_menu.add_command(
            label="SIFT",
            command=lambda: self._on_auto_detect_points("sift"),
        )
        tools_menu.add_separator()
        tools_menu.add_command(
            label="Set MatchAnything checkpoint...",
            command=self._on_set_checkpoint_path,
        )

    def _create_main_layout(self):
        """Create main layout with image viewers."""
        # Main container
        self.main_frame = ttk.Frame(self)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=0)
        self.main_frame.grid_columnconfigure(2, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=0)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=0)

        # Left viewer (source/distorted)
        self.left_frame = ttk.LabelFrame(self.main_frame, text="Source (Distorted)")
        self.left_frame.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=2)

        self.left_canvas = tk.Canvas(self.left_frame, bg=self.bg, cursor="crosshair")
        left_h_scrollbar = ttk.Scrollbar(
            self.left_frame,
            orient=tk.HORIZONTAL,
            command=lambda *args: self._on_scroll("source", "x", *args),
            cursor="sb_h_double_arrow",
        )
        left_v_scrollbar = ttk.Scrollbar(
            self.left_frame,
            orient=tk.VERTICAL,
            command=lambda *args: self._on_scroll("source", "y", *args),
            cursor="sb_v_double_arrow",
        )
        self.left_canvas.config(
            xscrollcommand=left_h_scrollbar.set, yscrollcommand=left_v_scrollbar.set
        )
        left_h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        left_v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.pack(fill="both", expand=True)

        # Right viewer (destination/control)
        self.right_frame = ttk.LabelFrame(self.main_frame, text="Destination (Control)")
        self.right_frame.grid(row=0, column=2, rowspan=3, sticky="nsew", padx=2)

        self.right_canvas = tk.Canvas(self.right_frame, bg=self.bg, cursor="crosshair")
        right_h_scrollbar = ttk.Scrollbar(
            self.right_frame,
            orient=tk.HORIZONTAL,
            command=lambda *args: self._on_scroll("destination", "x", *args),
            cursor="sb_h_double_arrow",
        )
        right_v_scrollbar = ttk.Scrollbar(
            self.right_frame,
            orient=tk.VERTICAL,
            command=lambda *args: self._on_scroll("destination", "y", *args),
            cursor="sb_v_double_arrow",
        )
        self.right_canvas.config(
            xscrollcommand=right_h_scrollbar.set, yscrollcommand=right_v_scrollbar.set
        )
        right_h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        right_v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_canvas.pack(fill="both", expand=True)

    def _create_controls(self):
        """Create control panel."""
        # Top controls
        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Slice selector
        ttk.Label(controls_frame, text="Slice:").pack(side="left", padx=5)
        self.slice_var = tk.IntVar(value=0)
        self.slice_spinbox = ttk.Spinbox(
            controls_frame,
            from_=0,
            to=0,
            width=7,
            textvariable=self.slice_var,
            state="disabled",
            command=self._on_slice_changed,
        )
        self.slice_spinbox.pack(side="left", padx=5)

        # Mode selectors
        ttk.Label(controls_frame, text="Mode (src):").pack(side="left", padx=(20, 5))
        self.source_mode_var = tk.StringVar(value="Intensity")
        self.source_mode_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.source_mode_var,
            state="readonly",
            width=12,
        )
        self.source_mode_combo.pack(side="left", padx=5)
        self.source_mode_combo.bind(
            "<<ComboboxSelected>>", self._on_source_mode_changed
        )

        ttk.Label(controls_frame, text="Mode (dst):").pack(side="left", padx=(20, 5))
        self.dest_mode_var = tk.StringVar(value="Intensity")
        self.dest_mode_combo = ttk.Combobox(
            controls_frame, textvariable=self.dest_mode_var, state="readonly", width=12
        )
        self.dest_mode_combo.pack(side="left", padx=5)
        self.dest_mode_combo.bind("<<ComboboxSelected>>", self._on_dest_mode_changed)

        # CLAHE toggles
        self.clahe_source_var = tk.BooleanVar(value=False)
        self.clahe_source_check = ttk.Checkbutton(
            controls_frame,
            text="CLAHE (src)",
            variable=self.clahe_source_var,
            command=lambda: self.presenter.toggle_clahe("source"),
        )
        self.clahe_source_check.pack(side="left", padx=(20, 5))

        self.clahe_dest_var = tk.BooleanVar(value=False)
        self.clahe_dest_check = ttk.Checkbutton(
            controls_frame,
            text="CLAHE (dst)",
            variable=self.clahe_dest_var,
            command=lambda: self.presenter.toggle_clahe("destination"),
        )
        self.clahe_dest_check.pack(side="left", padx=5)

        # Zoom control
        ttk.Label(controls_frame, text="Zoom (src):").pack(side="left", padx=(20, 5))
        self.zoom_src_var = tk.StringVar(value="100%")
        self.zoom_src_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.zoom_src_var,
            values=[
                "5%",
                "10%",
                "25%",
                "50%",
                "75%",
                "100%",
                "150%",
                "200%",
                "300%",
                "500%",
                "800%",
                "1000%",
                "1500%",
            ],
            state="readonly",
            width=8,
        )
        self.zoom_src_combo.pack(side="left", padx=5)
        self.zoom_src_combo.bind("<<ComboboxSelected>>", self._on_zoom_changed)
        ttk.Label(controls_frame, text="Zoom (dst):").pack(side="left", padx=(20, 5))
        self.zoom_dst_var = tk.StringVar(value="100%")
        self.zoom_dst_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.zoom_dst_var,
            values=[
                "5%",
                "10%",
                "25%",
                "50%",
                "75%",
                "100%",
                "150%",
                "200%",
                "300%",
                "500%",
                "800%",
                "1000%",
                "1500%",
            ],
            state="readonly",
            width=8,
        )
        self.zoom_dst_combo.pack(side="left", padx=5)
        self.zoom_dst_combo.bind("<<ComboboxSelected>>", self._on_zoom_changed)

        # Link the two viewers. Comparing the same feature side by side means
        # keeping both panels at the same zoom and scroll position, which
        # otherwise has to be done by hand after every adjustment.
        self.link_views_var = tk.BooleanVar(value=False)
        self.link_views_check = ttk.Checkbutton(
            controls_frame,
            text="Link views",
            variable=self.link_views_var,
            command=self._on_link_views_changed,
        )
        self.link_views_check.pack(side="left", padx=(20, 5))

        # Match resolutions control
        self.match_resolutions_var = tk.BooleanVar(value=False)
        self.match_resolutions_check = ttk.Checkbutton(
            controls_frame,
            text="Match Res",
            variable=self.match_resolutions_var,
            command=self.presenter.toggle_match_resolutions,
        )
        self.match_resolutions_check.pack(side="left", padx=(20, 5))

        # Smoothing. Off by default, because it changes results and should be
        # asked for; "Automatic" is the one worth reaching for, since the
        # manual number has no units anyone has an intuition about.
        ttk.Label(controls_frame, text="Smoothing:").pack(side="left", padx=(20, 5))
        self.smoothing_var = tk.StringVar(value=SMOOTHING_OFF)
        self.smoothing_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.smoothing_var,
            values=[SMOOTHING_OFF, SMOOTHING_AUTO, SMOOTHING_MANUAL],
            state="readonly",
            width=11,
        )
        self.smoothing_combo.pack(side="left", padx=5)
        self.smoothing_combo.bind("<<ComboboxSelected>>", self._on_smoothing_changed)

    def _on_smoothing_changed(self, event=None):
        """Apply the smoothing choice, asking for a number if it needs one."""
        choice = self.smoothing_var.get()

        if choice == SMOOTHING_OFF:
            self.presenter.set_regularization(0.0)
            self.set_status("Smoothing off: the fit passes through every point")
            return

        if choice == SMOOTHING_AUTO:
            self.presenter.set_regularization("auto")
            self.set_status(
                "Smoothing chosen automatically by cross-validation when estimating"
            )
            return

        current = self.presenter.regularization
        strength = simpledialog.askfloat(
            "Smoothing strength",
            "Strength (0 = pass through every point).\n\n"
            "Normalised, so the same value means roughly the same thing at any "
            "image size. Useful values are typically between 0.001 and 1.",
            initialvalue=float(current) if isinstance(current, (int, float)) else 0.01,
            minvalue=0.0,
            parent=self,
        )

        if strength is None:
            # Cancelled: put the selector back rather than leaving it lying
            # about a setting that was not applied.
            self._sync_smoothing_selector()
            return

        self.presenter.set_regularization(strength)
        self.set_status(f"Smoothing strength set to {strength:g}")

    def _sync_smoothing_selector(self):
        """Point the selector at whatever the presenter actually has."""
        current = self.presenter.regularization
        if isinstance(current, str):
            self.smoothing_var.set(SMOOTHING_AUTO)
        elif current:
            self.smoothing_var.set(SMOOTHING_MANUAL)
        else:
            self.smoothing_var.set(SMOOTHING_OFF)

    def _bind_events(self):
        """Bind keyboard and mouse events."""
        # Keyboard shortcuts
        self.bind("<Control-s>", lambda e: self._on_save_project())
        self.bind("<Control-z>", lambda e: self._on_undo())
        self.bind("<Control-y>", lambda e: self._on_redo())
        self.bind("<Control-equal>", lambda e: self._on_zoom_in())
        self.bind("<Control-minus>", lambda e: self._on_zoom_out())
        self.bind("<Control-0>", lambda e: self._on_zoom_reset())

        # Mouse events for canvases. Placing a point happens on release, not
        # press, so that a press landing on an existing marker can turn into a
        # drag instead. See _on_canvas_press.
        self.left_canvas.bind(
            "<Button-1>", lambda e: self._on_canvas_press(e, "source")
        )
        self.right_canvas.bind(
            "<Button-1>", lambda e: self._on_canvas_press(e, "destination")
        )
        self.left_canvas.bind(
            "<B1-Motion>", lambda e: self._on_canvas_drag(e, "source")
        )
        self.right_canvas.bind(
            "<B1-Motion>", lambda e: self._on_canvas_drag(e, "destination")
        )
        self.left_canvas.bind(
            "<ButtonRelease-1>", lambda e: self._on_canvas_release(e, "source")
        )
        self.right_canvas.bind(
            "<ButtonRelease-1>", lambda e: self._on_canvas_release(e, "destination")
        )

        # Mouse motion events for cursor tracking
        self.left_canvas.bind("<Motion>", lambda e: self._on_canvas_motion(e, "source"))
        self.right_canvas.bind(
            "<Motion>", lambda e: self._on_canvas_motion(e, "destination")
        )
        self.left_canvas.bind("<Leave>", lambda _: self._on_canvas_leave())
        self.right_canvas.bind("<Leave>", lambda _: self._on_canvas_leave())
        if os.name == "posix":
            remove_string = "<Button 2>"
            scroll_multiplier = 1
        else:
            remove_string = "<Button 3>"
            scroll_multiplier = 120

        self.left_canvas.bind(
            remove_string, lambda e: self._on_canvas_right_click(e, "source")
        )
        self.right_canvas.bind(
            remove_string, lambda e: self._on_canvas_right_click(e, "destination")
        )
        # Routed through _on_scroll so a linked partner panel follows the
        # wheel as well as the scrollbars.
        for canvas_type, axis, sequence in (
            ("source", "y", "<MouseWheel>"),
            ("source", "x", "<Shift-MouseWheel>"),
            ("destination", "y", "<MouseWheel>"),
            ("destination", "x", "<Shift-MouseWheel>"),
        ):
            canvas = self.left_canvas if canvas_type == "source" else self.right_canvas
            canvas.bind(
                sequence,
                lambda event, t=canvas_type, a=axis: self._on_scroll(
                    t, a, "scroll", int(-1 * (event.delta / scroll_multiplier)), "units"
                ),
            )

    # ========== Event Handlers ==========

    def _on_new_project(self):
        """Handle creating a new project."""
        if self.presenter.has_unsaved_changes():
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save the current project?",
            )
            if response is None:
                return
            elif response:
                self._on_save_project()

        self.presenter.new_project()
        self.set_status("New project created")
        self.title("Multimodal Data Alignment Tool - New Project")

    def _on_open_source(self):
        """Handle opening source image."""
        file_paths = self._ask_open_many(
            title="Open Source Image",
            filetypes=[
                ("All Supported", "*.ang *.h5 *.dream3d *.tif *.tiff *.png *.jpg"),
                ("EBSD Files", "*.ang *.h5 *.dream3d"),
                ("Image Files", "*.tif *.tiff *.png *.jpg"),
                ("All Files", "*.*"),
            ],
        )

        if file_paths:
            try:
                self.show_progress(True)
                if len(file_paths) == 1:
                    path = Path(file_paths[0])
                    modality_name = None

                    # Check if this is a single image file that needs a modality name
                    if path.suffix.lower() in [
                        ".tif",
                        ".tiff",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ]:
                        # Ask for modality name
                        modality_name = self._get_modality_name_dialog(path.name)
                        if modality_name is None:
                            return  # User cancelled

                    self.show_progress(True)
                    self.set_status(f"Loading source image: {path.name}")
                    if self.presenter.load_source_image(
                        path, modality_name=modality_name
                    ):
                        self.set_status("Source image loaded successfully")

                else:
                    first_path = Path(file_paths[0])
                    # Check if this is a single image file that needs a modality name
                    if first_path.suffix.lower() in [
                        ".tif",
                        ".tiff",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ]:
                        # Ask for modality name
                        modality_name = self._get_modality_name_dialog(first_path.name)
                        if modality_name is None:
                            return  # User cancelled

                    self.set_status(f"Loading {len(file_paths)} source images")
                    if self.presenter.load_source_image(
                        file_paths, modality_name=modality_name
                    ):
                        self.set_status("Source image stack loaded successfully")
            finally:
                self.show_progress(False)

    def _on_open_destination(self):
        """Handle opening destination image."""
        file_paths = self._ask_open_many(
            title="Open Destination Image",
            filetypes=[
                ("Image Files", "*.tif *.tiff *.png *.jpg *.dream3d"),
                ("All Files", "*.*"),
            ],
        )

        if file_paths:
            try:
                self.show_progress(True)
                if len(file_paths) == 1:
                    path = Path(file_paths[0])
                    modality_name = None

                    # Check if this is a single image file that needs a modality name
                    if path.suffix.lower() in [
                        ".tif",
                        ".tiff",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ]:
                        # Ask for modality name
                        modality_name = self._get_modality_name_dialog(path.name)
                        if modality_name is None:
                            return  # User cancelled

                    self.set_status(f"Loading destination image: {path.name}")
                    if self.presenter.load_destination_image(
                        path, modality_name=modality_name
                    ):
                        self.set_status("Destination image loaded successfully")

                else:
                    first_path = Path(file_paths[0])
                    # Check if this is a single image file that needs a modality name
                    if first_path.suffix.lower() in [
                        ".tif",
                        ".tiff",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ]:
                        # Ask for modality name
                        modality_name = self._get_modality_name_dialog(first_path.name)
                        if modality_name is None:
                            return  # User cancelled

                    self.set_status(f"Loading {len(file_paths)} destination images")
                    if self.presenter.load_destination_image(
                        file_paths, modality_name=modality_name
                    ):
                        self.set_status("Destination image stack loaded successfully")
            finally:
                self.show_progress(False)

    def _on_load_source_points(self):
        """Handle loading source control points."""
        src_path = self._ask_open(
            title="Load Source Points",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )

        if src_path and self.presenter.load_source_points(Path(src_path)):
            self.set_status("Source points loaded successfully")

    def _on_load_destination_points(self):
        """Handle loading destination control points."""
        dst_path = self._ask_open(
            title="Load Destination Points",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )

        if dst_path and self.presenter.load_destination_points(Path(dst_path)):
            self.set_status("Destination points loaded successfully")

    def _on_save_points(self):
        """Handle saving control points."""
        # Uses the default paths set in presenter
        self.presenter._save_points()
        self.set_status("Points saved")

    def _on_open_project(self):
        """Handle opening a project."""
        self.show_progress(True)
        self._on_new_project()
        file_path = self._ask_open(
            title="Open Project",
            filetypes=[("Project Files", "*.json"), ("All Files", "*.*")],
        )

        if file_path:
            self.set_status("Loading project...")

            if self.presenter.load_project(Path(file_path)):
                self.set_status("Project loaded successfully")
                self.title(f"Multimodal Data Alignment Tool - {Path(file_path).name}")

        self.show_progress(False)

    def _on_save_project(self):
        """Handle saving current project."""
        if self.presenter.project_manager.project_path:
            if self.presenter.save_project(self.presenter.project_manager.project_path):
                self.set_status("Project saved")
        else:
            self._on_save_project_as()

    def _on_save_project_as(self):
        """Handle saving project with new name."""
        file_path = self._ask_save(
            title="Save Project As",
            defaultextension=".json",
            filetypes=[("Project Files", "*.json"), ("All Files", "*.*")],
        )

        if file_path and self.presenter.save_project(Path(file_path)):
            self.set_status(f"Project saved as {Path(file_path).name}")
            self.title(f"Multimodal Data Alignment Tool - {Path(file_path).name}")

    def _on_export_transform(self):
        """Handle exporting transformation."""
        # Get transform type
        transform_type = self._get_transform_type_dialog()
        if not transform_type:
            return

        # Warn before the file dialog, not after: there is no point choosing a
        # destination for a transform that will not estimate.
        if not self._confirm_point_quality(transform_type):
            return

        file_path = self._ask_save(
            title="Export Transform",
            defaultextension=".npy",
            filetypes=[
                ("NumPy Array", "*.npy"),
                ("CSV File", "*.csv"),
                ("Text File", "*.txt"),
                ("All Files", "*.*"),
            ],
        )

        if file_path and self.presenter.export_transform(
            Path(file_path), transform_type
        ):
            self.set_status("Transform exported successfully")

    def _on_export_corrected(self):
        """Handle exporting corrected image."""
        # This would need implementation for full export functionality
        self.set_status("Exporting corrected image")
        self.show_progress(True)

        transform_type = self._get_transform_type_dialog()
        if transform_type is None:
            self.show_progress(False)
            return

        if not self._confirm_point_quality(transform_type):
            self.show_progress(False)
            return

        data_format = self._get_export_format_dialog()
        if data_format is None:
            self.show_progress(False)
            return

        if data_format == DataFormat.DREAM3D:
            crop_mode = CropMode.SOURCE
        else:
            crop_mode = self._get_crop_mode_dialog()
            if crop_mode is None:
                self.show_progress(False)
                return

        if data_format == DataFormat.IMAGE:
            ftypes = [
                ("TIFF Image", "*.tif *.tiff"),
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("All Files", "*.*"),
            ]
        elif data_format == DataFormat.RAW_IMAGE:
            ftypes = [
                ("TIFF Image", "*.tif *.tiff"),
                ("All Files", "*.*"),
            ]
        elif data_format == DataFormat.ANG:
            ftypes = [
                ("ANG File", "*.ang"),
                ("All Files", "*.*"),
            ]
        elif data_format == DataFormat.DREAM3D:
            ftypes = [
                ("Dream3D File", "*.dream3d"),
                ("All Files", "*.*"),
            ]

        path = self._ask_save(
            title="Export Corrected Data",
            defaultextension=ftypes[0][1].split(" ")[0].replace("*", ""),
            filetypes=ftypes,
        )

        if not path:
            self.show_progress(False)
            return

        self.set_status(
            f"Exporting {transform_type.value} corrected image as {data_format.value} cropped to {crop_mode.value}..."
        )
        self.presenter.export_data(Path(path), data_format, crop_mode, transform_type)
        self.show_progress(False)
        self.set_status(f"Data exported to '{path}'")

    #: How close, in screen pixels, a press has to be to a marker to grab it.
    GRAB_RADIUS_PIXELS = 8

    #: How far a press has to travel before it counts as a drag rather than a
    #: click. Without this, the shake in an ordinary click would nudge points.
    DRAG_THRESHOLD_PIXELS = 3

    def _canvas_for(self, canvas_type):
        """The canvas and zoom scale for one side."""
        if canvas_type == "source":
            return self.left_canvas, self.current_src_zoom / 100.0
        return self.right_canvas, self.current_dst_zoom / 100.0

    def _event_to_image(self, event, canvas_type):
        """Convert an event's widget coordinates to image coordinates.

        ``canvasx``/``canvasy`` do the whole job: they account for the scroll
        offset and for the widget's highlight border, which shifts the drawing
        origin by a pixel. Converting once and dividing once matters -- the
        older form divided the event and the offset separately and truncated
        each, which put a click up to a whole image pixel off its target at
        high zoom, exactly where the precision is wanted.
        """
        canvas, scale = self._canvas_for(canvas_type)
        x = int(canvas.canvasx(event.x) / scale)
        y = int(canvas.canvasy(event.y) / scale)
        return x, y

    def _on_canvas_press(self, event, canvas_type):
        """Begin either a point placement or a marker drag.

        Nothing is committed here. A press that lands on an existing marker
        arms a drag; anything else arms a placement. Which one actually happens
        is decided in :meth:`_on_canvas_release`, once it is known whether the
        mouse moved.
        """
        if canvas_type == "source" and not self.presenter.source_image:
            self.set_status("No source image loaded")
            return
        if canvas_type == "destination" and not self.presenter.destination_image:
            self.set_status("No destination image loaded")
            return

        _, scale = self._canvas_for(canvas_type)
        x, y = self._event_to_image(event, canvas_type)

        # The grab radius is in screen pixels, so it stays the same physical
        # size on screen no matter how far the image is zoomed.
        index = None
        if self.show_points:
            index = self.presenter.find_point_near(
                canvas_type, x, y, self.GRAB_RADIUS_PIXELS / max(scale, 1e-6)
            )

        self._drag = {
            "canvas_type": canvas_type,
            "index": index,
            "start": (event.x, event.y),
            "moved": False,
            "recorded": False,
            "position": (x, y),
        }

        if index is not None:
            self.set_status(f"Drag to move point {index}, or release to leave it")

    def _on_canvas_drag(self, event, canvas_type):
        """Track a marker being dragged, updating the display as it goes."""
        drag = self._drag
        if drag is None or drag["canvas_type"] != canvas_type:
            return

        start_x, start_y = drag["start"]
        if not drag["moved"]:
            travelled = max(abs(event.x - start_x), abs(event.y - start_y))
            if travelled < self.DRAG_THRESHOLD_PIXELS:
                return
            drag["moved"] = True

        x, y = self._event_to_image(event, canvas_type)
        drag["position"] = (x, y)

        if drag["index"] is None:
            # Dragging from empty space is a no-op rather than a stray point.
            return

        # Move as the mouse moves so the marker follows the cursor. Only the
        # first step of a gesture is recorded, so the whole drag undoes in one.
        moved = self.presenter.move_point(
            canvas_type, drag["index"], x, y, transient=drag["recorded"]
        )
        if moved:
            drag["recorded"] = True
            self.set_status(f"Point {drag['index']} -> ({x}, {y})")

    def _on_canvas_release(self, event, canvas_type):
        """Finish a drag, or place a point if the press never became one."""
        drag = self._drag
        self._drag = None

        if drag is None or drag["canvas_type"] != canvas_type:
            return

        if drag["moved"]:
            if drag["index"] is not None and drag["recorded"]:
                x, y = drag["position"]
                self.presenter.commit_point_move()
                self.set_status(f"Moved point {drag['index']} to ({x}, {y})")
            return

        # A press that did not move is an ordinary click. Grabbing a marker
        # and releasing without moving deliberately does nothing, so that a
        # misjudged click near a point cannot silently place another one.
        if drag["index"] is not None:
            self.set_status(f"Point {drag['index']} unchanged")
            return

        self._place_point(event, canvas_type)

    def _place_point(self, event, canvas_type):
        """Handle canvas click for point placement."""
        x, y = self._event_to_image(event, canvas_type)

        # Validate point is within image bounds
        if not self.presenter.is_point_in_bounds(canvas_type, x, y):
            self.set_status(f"Point ({x}, {y}) is outside image bounds - click ignored")
            return

        if self.awaiting_corresponding_point == canvas_type:
            # Add corresponding point
            self.presenter.add_point(canvas_type, x, y)
            self.awaiting_corresponding_point = None
            self.set_status("Point pair added")
        else:
            # Start new point pair
            self.presenter.add_point(canvas_type, x, y)
            self.awaiting_corresponding_point = (
                "destination" if canvas_type == "source" else "source"
            )
            if self.awaiting_corresponding_point:
                self.set_status(
                    f"Click on {self.awaiting_corresponding_point} image to add corresponding point"
                )

    def _on_canvas_right_click(self, event, canvas_type):
        """Handle right-click for point removal."""
        # Find nearest point and remove it

        if canvas_type == "source":
            canvas = self.left_canvas
        else:
            canvas = self.right_canvas

        closest = canvas.find_closest(canvas.canvasx(event.x), canvas.canvasy(event.y))
        tag = canvas.itemcget(closest[0], "tags")
        tag = (
            tag.replace("current", "")
            .replace("text", "")
            .replace("bbox", "")
            .replace("point_", "")
            .strip()
        )
        if tag == "":
            return
        self.presenter.remove_point(int(tag))
        logger.debug("Removed point with index %s", int(tag))
        self.set_status(f"Removed point pair {int(tag)}")

    def _on_canvas_motion(self, event, canvas_type):
        """Handle mouse motion for cursor position tracking."""
        if canvas_type == "source":
            image = self.presenter.source_image
        else:
            image = self.presenter.destination_image

        # Check if image is loaded
        if image is None:
            return

        # Same conversion the click handlers use, so the readout and the point
        # that gets placed can never disagree.
        x, y = self._event_to_image(event, canvas_type)
        self.cursor_label.config(text=f"Cursor: {x}, {y}")

    def _on_canvas_leave(self):
        """Handle mouse leaving canvas."""
        self.cursor_label.config(text="Cursor: --, --")

    # ========== File dialogs ==========
    #
    # Every dialog goes through these so it opens where the last one left off.
    # Tk otherwise starts each one in the process working directory, which
    # means re-navigating to the data from scratch for the source image, the
    # destination image, the points, and the export.

    def _remember_directory(self, path):
        """Record the folder a chosen file lives in."""
        if not path:
            return
        try:
            self._last_directory = Path(path).expanduser().resolve().parent
        except (OSError, ValueError):  # pragma: no cover - odd platform paths
            logger.debug("Could not remember the directory for %s", path, exc_info=True)

    def _initial_directory(self, override=None):
        """Directory a dialog should open in, or None to let Tk decide."""
        candidate = override if override is not None else self._last_directory
        if candidate is None:
            return None
        candidate = Path(candidate)
        return str(candidate) if candidate.is_dir() else None

    def _ask_open(self, initialdir=None, **kwargs):
        """askopenfilename, starting from the last-used folder."""
        path = filedialog.askopenfilename(
            initialdir=self._initial_directory(initialdir), **kwargs
        )
        self._remember_directory(path)
        return path

    def _ask_open_many(self, initialdir=None, **kwargs):
        """askopenfilenames, starting from the last-used folder."""
        paths = filedialog.askopenfilenames(
            initialdir=self._initial_directory(initialdir), **kwargs
        )
        if paths:
            self._remember_directory(paths[0])
        return paths

    def _ask_save(self, initialdir=None, **kwargs):
        """asksaveasfilename, starting from the last-used folder."""
        path = filedialog.asksaveasfilename(
            initialdir=self._initial_directory(initialdir), **kwargs
        )
        self._remember_directory(path)
        return path

    def _refresh_edit_menu(self):
        """Grey out Undo and Redo when there is nothing to undo or redo.

        Both used to be permanently enabled and silently did nothing at the
        ends of the history, which reads as the application ignoring you.
        """
        try:
            self.edit_menu.entryconfig(
                "Undo", state="normal" if self.presenter.can_undo() else "disabled"
            )
            self.edit_menu.entryconfig(
                "Redo", state="normal" if self.presenter.can_redo() else "disabled"
            )
        except tk.TclError:  # pragma: no cover - menu torn down
            logger.debug("Could not update the Edit menu", exc_info=True)

    def _update_point_count(self):
        """Update the point count display for current slice."""
        src_points, dst_points = self.presenter.get_points()
        src_count = len(src_points)
        dst_count = len(dst_points)
        self.points_label.config(text=f"Points: {src_count} / {dst_count}")

    def _on_slice_changed(self):
        """Handle slice change."""
        self.presenter.set_current_slice(self.slice_var.get())
        self._update_point_count()

    def _on_source_mode_changed(self, event=None):
        """Handle source mode change."""
        self.presenter.set_source_mode(self.source_mode_var.get())

    def _on_dest_mode_changed(self, event=None):
        """Handle destination mode change."""
        self.presenter.set_destination_mode(self.dest_mode_var.get())

    def _on_zoom_changed(self, event=None):
        """Handle zoom change."""
        # With the viewers linked, whichever selector was used drives both.
        if self.link_views_var.get() and event is not None:
            widget = getattr(event, "widget", None)
            if widget is self.zoom_src_combo:
                self.zoom_dst_var.set(self.zoom_src_var.get())
            elif widget is self.zoom_dst_combo:
                self.zoom_src_var.set(self.zoom_dst_var.get())

        zoom_str_src = self.zoom_src_var.get().rstrip("%")
        self.current_src_zoom = int(zoom_str_src)
        zoom_str_dst = self.zoom_dst_var.get().rstrip("%")
        self.current_dst_zoom = int(zoom_str_dst)
        self.update_display()

    def _on_link_views_changed(self):
        """Bring the two viewers together when linking is switched on."""
        if not self.link_views_var.get():
            self.set_status("Viewers unlinked")
            return

        # The source panel wins, so switching the box on has a predictable
        # result rather than depending on which panel was touched last.
        self.zoom_dst_var.set(self.zoom_src_var.get())
        self.current_dst_zoom = self.current_src_zoom
        self.update_display()
        self._sync_view("source", "x")
        self._sync_view("source", "y")
        self.set_status("Viewers linked: zoom and scrolling stay in step")

    def _on_scroll(self, canvas_type, axis, *args):
        """Scroll one canvas from its scrollbar, carrying the other along."""
        canvas, _ = self._canvas_for(canvas_type)
        view = canvas.xview if axis == "x" else canvas.yview
        view(*args)
        self._sync_view(canvas_type, axis)

    def _sync_view(self, canvas_type, axis):
        """Match the other canvas to this one's scroll position.

        Does nothing unless the viewers are linked. Fractions are used rather
        than pixels so panels showing differently sized images still line up.
        """
        if not self.link_views_var.get():
            return

        if canvas_type == "source":
            driver, follower = self.left_canvas, self.right_canvas
        else:
            driver, follower = self.right_canvas, self.left_canvas

        if axis == "x":
            follower.xview_moveto(driver.xview()[0])
        else:
            follower.yview_moveto(driver.yview()[0])

    def _on_zoom_in(self):
        """Zoom in."""
        zoom_levels = [5, 10, 25, 50, 75, 100, 150, 200, 300, 500]
        current_src_idx = (
            zoom_levels.index(self.current_src_zoom)
            if self.current_src_zoom in zoom_levels
            else 3
        )
        current_dst_idx = (
            zoom_levels.index(self.current_dst_zoom)
            if self.current_dst_zoom in zoom_levels
            else 3
        )
        if current_src_idx < len(zoom_levels) - 1:
            self.current_src_zoom = zoom_levels[current_src_idx + 1]
            self.zoom_src_var.set(f"{self.current_src_zoom}%")
            self.update_display()
        if current_dst_idx < len(zoom_levels) - 1:
            self.current_dst_zoom = zoom_levels[current_dst_idx + 1]
            self.zoom_dst_var.set(f"{self.current_dst_zoom}%")
            self.update_display()

    def _on_zoom_out(self):
        """Zoom out."""
        zoom_levels = [5, 10, 25, 50, 75, 100, 150, 200, 300, 500]
        current_src_idx = (
            zoom_levels.index(self.current_src_zoom)
            if self.current_src_zoom in zoom_levels
            else 3
        )
        current_dst_idx = (
            zoom_levels.index(self.current_dst_zoom)
            if self.current_dst_zoom in zoom_levels
            else 3
        )
        if current_src_idx > 0:
            self.current_src_zoom = zoom_levels[current_src_idx - 1]
            self.zoom_src_var.set(f"{self.current_src_zoom}%")
            self.update_display()
        if current_dst_idx > 0:
            self.current_dst_zoom = zoom_levels[current_dst_idx - 1]
            self.zoom_dst_var.set(f"{self.current_dst_zoom}%")
            self.update_display()

    def _on_zoom_reset(self):
        """Reset zoom to 100%."""
        self.current_src_zoom = 100
        self.current_dst_zoom = 100
        self.zoom_src_var.set("100%")
        self.zoom_dst_var.set("100%")
        self.update_display()

    def _on_undo(self):
        """Undo last action."""
        self.presenter.undo()
        self.set_status("Undone")

    def _on_redo(self):
        """Redo last undone action."""
        self.presenter.redo()
        self.set_status("Redone")

    def _on_clear_points(self):
        """Clear points on current slice."""
        if self.presenter.source_image.shape[0] > 1:
            response = self._get_point_clear_dialog()
            if response is None:
                return
            elif response == "image":
                self.presenter.clear_points(slice_only=True)
                self.set_status("Points cleared for current image")
            elif response == "stack":
                self.presenter.clear_points(slice_only=False)
                self.set_status("Points cleared for entire stack")
        else:
            if messagebox.askyesno("Clear Points", "Clear all points?"):
                self.presenter.clear_points(slice_only=True)
                self.set_status("Points cleared for current image")

    def _on_toggle_points(self):
        """Toggle point visibility."""
        self.show_points = not self.show_points
        self.update_display()

    def _on_set_resolution(self):
        """Set image resolution."""
        self.show_progress(True)
        src_res, dst_res = self._get_image_resolutions_dialog()
        if src_res is None or dst_res is None:
            self.show_progress(False)
            return
        elif src_res and dst_res:
            # Update image resolutions
            self.presenter.set_image_resolutions(src_res, dst_res)
        self.show_progress(False)

    def _confirm_point_quality(self, transform_type) -> bool:
        """Check the control points, and let the user decide about warnings.

        Estimation over bad points either raises out of numpy several seconds
        later or returns a mangled image, neither of which says what to fix.
        Checking first means the problem is described while the points are
        still on screen.

        Returns
        -------
        bool
            True to go ahead. False when the points cannot work, or when the
            user chose not to continue past a warning.
        """
        issues = self.presenter.check_points(transform_type)
        if not issues:
            return True

        errors = [issue for issue in issues if issue.is_error]
        if errors:
            messagebox.showerror(
                "Control points cannot be used",
                "The transform cannot be estimated:\n\n"
                + validation.format_issues(errors),
            )
            self.set_status("Transform not estimated: check the control points")
            return False

        return bool(
            messagebox.askokcancel(
                "Check the control points",
                validation.format_issues(issues) + "\n\nEstimate the transform anyway?",
                icon=messagebox.WARNING,
            )
        )

    def _on_apply(self, is_3d=False):
        """Apply transformation to current slice."""
        self.show_progress(True)
        transform_type = self._get_transform_type_dialog()
        if transform_type and not self._confirm_point_quality(transform_type):
            self.show_progress(False)
            return
        if transform_type:
            crop_mode = self._get_crop_mode_dialog()
            if crop_mode is not None:
                self.set_status(f"Generating {transform_type.value} preview...")
                if is_3d:
                    if self.presenter.source_image.shape[0] == 1:
                        raise ValueError(
                            "3D transformation can only be applied to image stacks"
                        )
                    self.presenter.apply_transform_3d(
                        transform_type, crop_mode, preview=True
                    )
                else:
                    self.presenter.apply_transform(
                        transform_type, crop_mode, preview=True
                    )
                self.set_status("Transformation preview completed")
        self.show_progress(False)

    def _on_auto_detect_points(self, method: str):
        """Handle automatic point detection."""
        # Show parameter dialog
        self.show_progress(True)
        params = self._get_auto_detect_params_dialog(method)
        if params is None:
            self.show_progress(False)
            return  # User cancelled

        self.set_status(f"Detecting points using {method}...")
        original_n_points = len(self.presenter.get_points()[0])
        success = self.presenter.auto_detect_points(method, **params)
        new_n_points = len(self.presenter.get_points()[0])
        if success:
            self.set_status(
                f"Points detected using {method}: {new_n_points - original_n_points} new points"
            )
        else:
            self.set_status(f"Point detection using {method} failed")
        self.show_progress(False)

    def _on_set_checkpoint_path(self):
        """Handle setting the MatchAnything checkpoint path."""
        self.show_progress(True)
        current_path = self.presenter.get_checkpoint_path()
        initial_dir = Path(current_path).parent if current_path else None

        file_path = self._ask_open(
            title="Select MatchAnything Checkpoint",
            initialdir=initial_dir,
            filetypes=[
                ("Checkpoint Files", "*.pth *.pt *.ckpt"),
                ("All Files", "*.*"),
            ],
        )

        if file_path:
            self.presenter.set_checkpoint_path(Path(file_path))
            self.set_status(f"Checkpoint path set to: {Path(file_path).name}")
        self.show_progress(False)

    def _on_view_matched_points(self):
        """Handle viewing matched points visualization."""
        if not self.presenter.source_image or not self.presenter.destination_image:
            self.on_error("Both source and destination images must be loaded")
            return

        src_points, dst_points = self.presenter.get_points()
        if src_points.size == 0 or dst_points.size == 0:
            self.on_error("No control points defined to visualize")
            return
        elif src_points.shape[0] != dst_points.shape[0]:
            self.on_error("Source and destination points counts do not match")
            return

        self.show_progress(True)
        self.set_status("Generating matched points visualization...")
        self.presenter.show_matched_points()
        self.set_status("Matched points visualization opened")
        self.show_progress(False)

    def _on_check_quality(self):
        """Fit a transform and report how good the correspondences look.

        Separate from placing points because the useful measure is expensive:
        it refits the spline once per control point. See
        :func:`tpsreg.metrics.leave_one_out_residuals`.
        """
        if not self.presenter.source_image or not self.presenter.destination_image:
            self.on_error("Both source and destination images must be loaded")
            return

        transform_type = self._get_transform_type_dialog()
        if transform_type is None:
            return
        if not self._confirm_point_quality(transform_type):
            return

        self.show_progress(True)
        self.set_status("Checking registration quality...")
        quality = self.presenter.assess_transform(transform_type)
        self.show_progress(False)

        if quality is None:
            self.set_status("Could not assess the transform")
            return

        # Colouring the markers is what makes a numbered outlier findable on a
        # canvas holding several dozen points.
        self.point_quality = quality
        self.update_display()

        self.set_status(quality.summary())
        messagebox.showinfo("Registration quality", self._quality_report(quality))

    @staticmethod
    def _quality_report(quality) -> str:
        """The assessment as readable prose."""
        lines = []

        if quality.leave_one_out.size:
            lines.append(
                "Leave-one-out residuals -- how far each point falls from a "
                "fit that excludes it. A spline passes exactly through its "
                "own control points, so this is what a bad correspondence "
                "actually shows up in.\n"
                f"    median {quality.median_residual:.2f} px"
            )
            worst = quality.worst_point
            if worst is not None:
                lines[-1] += (
                    f", worst point {worst} at {quality.leave_one_out[worst]:.2f} px"
                )

            flagged = np.flatnonzero(quality.outliers)
            if flagged.size:
                lines.append(
                    f"Points worth re-checking: {', '.join(map(str, flagged))}\n"
                    "    Shown outlined in red on the canvas."
                )
            else:
                lines.append("No point disagrees with the others.")

        if quality.has_folds:
            lines.append(
                f"The mapping folds over itself across {quality.folded_fraction:.1%} "
                "of the image (smallest Jacobian "
                f"{quality.min_jacobian:.2f}).\n"
                "    Folded regions come out mirrored. This usually means two "
                "correspondences cross over each other."
            )
        else:
            lines.append("The mapping does not fold anywhere.")

        if quality.coverage is not None:
            lines.append(
                f"The points enclose {quality.coverage:.0%} of the image. "
                "Everything outside that is extrapolated."
            )

        lines.append(f"Bending energy: {quality.bending_energy:.4g}")
        return "\n\n".join(lines)

    # ========== ViewInterface Implementation ==========

    def on_data_loaded(self):
        """Called when image data has been loaded."""
        # Update UI elements based on loaded data
        if self.presenter.source_image:
            modes = self.presenter.get_source_modalities()
            self.source_mode_combo["values"] = modes
            if modes:
                # Keep current mode if it still exists, otherwise use presenter's current mode
                current_mode = self.presenter.current_source_mode
                if current_mode in modes:
                    self.source_mode_var.set(current_mode)
                else:
                    # Use the first mode if current doesn't exist
                    self.source_mode_var.set(modes[0])
                    self.presenter.current_source_mode = modes[0]

        if self.presenter.destination_image:
            modes = self.presenter.get_destination_modalities()
            self.dest_mode_combo["values"] = modes
            if modes:
                # Keep current mode if it still exists, otherwise use presenter's current mode
                current_mode = self.presenter.current_dest_mode
                if current_mode in modes:
                    self.dest_mode_var.set(current_mode)
                else:
                    # Use the first mode if current doesn't exist
                    self.dest_mode_var.set(modes[0])
                    self.presenter.current_dest_mode = modes[0]

        # Update slice control
        min_slice, max_slice = self.presenter.get_slice_range()
        self.slice_spinbox.config(from_=min_slice, to=max_slice)
        self.slice_spinbox.config(state="normal" if max_slice > 0 else "disabled")

        # Update match resolutions checkbox
        self.match_resolutions_var.set(self.presenter.match_resolutions)

        # Update clahe checkboxes
        self.clahe_source_var.set(self.presenter.clahe_active_source)
        self.clahe_dest_var.set(self.presenter.clahe_active_dest)

        self.update_display()

    def on_points_changed(self):
        """Called when control points have changed."""
        # The assessment described the points as they were; it no longer
        # describes the points as they are.
        self.point_quality = None
        self._update_point_count()
        self._refresh_edit_menu()
        self.update_display()

    def on_display_update_needed(self):
        """Called when display needs to be updated."""
        self.update_display()

    def on_error(self, message: str):
        """Called when an error occurs."""
        messagebox.showerror("Error", message)
        self.set_status(f"Error: {message}")

    def on_show_preview_2d(self, warped: np.ndarray, reference: np.ndarray):
        """Called to show transformation preview."""
        # Create preview window
        Viewer = Interactive2DViewer(self, warped, reference, "Transformation Preview")
        self.set_status("Preview window opened")
        Viewer.root.wait_window()
        self.set_status("Preview window closed")

    def on_show_preview_3d(self, warped_stack: np.ndarray, reference_stack: np.ndarray):
        """Called to show transformation preview."""
        # Create preview window
        Viewer = Interactive3DViewer(
            self, warped_stack, reference_stack, "Transformation Preview"
        )
        self.set_status("Preview window opened")
        Viewer.root.wait_window()
        self.set_status("Preview window closed")

    def on_show_matched_points(
        self,
        src_img: np.ndarray,
        dst_img: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ):
        """Called to show matched points visualization."""
        # Create matched points viewer window
        Viewer = MatchedPointsViewer(
            self,
            src_img,
            dst_img,
            src_points,
            dst_points,
            "Matched Points Visualization",
        )
        self.set_status("Matched points viewer opened")
        Viewer.root.wait_window()
        self.set_status("Matched points viewer closed")

    def on_project_loaded(self):
        """Called when a project has been loaded."""
        self.on_data_loaded()
        self._update_point_count()
        self.set_status("Project loaded")

    def on_request_corresponding_point(self, target: str):
        """Called when a corresponding point is needed."""
        self.awaiting_corresponding_point = target
        self.set_status(f"Click on {target} image to add corresponding point")

    def on_project_reset(self):
        """Called when a new project is created."""
        # Clear canvases
        self.left_canvas.delete("all")
        self.right_canvas.delete("all")

        # Reset UI state
        self.current_src_zoom = 100
        self.current_dst_zoom = 100
        self.zoom_src_var.set("100%")
        self.zoom_dst_var.set("100%")
        self.show_points = True
        self.awaiting_corresponding_point = None
        self._drag = None
        self.point_quality = None
        self._sync_smoothing_selector()
        self._refresh_edit_menu()

        # Reset slice control
        self.slice_var.set(0)
        self.slice_spinbox.config(from_=0, to=0, state="disabled")

        # Reset mode selectors
        self.source_mode_var.set("Intensity")
        self.dest_mode_var.set("Intensity")
        self.source_mode_combo["values"] = []
        self.dest_mode_combo["values"] = []

        # Reset CLAHE toggles
        self.clahe_source_var.set(False)
        self.clahe_dest_var.set(False)

        # Reset match resolutions
        self.match_resolutions_var.set(False)

        # Reset cursor and point display
        self.cursor_label.config(text="Cursor: --, --")
        self.points_label.config(text="Points: 0 / 0")

        self.set_status("Ready")

    # ========== Helper Methods ==========

    def update_display(self):
        """Update the image display."""
        # if not self.presenter.source_image or not self.presenter.destination_image:
        #     return

        # Get current images

        src_scale = self.current_src_zoom / 100.0
        dst_scale = self.current_dst_zoom / 100.0
        src_img, dst_img = self.presenter.get_current_images(
            src_scale=src_scale, dst_scale=dst_scale
        )

        if src_img is not None:
            # Clear canvases
            self.left_canvas.delete("all")

            # Convert to PhotoImage and display
            self._display_image(self.left_canvas, src_img)

            # Update scroll region
            self.left_canvas.config(
                scrollregion=(0, 0, src_img.shape[1], src_img.shape[0])
            )

        if dst_img is not None:
            # Clear canvases
            self.right_canvas.delete("all")

            # Convert to PhotoImage and display
            self._display_image(self.right_canvas, dst_img)

            # Update scroll region
            self.right_canvas.config(
                scrollregion=(0, 0, dst_img.shape[1], dst_img.shape[0])
            )

        # Draw points if enabled
        if self.show_points:
            self._draw_points()

        # Ensure progress bar is stopped
        self.show_progress(False)

    def _display_image(self, canvas, image):
        """Display image on canvas."""
        # This would need proper implementation to convert numpy array to PhotoImage
        # For now, just create a placeholder
        image = self._photo_image(image)
        canvas.image = image  # Keep a reference to avoid garbage collection
        canvas.create_image(0, 0, anchor="nw", image=image)

    def _photo_image(self, image: np.ndarray):
        """Creates a PhotoImage object that plays nicely with a tkinter canvas for viewing purposes."""
        height, width, channels = image.shape
        if channels == 1:
            data = (
                f"P5 {width} {height} 255 ".encode() + image.astype(np.uint8).tobytes()
            )
        else:
            ppm_header = f"P6 {width} {height} 255 ".encode()
            data = ppm_header + image.tobytes()
        return tk.PhotoImage(width=width, height=height, data=data, format="PPM")

    def _is_flagged(self, index: int) -> bool:
        """True if the last quality check singled this point out."""
        quality = getattr(self, "point_quality", None)
        if quality is None:
            return False
        outliers = quality.outliers
        return bool(index < len(outliers) and outliers[index])

    def _draw_points(self):
        """Draw control points on canvases."""
        # Scale points for current zoom
        src_points, dst_points = self.presenter.get_points()
        src_scale = self.current_src_zoom / 100.0
        dst_scale = self.current_dst_zoom / 100.0

        # Scale destination points if resolutions are matched
        if self.presenter.match_resolutions:
            src_res, dst_res = self.presenter.get_resolutions()
            res_scale = dst_res / src_res
            dst_points = [(p[0] * res_scale, p[1] * res_scale) for p in dst_points]

        # Draw source points
        for i, point in enumerate(src_points):
            x, y = point[0] * src_scale, point[1] * src_scale
            # A flagged point gets a bigger, warning-coloured ring: a number in
            # a report is no use if you cannot find the point on the canvas.
            flagged = self._is_flagged(i)
            radius = 7 if flagged else 4
            self.left_canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="white",
                outline=self.palette.warning if flagged else "red",
                width=3 if flagged else 1,
                tags=f"point_{i}",
            )
            self.left_canvas.create_text(
                x + 5,
                y - 5,
                text=str(i),
                fill="red",
                anchor="sw",
                tags=f"point_{i}",
                font=("Arial", 11, "bold"),
            )
            self.left_canvas.create_text(
                x + 5,
                y - 5,
                text=str(i),
                fill="white",
                anchor="sw",
                tags=f"point_{i}",
                font=("Arial", 10),
            )

        # Draw destination points
        for i, point in enumerate(dst_points):
            x, y = point[0] * dst_scale, point[1] * dst_scale
            flagged = self._is_flagged(i)
            radius = 7 if flagged else 4
            self.right_canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="white",
                outline=self.palette.warning if flagged else "green",
                width=3 if flagged else 1,
                tags=f"point_{i}",
            )
            self.right_canvas.create_text(
                x + 5,
                y - 5,
                text=str(i),
                fill="green",
                anchor="sw",
                tags=f"point_{i}",
                font=("Arial", 11, "bold"),
            )
            self.right_canvas.create_text(
                x + 5,
                y - 5,
                text=str(i),
                fill="white",
                anchor="sw",
                tags=f"point_{i}",
                font=("Arial", 10),
            )

    def set_status(self, message: str):
        """Update status bar."""
        self.status_label.config(text=message)
        self.update_idletasks()
        logger.info(message)

    def show_progress(self, show: bool):
        """Show or hide progress indicator."""
        if show:
            self.progress_bar.start(1)
        else:
            self.progress_bar.stop()

    def _get_image_resolutions_dialog(self) -> tuple:
        """Show dialog to select transformation type."""
        dialog = tk.Toplevel(self)
        apply_to_window(dialog, self.palette)
        dialog.title("Enter Resolution (µm)")
        dialog.geometry("250x100")
        dialog.transient(self)
        dialog.grab_set()
        dialog.rowconfigure([0, 1, 2], weight=1)
        dialog.columnconfigure([0, 1], weight=1)

        default_src_res, default_dst_res = self.presenter.get_resolutions()
        src_res = tk.StringVar(value=default_src_res)
        dst_res = tk.StringVar(value=default_dst_res)
        result = [default_src_res, default_dst_res]

        sl = ttk.Label(dialog, text="Source:")
        sl.grid(row=0, column=0, sticky="nse", padx=3, pady=3)
        se = ttk.Entry(dialog, textvariable=src_res, width=10)
        se.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)

        dl = ttk.Label(dialog, text="Destination:")
        dl.grid(row=1, column=0, sticky="nse", padx=3, pady=3)
        se = ttk.Entry(dialog, textvariable=dst_res, width=10)
        se.grid(row=1, column=1, sticky="nsew", padx=3, pady=3)

        def on_ok():
            result[0] = float(src_res.get())
            result[1] = float(dst_res.get())
            dialog.destroy()

        def on_cancel():
            dialog.destroy()
            result[0] = None
            result[1] = None

        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=2, column=0, columnspan=2, padx=3, pady=3)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side="left", padx=5
        )

        dialog.wait_window()
        return result[0], result[1]

    def _get_transform_type_dialog(self) -> TransformType | None:
        """Show dialog to select transformation type."""
        dialog = tk.Toplevel(self)
        dialog.title("Select Transform Type")
        dialog.geometry("300x130")
        dialog.transient(self)
        dialog.grab_set()
        # Set background to match main window
        apply_to_window(dialog, self.palette)

        selected = tk.StringVar(value="TPS")

        for transform_type in TransformType:
            tk.Radiobutton(
                dialog,
                text=transform_type.value.replace("_", " ").title(),
                variable=selected,
                value=transform_type.value,
                bg=self.bg,
                fg=self.fg,
                selectcolor=self.bg,
            ).pack(anchor="w", padx=20, pady=5)

        result = [None]

        def on_ok():
            result[0] = TransformType(selected.get())
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(side="bottom", pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side="left", padx=5
        )

        dialog.wait_window()
        return result[0]

    def _get_crop_mode_dialog(self) -> CropMode | None:
        """Show dialog to select crop mode."""
        dialog = tk.Toplevel(self)
        dialog.title("Select Crop Mode")
        dialog.geometry("290x350")
        dialog.transient(self)
        dialog.grab_set()
        # Set background to match main window
        apply_to_window(dialog, self.palette)

        selected = tk.StringVar(value="none")

        for crop_mode in CropMode:
            tk.Radiobutton(
                dialog,
                text=crop_mode.value.replace("_", " ").title(),
                variable=selected,
                value=crop_mode.value,
                bg=self.bg,
                fg=self.fg,
                selectcolor=self.bg,
            ).pack(anchor="w", padx=20, pady=5)

        result = [None]

        # Add a description to the dialog
        description = (
            "Choose how to crop the corrected image:\n"
            "- Source: Crop onto a grid equal to the source image (i.e., source is 100x100 pixels so output is 100x100 pixels). Typically involves cropping the output.\n"
            "- Destination: Crop onto a grid equal to the destination image (i.e., source is 100x100 pixels, destination is 200x200 pixels so output is 200x200 pixels). Typically involves upsampling.\n"
        )
        desc_label = ttk.Label(dialog, text=description, wraplength=260, justify="left")
        desc_label.pack(padx=10, pady=(0, 10))

        def on_ok():
            result[0] = CropMode(selected.get())
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(side="bottom", pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side="left", padx=5
        )

        dialog.wait_window()
        return result[0]

    def _get_export_format_dialog(self) -> DataFormat | None:
        """Show dialog to select export format."""
        dialog = tk.Toplevel(self)
        dialog.title("Select Export Format")
        dialog.geometry("250x260")
        dialog.transient(self)
        dialog.grab_set()
        # Set background to match main window
        apply_to_window(dialog, self.palette)

        selected_format = tk.StringVar(value=DataFormat.IMAGE.value)

        ttk.Label(dialog, text="Data Format:").pack(anchor="w", padx=20, pady=5)
        for data_format in DataFormat:
            tk.Radiobutton(
                dialog,
                text=data_format.value.replace("_", " ").title(),
                variable=selected_format,
                value=data_format.value,
                bg=self.bg,
                fg=self.fg,
                selectcolor=self.bg,
            ).pack(anchor="w", padx=20, pady=5)

        result = [None]

        def on_ok():
            result[0] = DataFormat(selected_format.get())
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(side="bottom", pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side="left", padx=5
        )

        dialog.wait_window()
        return result[0]

    def _get_modality_name_dialog(self, filename: str) -> str | None:
        """Show dialog to enter a modality name for an image."""
        dialog = tk.Toplevel(self)
        dialog.title(f"Loading {filename}")
        dialog.geometry("350x150")
        dialog.transient(self)
        dialog.grab_set()
        apply_to_window(dialog, self.palette)

        # Main frame
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Label
        ttk.Label(
            main_frame, text="Enter a name for this image modality:", wraplength=300
        ).pack(pady=(0, 10))

        # Entry field
        modality_var = tk.StringVar(value="")
        entry = ttk.Entry(main_frame, textvariable=modality_var, width=25)
        entry.pack(pady=5)
        entry.focus()

        result = [None]

        def on_ok():
            name = modality_var.get().strip()
            if name:
                result[0] = name
                dialog.destroy()
            else:
                messagebox.showwarning(
                    "Invalid Name", "Please enter a valid modality name."
                )

        def on_cancel():
            dialog.destroy()

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side="left", padx=5
        )

        # Bind Enter key to OK
        entry.bind("<Return>", lambda e: on_ok())
        dialog.bind("<Escape>", lambda e: on_cancel())

        dialog.wait_window()
        return result[0]

    def _get_point_clear_dialog(self) -> str | None:
        """Show dialog to choose point clearing option."""
        dialog = tk.Toplevel(self)
        dialog.title("Clear Points")
        dialog.geometry("300x180")
        dialog.transient(self)
        dialog.grab_set()
        apply_to_window(dialog, self.palette)

        # Main frame
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Label
        ttk.Label(
            main_frame, text="Choose an option to clear points:", wraplength=250
        ).pack(pady=(0, 10))

        result = [None]

        def on_ok():
            result[0] = selected_option.get()
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        # Radio buttons
        selected_option = tk.StringVar(value="image")
        options = [("Current Image", "image"), ("Entire Stack", "stack")]
        for text, value in options:
            tk.Radiobutton(
                main_frame,
                text=text,
                variable=selected_option,
                value=value,
                bg=self.bg,
                fg=self.fg,
                selectcolor=self.bg,
            ).pack(anchor="w", padx=20, pady=5)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side="left", padx=5
        )

        dialog.wait_window()
        return result[0]

    def _get_auto_detect_params_dialog(self, method: str) -> dict | None:
        """Show dialog to configure auto point detection parameters.

        Args:
            method: Detection method ('sift' or 'matchanything')

        Returns:
            Dictionary of parameters or None if cancelled
        """
        dialog = tk.Toplevel(self)
        dialog.title(f"Auto Detection Parameters ({method.upper()})")
        dialog.transient(self)
        dialog.grab_set()
        apply_to_window(dialog, self.palette)

        # Main frame
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Parameter variables
        entries = {}

        if method == "sift":
            dialog.geometry("450x340")
            # SIFT-specific parameters
            param_defs = [
                ("max_ratio", "Max Ratio:", 0.75, "Lowe's ratio test threshold (0-1)"),
                (
                    "min_matches",
                    "Min Matches:",
                    4,
                    "Minimum number of matches required",
                ),
                ("sigma", "Sigma:", 0.5, "Gaussian blur sigma"),
                ("num_samples", "Num Samples:", 10, "Number of RANSAC samples"),
                (
                    "ransac_threshold",
                    "RANSAC Threshold:",
                    5.5,
                    "RANSAC inlier threshold",
                ),
                (
                    "ransac_max_trials",
                    "RANSAC Max Trials:",
                    1000,
                    "Maximum RANSAC iterations",
                ),
            ]
        else:  # matchanything
            dialog.geometry("430x270")
            param_defs = [
                ("num_samples", "Num Samples:", 10, "Number of point samples"),
                (
                    "ransac_threshold",
                    "RANSAC Threshold:",
                    0.05,
                    "RANSAC inlier threshold (relative)",
                ),
                (
                    "ransac_max_trials",
                    "RANSAC Max Trials:",
                    100,
                    "Maximum RANSAC iterations",
                ),
            ]

        # RANSAC method (common to both)
        ransac_method_var = tk.StringVar(value="deformable")
        ransac_filter_var = tk.BooleanVar(value=True)

        # Create parameter entries
        for _i, (key, label, default, tooltip) in enumerate(param_defs):
            row_frame = ttk.Frame(main_frame)
            row_frame.pack(fill="x", pady=3)

            lbl = ttk.Label(row_frame, text=label, width=18, anchor="e")
            lbl.pack(side="left", padx=(0, 5))

            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(row_frame, textvariable=var, width=12)
            entry.pack(side="left")
            entries[key] = var

            # Tooltip label
            tip_lbl = ttk.Label(
                row_frame, text=tooltip, foreground=self.palette.muted_foreground
            )
            tip_lbl.pack(side="left", padx=(10, 0))

        # RANSAC filter checkbox (for matchanything only)
        if method == "matchanything":
            filter_frame = ttk.Frame(main_frame)
            filter_frame.pack(fill="x", pady=3)
            ttk.Label(filter_frame, text="", width=18).pack(side="left")
            ttk.Checkbutton(
                filter_frame, text="Enable RANSAC filtering", variable=ransac_filter_var
            ).pack(side="left")

        # RANSAC method selection
        method_frame = ttk.Frame(main_frame)
        method_frame.pack(fill="x", pady=(10, 3))
        ttk.Label(method_frame, text="RANSAC Method:", width=18, anchor="e").pack(
            side="left", padx=(0, 5)
        )
        ransac_combo = ttk.Combobox(
            method_frame,
            textvariable=ransac_method_var,
            values=["deformable", "affine", "projective"],
            state="readonly",
            width=15,
        )
        ransac_combo.pack(side="left")

        result = [None]

        def on_ok():
            try:
                params = {}
                for key, var in entries.items():
                    val = var.get()
                    # Try to convert to appropriate type
                    if "." in val:
                        params[key] = float(val)
                    else:
                        params[key] = int(val)

                params["ransac_method"] = ransac_method_var.get()
                if method == "matchanything":
                    params["ransac_filter"] = ransac_filter_var.get()

                result[0] = params
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror(
                    "Invalid Input", f"Please enter valid numbers: {e}"
                )

        def on_cancel():
            dialog.destroy()

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 5))
        ttk.Button(button_frame, text="Detect", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side="left", padx=5
        )

        # Bind Enter key
        dialog.bind("<Return>", lambda e: on_ok())
        dialog.bind("<Escape>", lambda e: on_cancel())

        dialog.wait_window()
        return result[0]


class MatchedPointsViewer:
    """Tkinter implementation of a matched points visualization viewer"""

    def __init__(
        self, master, src_img, dst_img, src_points, dst_points, title="Matched Points"
    ):
        """
        Initialize the matched points viewer

        Parameters:
        -----------
        src_img : numpy.ndarray
            Source image
        dst_img : numpy.ndarray
            Destination image
        src_points : numpy.ndarray
            Source points array (N x 2)
        dst_points : numpy.ndarray
            Destination points array (N x 2)
        title : str
            Window title
        """
        self.master = master
        # Popups are separate Toplevels: ttk styling reaches ttk widgets
        # but not the window itself or plain Tk widgets like the canvas.
        self.palette = palette_of(master)
        self.src_img = self._normalize_image(src_img)
        self.dst_img = self._normalize_image(dst_img)
        self.src_points = src_points
        self.dst_points = dst_points
        self.title = title
        self.monochromatic = False

        # Setup the GUI
        self.setup_gui()

    def _normalize_image(self, img):
        """Normalize image to 0-255 range and ensure it's in the right format"""
        if img.dtype == np.float64 or img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)

        return img

    def _get_window_size(self):
        """Calculate appropriate window size based on image dimensions"""
        # Calculate combined width (both images side by side)
        total_width = self.src_img.shape[1] + self.dst_img.shape[1]
        max_height = max(self.src_img.shape[0], self.dst_img.shape[0])

        display_height = self.root.winfo_screenheight()
        display_width = self.root.winfo_screenwidth()

        # Scale to fit screen with some margin
        ratio = total_width / max_height
        if ratio >= 1:
            width = min(display_width * 0.9, int(display_height * ratio * 0.8))
            height = int(display_height * 0.8)
        else:
            width = int(display_width * 0.9)
            height = min(display_height * 0.9, int(display_width * 0.9 / ratio))

        return int(width), int(height)

    def setup_gui(self):
        """Setup the tkinter GUI"""
        # Create main window
        self.root = tk.Toplevel(self.master)
        apply_to_window(self.root, self.palette)
        self.root.title(self.title)
        width, height = self._get_window_size()
        self.root.geometry(f"{width}x{height}")

        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Create canvas for display
        self.canvas = tk.Canvas(main_frame, bg=self.palette.canvas)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        info_text = f"Showing {len(self.src_points)} matched point pairs"
        info_label = ttk.Label(info_frame, text=info_text)
        info_label.pack(side=tk.LEFT, padx=5)

        # Close button
        close_button = ttk.Button(info_frame, text="Close", command=self.root.destroy)
        close_button.pack(side=tk.RIGHT, padx=5)

        # Initial display
        self.update_display()

        # Bind canvas resize event
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def create_matched_visualization(self):
        """Create visualization with both images and connecting lines"""
        # Get dimensions
        h1, w1 = self.src_img.shape[:2]
        h2, w2 = self.dst_img.shape[:2]

        # Create combined canvas
        max_height = max(h1, h2)
        combined_width = w1 + w2
        combined_img = np.ones((max_height, combined_width, 3), dtype=np.uint8) * 128

        # Place source image on left
        combined_img[:h1, :w1] = self.src_img

        # Place destination image on right
        combined_img[:h2, w1 : w1 + w2] = self.dst_img

        # Draw lines and points
        for i in range(len(self.src_points)):
            src_x, src_y = int(self.src_points[i, 0]), int(self.src_points[i, 1])
            dst_x, dst_y = int(self.dst_points[i, 0]) + w1, int(self.dst_points[i, 1])
            # Randomly generate a very bright color for visibility
            lc, p0c, p1c = self._make_color()

            # Draw line between points (using simple line drawing)
            self._draw_line(combined_img, src_x, src_y, dst_x, dst_y, lc)

            # Draw circles at point locations
            self._draw_circle(combined_img, src_x, src_y, 5, p0c)
            self._draw_circle(combined_img, dst_x, dst_y, 5, p1c)
        return Image.fromarray(combined_img)

    def _make_color(self):
        if self.monochromatic:
            return (255, 0, 0), (255, 0, 0), (0, 0, 255)
        H = float(np.random.randint(0, 360))
        S = 1.0
        V = 1.0
        C = V * S
        X = C * (1 - abs((H / 60) % 2 - 1))
        m = V - C
        if 0 <= H < 60:
            r1, g1, b1 = C, X, 0
        elif 60 <= H < 120:
            r1, g1, b1 = X, C, 0
        elif 120 <= H < 180:
            r1, g1, b1 = 0, C, X
        elif 180 <= H < 240:
            r1, g1, b1 = 0, X, C
        elif 240 <= H < 300:
            r1, g1, b1 = X, 0, C
        else:
            r1, g1, b1 = C, 0, X
        r, g, b = int((r1 + m) * 255), int((g1 + m) * 255), int((b1 + m) * 255)
        return (r, g, b), (r, g, b), (r, g, b)

    def _draw_line(self, img, x0, y0, x1, y1, color):
        """Draw a line on the image using Bresenham's algorithm"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # Check bounds
            if 0 <= y0 < img.shape[0] and 0 <= x0 < img.shape[1]:
                img[y0, x0] = color

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _draw_circle(self, img, cx, cy, radius, color):
        """Draw a filled circle on the image"""
        for y in range(max(0, cy - radius), min(img.shape[0], cy + radius + 1)):
            for x in range(max(0, cx - radius), min(img.shape[1], cx + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    img[y, x] = color

    def update_display(self):
        """Update the displayed image"""
        # Create matched visualization
        pil_image = self.create_matched_visualization()

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            # Calculate scaling to fit canvas while maintaining aspect ratio
            scale_x = canvas_width / pil_image.width
            scale_y = canvas_height / pil_image.height
            scale = min(scale_x, scale_y)

            new_width = int(pil_image.width * scale)
            new_height = int(pil_image.height * scale)

            # Resize image
            pil_image = pil_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)

        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2 if canvas_width > 1 else 200,
            canvas_height // 2 if canvas_height > 1 else 200,
            image=self.photo,
            anchor=tk.CENTER,
        )

    def on_canvas_resize(self, event):
        """Handle canvas resize event"""
        self.update_display()


class Interactive3DViewer:
    """Tkinter implementation of an interactive 3D stack viewer with plane selection and split view controls"""

    def __init__(self, master, stack0, stack1, title="Interactive View"):
        """
        Initialize the interactive 3D viewer

        Parameters:
        -----------
        stack0 : numpy.ndarray
            First image stack (4D: slices, rows, cols, channels or 3D: slices, rows, cols)
        stack1 : numpy.ndarray
            Second image stack (4D: slices, rows, cols, channels or 3D: slices, rows, cols)
        title : str
            Window title
        """
        print("Initializing Interactive 3D Viewer...")
        stack0, stack1 = self._prepare_stacks(stack0, stack1)

        self.master = master
        # Popups are separate Toplevels: ttk styling reaches ttk widgets
        # but not the window itself or plain Tk widgets like the canvas.
        self.palette = palette_of(master)
        self.stack0 = stack0
        self.stack1 = stack1
        self.title = title
        self.active = 0  # 0 for row slider, 1 for col slider

        # Initialize dimensions
        self.max_x = self.stack0.shape[2]
        self.max_y = self.stack0.shape[1]
        self.max_z = self.stack0.shape[0] - 1
        self.max_r = self.max_y
        self.max_c = self.max_x
        self.max_s = self.max_z
        if self.max_s == 0:
            self.max_s = 1

        # Current state
        self.current_slice = 0
        self.current_plane = 0  # 0=XY, 1=XZ, 2=YZ
        self.current_row_split = 0
        self.current_col_split = 0

        # Setup the GUI
        self.setup_gui()

    def get_limits(self, axis, shape):
        """Update dimension limits based on selected plane"""
        if axis == 0:
            self.max_r = self.max_y
            self.max_c = self.max_x
            self.max_s = self.max_z
        elif axis == 1:
            self.max_r = self.max_z
            self.max_c = self.max_x
            self.max_s = self.max_y
        elif axis == 2:
            self.max_r = self.max_z
            self.max_c = self.max_y
            self.max_s = self.max_x

    def _create_slice(self, slice_num, axis, split_num, split_axis):
        """Create a composite slice from the two stacks"""
        # Extract slices from both stacks based on axis
        if axis == 0:  # XY plane
            im0 = self.stack0[slice_num]
            im1 = self.stack1[slice_num]
        elif axis == 1:  # XZ plane
            im0 = self.stack0[:, slice_num]
            im1 = self.stack1[:, slice_num]
        elif axis == 2:  # YZ plane
            im0 = self.stack0[:, :, slice_num]
            im1 = self.stack1[:, :, slice_num]

        # Adjust aspect ratio for thin slices
        if im0.shape[0] < im0.shape[1] / 2:
            repeat_factor = int(np.floor(im0.shape[1] / im0.shape[0]))
            im0 = np.repeat(im0, repeat_factor, axis=0)
            im1 = np.repeat(im1, repeat_factor, axis=0)
        elif im0.shape[1] < im0.shape[0] / 2:
            repeat_factor = int(np.floor(im0.shape[0] / im0.shape[1]))
            im0 = np.repeat(im0, repeat_factor, axis=1)
            im1 = np.repeat(im1, repeat_factor, axis=1)

        # Create split view
        if split_axis == 0:  # Row split
            split_num = im0.shape[0] - split_num
            image = np.vstack((im0[:split_num], im1[split_num:]))
        elif split_axis == 1:  # Column split
            image = np.hstack((im0[:, :split_num], im1[:, split_num:]))

        return image

    def _prepare_stacks(self, stack0, stack1):
        # Make sure both have the same number of channels
        if stack0.shape[-1] < stack1.shape[-1]:
            stack0 = np.repeat(stack0, stack1.shape[-1], axis=-1)
        elif stack1.shape[-1] < stack0.shape[-1]:
            stack1 = np.repeat(stack1, stack0.shape[-1], axis=-1)

        # If 1 channel, convert to 3-channel RGB
        if stack0.shape[-1] == 1:
            stack0 = np.concatenate([stack0, stack0, stack0], axis=-1)
            stack1 = np.concatenate([stack1, stack1, stack1], axis=-1)

        # Normalize both stacks to uint8
        if stack0.dtype != np.uint8:
            stack0 = (stack0.astype(np.float32) / np.max(stack0) * 255).astype(np.uint8)
        if stack1.dtype != np.uint8:
            stack1 = (stack1.astype(np.float32) / np.max(stack1) * 255).astype(np.uint8)

        return stack0, stack1

    def _get_window_size(self):
        """Calculate appropriate window size based on image dimensions and screen size"""
        display_height = self.root.winfo_screenheight()
        display_width = self.root.winfo_screenwidth()

        # Get current slice to determine aspect ratio
        test_image = self._create_slice(0, self.current_plane, 0, 0)
        ratio = test_image.shape[1] / test_image.shape[0]

        if ratio >= 1:
            width = min(display_width, int(display_height * ratio * 0.8))
            height = int(display_height * 0.8)
        else:
            width = int(display_width * 0.8)
            height = min(display_height, int(display_width * 0.8 / ratio))
        return width, height

    def setup_gui(self):
        """Setup the tkinter GUI"""
        # Create main window
        self.root = tk.Toplevel(self.master)
        apply_to_window(self.root, self.palette)
        self.root.title(self.title)
        width, height = self._get_window_size()
        self.root.geometry(f"{width}x{height}")

        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Create canvas for image display
        self.canvas = tk.Canvas(main_frame, bg=self.palette.canvas)
        self.canvas.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left controls (row split slider)
        left_controls = ttk.Frame(main_frame)
        left_controls.grid(row=0, column=0, sticky=(tk.N, tk.S), padx=5)

        ttk.Label(left_controls, text="Y split").pack(side=tk.TOP)
        self.row_slider = ttk.Scale(
            left_controls,
            from_=self.max_r,
            to=0,
            orient=tk.VERTICAL,
            value=self.max_r,
            command=self.update_row_split,
        )
        self.row_slider.pack(side=tk.TOP, fill=tk.Y, expand=True)
        self.row_value_label = ttk.Label(left_controls, text=str(self.max_r))
        self.row_value_label.pack(side=tk.TOP)

        # Bottom controls frame
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        # Column split slider
        col_controls = ttk.Frame(bottom_frame)
        col_controls.pack(side=tk.TOP, fill=tk.X, expand=True)

        ttk.Label(col_controls, text="X split:").pack(side=tk.LEFT)
        self.col_slider = ttk.Scale(
            col_controls,
            from_=0,
            to=self.max_c,
            orient=tk.HORIZONTAL,
            value=0,
            command=self.update_col_split,
        )
        self.col_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.col_value_label = ttk.Label(col_controls, text="0")
        self.col_value_label.pack(side=tk.LEFT)

        # Slice and plane controls
        control_row = ttk.Frame(bottom_frame)
        control_row.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Slice control
        ttk.Label(control_row, text="Slice:").pack(side=tk.LEFT, padx=5)
        self.slice_slider = ttk.Scale(
            control_row,
            from_=0,
            to=self.max_s,
            orient=tk.HORIZONTAL,
            value=0,
            command=self.update_slice,
        )
        self.slice_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.slice_value_label = ttk.Label(control_row, text="0")
        self.slice_value_label.pack(side=tk.LEFT)

        # Plane selection
        ttk.Label(control_row, text="Plane:").pack(side=tk.LEFT, padx=(20, 5))
        self.plane_var = tk.StringVar(value="XY")
        plane_combo = ttk.Combobox(
            control_row,
            textvariable=self.plane_var,
            values=["XY", "XZ", "YZ"],
            state="readonly",
            width=5,
        )
        plane_combo.pack(side=tk.LEFT, padx=5)
        plane_combo.bind("<<ComboboxSelected>>", self.change_plane)

        # Reset button
        ttk.Button(control_row, text="Reset", command=self.reset_controls).pack(
            side=tk.RIGHT, padx=5
        )

        # Right controls (slice slider) - removed since we have it in bottom controls

        # Initial display
        self.update_display()

        # Bind canvas resize event
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def reset_controls(self):
        """Reset all controls to initial positions"""
        self.row_slider.set(0)
        self.col_slider.set(0)
        self.slice_slider.set(0)
        self.plane_var.set("XY")
        self.current_plane = 0
        self.update_display()

    def update_row_split(self, val):
        """Update function for row split slider"""
        val = int(float(val))
        self.current_row_split = val
        self.active = 0
        self.row_value_label.config(text=str(val))
        self.update_display()

    def update_col_split(self, val):
        """Update function for column split slider"""
        val = int(float(val))
        self.current_col_split = val
        self.active = 1
        self.col_value_label.config(text=str(val))
        self.update_display()

    def update_slice(self, val):
        """Update function for slice slider"""
        val = int(float(val))
        self.current_slice = val
        self.slice_value_label.config(text=str(val))
        self.update_display()

    def change_plane(self, event=None):
        """Handle plane selection change"""
        plane_str = self.plane_var.get()
        self.current_plane = ["XY", "XZ", "YZ"].index(plane_str)

        # Get new dimensions
        test_image = self._create_slice(0, self.current_plane, 0, 0)
        self.get_limits(self.current_plane, test_image.shape)

        # Update slider ranges
        self.row_slider.config(to=self.max_r)
        self.col_slider.config(to=self.max_c)
        self.slice_slider.config(to=self.max_s)

        # Reset splits
        self.current_row_split = 0
        self.current_col_split = 0
        self.current_slice = 0
        self.row_slider.set(0)
        self.col_slider.set(0)
        self.slice_slider.set(0)

        self.update_display()

    def update_display(self):
        """Update the displayed image"""
        # Create composite image
        image = self._create_slice(
            self.current_slice,
            self.current_plane,
            self.current_row_split if self.active == 0 else self.current_col_split,
            self.active,
        )

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            # Calculate scaling to fit canvas while maintaining aspect ratio
            scale_x = canvas_width / image.shape[1]
            scale_y = canvas_height / image.shape[0]
            scale = min(scale_x, scale_y)

            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)

            # Resize image
            pil_image = pil_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)

        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2 if canvas_width > 1 else 200,
            canvas_height // 2 if canvas_height > 1 else 200,
            image=self.photo,
            anchor=tk.CENTER,
        )

        # Update title with current slice info
        plane_name = ["XY", "XZ", "YZ"][self.current_plane]
        self.root.title(
            f"{self.title} - {plane_name} Plane (Slice {self.current_slice})"
        )

    def on_canvas_resize(self, event):
        """Handle canvas resize event"""
        self.update_display()

    def run(self):
        """Start the GUI main loop"""
        self.root.after(100, self.update_display)
        self.root.mainloop()


class Interactive2DViewer:
    """Tkinter implementation of an interactive image overlay viewer with slider controls"""

    def __init__(self, master, im0, im1, title="Interactive View"):
        """
        Initialize the interactive viewer

        Parameters:
        -----------
        im0 : numpy.ndarray
            First image (overlay image) - should be grayscale or RGB
        im1 : numpy.ndarray
            Second image (background image) - should be grayscale or RGB
        title : str
            Window title
        """
        self.master = master
        # Popups are separate Toplevels: ttk styling reaches ttk widgets
        # but not the window itself or plain Tk widgets like the canvas.
        self.palette = palette_of(master)
        self.im0_original = self._normalize_image(im0)
        self.im1_original = self._normalize_image(im1)
        self.title = title

        # Get dimensions
        self.max_r = im0.shape[0]
        self.max_c = im0.shape[1]

        # Initialize alpha mask
        self.alphas = np.ones((self.max_r, self.max_c))

        # Current slider values
        self.current_row_val = self.max_r // 2
        self.current_col_val = self.max_c // 2

        # Setup the GUI
        self.setup_gui()

        # Force row/col update
        self.update_row(self.current_row_val)

    def _normalize_image(self, img):
        """Normalize image to 0-255 range and ensure it's in the right format"""
        # Handle different image formats
        if img.dtype == np.float64 or img.dtype == np.float32:
            # Assume values are in [0, 1] range
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            # Convert to uint8
            img = img.astype(np.uint8)

        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)

        return img

    def _get_window_size(self):
        """Calculate appropriate window size based on image dimensions and screen size"""
        ratio = self.max_c / self.max_r
        display_height = self.root.winfo_screenheight()
        display_width = self.root.winfo_screenwidth()
        if ratio >= 1:
            width = min(display_width, int(display_height * ratio * 0.8))
            height = int(display_height * 0.8)
        else:
            width = int(display_width * 0.8)
            height = min(display_height, int(display_width * 0.8 / ratio))
        return width, height

    def setup_gui(self):
        """Setup the tkinter GUI"""
        # Create main window
        self.root = tk.Toplevel(self.master)
        apply_to_window(self.root, self.palette)
        self.root.title(self.title)
        width, height = self._get_window_size()
        self.root.geometry(f"{width}x{height}")

        # Tk variables need a window to own them, so they are created here
        # rather than in __init__.
        self.blend_mode_var = tk.StringVar(value=overlays.BLEND_MODES[0])
        self.tile_size_var = tk.IntVar(value=overlays.DEFAULT_TILE_SIZE)

        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Create canvas for image display
        self.canvas = tk.Canvas(main_frame, bg=self.palette.canvas)
        self.canvas.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create vertical slider (Y position)
        self.y_slider_frame = ttk.Frame(main_frame)
        self.y_slider_frame.grid(row=0, column=0, sticky=(tk.N, tk.S))

        y_label = ttk.Label(self.y_slider_frame, text="Y pos")
        y_label.pack(side=tk.TOP)

        self.y_slider = ttk.Scale(
            self.y_slider_frame,
            from_=self.max_r,
            to=0,
            orient=tk.VERTICAL,
            value=self.max_r,
            command=self.update_row,
        )
        self.y_slider.pack(side=tk.TOP, fill=tk.Y, expand=True)

        self.y_value_label = ttk.Label(self.y_slider_frame, text=str(self.max_r))
        self.y_value_label.pack(side=tk.TOP)

        # Create horizontal slider (X position)
        self.x_slider_frame = ttk.Frame(main_frame)
        self.x_slider_frame.grid(row=1, column=1, sticky=(tk.W, tk.E))

        x_label = ttk.Label(self.x_slider_frame, text="X pos: ")
        x_label.pack(side=tk.LEFT)

        self.x_slider = ttk.Scale(
            self.x_slider_frame,
            from_=0,
            to=self.max_c,
            orient=tk.HORIZONTAL,
            value=0,
            command=self.update_col,
        )
        self.x_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.x_value_label = ttk.Label(self.x_slider_frame, text="0")
        self.x_value_label.pack(side=tk.LEFT)

        # Add info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        info_text = f"Image dimensions: {self.max_c} x {self.max_r}"
        info_label = ttk.Label(info_frame, text=info_text)
        info_label.pack(side=tk.LEFT, padx=5)

        # Comparison mode. A wipe answers "is this edge in the right place?";
        # the checkerboard and difference answer "is the whole thing aligned?"
        ttk.Label(info_frame, text="Mode:").pack(side=tk.LEFT, padx=(20, 5))
        self.mode_combo = ttk.Combobox(
            info_frame,
            textvariable=self.blend_mode_var,
            values=list(overlays.BLEND_MODES),
            state="readonly",
            width=12,
        )
        self.mode_combo.pack(side=tk.LEFT, padx=5)
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_blend_mode_changed)

        ttk.Label(info_frame, text="Tile:").pack(side=tk.LEFT, padx=(10, 5))
        self.tile_spinbox = ttk.Spinbox(
            info_frame,
            from_=4,
            to=256,
            increment=4,
            textvariable=self.tile_size_var,
            width=5,
            command=self.update_display,
        )
        self.tile_spinbox.pack(side=tk.LEFT, padx=5)

        # Add reset button
        reset_button = ttk.Button(info_frame, text="Reset", command=self.reset_sliders)
        reset_button.pack(side=tk.RIGHT, padx=5)

        # Leaves the mode-specific controls in the right state for the
        # starting mode, and draws the first frame.
        self._on_blend_mode_changed()

        # Bind canvas resize event
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def reset_sliders(self):
        """Reset sliders to initial positions"""
        self.y_slider.set(self.max_r)
        self.x_slider.set(0)
        self.update_display()

    def update_row(self, val):
        """Update function for Y position slider"""
        val = int(float(val))
        self.current_row_val = val
        self.y_value_label.config(text=str(val))

        # Update alpha mask
        new_alphas = np.ones_like(self.alphas)
        new_alphas[:val, :] = 0
        # Flip to match matplotlib behavior
        self.alphas = new_alphas[::-1]

        self.update_display()

    def update_col(self, val):
        """Update function for X position slider"""
        val = int(float(val))
        self.current_col_val = val
        self.x_value_label.config(text=str(val))

        # Update alpha mask based on current row value
        new_alphas = np.ones_like(self.alphas)

        # First apply row mask
        row_val = self.current_row_val
        new_alphas[:row_val, :] = 0
        new_alphas = new_alphas[::-1]

        # Then apply column mask
        new_alphas[:, :val] = 0
        self.alphas = new_alphas

        self.update_display()

    def blend_images(self):
        """Combine the two images using the selected comparison mode."""
        combined = overlays.composite(
            self.blend_mode_var.get(),
            self.im0_original,
            self.im1_original,
            alphas=self.alphas,
            tile_size=self.tile_size_var.get(),
        )
        return Image.fromarray(combined, "RGB")

    def _on_blend_mode_changed(self, event=None):
        """Show or hide the controls that only apply to one mode."""
        mode = self.blend_mode_var.get()

        # The wipe sliders position a boundary that only the wipe draws, and
        # the tile size only means anything to the checkerboard. Leaving both
        # live in every mode invites fiddling with a control that does nothing.
        wipe_state = "normal" if mode == "wipe" else "disabled"
        self.y_slider.state(["!disabled"] if mode == "wipe" else ["disabled"])
        self.x_slider.state(["!disabled"] if mode == "wipe" else ["disabled"])
        self.tile_spinbox.config(
            state="normal" if mode == "checkerboard" else "disabled"
        )
        logger.debug("Preview mode %s (wipe sliders %s)", mode, wipe_state)

        self.update_display()

    def update_display(self):
        """Update the displayed image"""
        # Blend images
        blended = self.blend_images()

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            # Calculate scaling to fit canvas while maintaining aspect ratio
            scale_x = canvas_width / self.max_c
            scale_y = canvas_height / self.max_r
            scale = min(scale_x, scale_y)

            new_width = int(self.max_c * scale)
            new_height = int(self.max_r * scale)

            # Resize image
            blended = blended.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(blended)

        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2 if canvas_width > 1 else 200,
            canvas_height // 2 if canvas_height > 1 else 200,
            image=self.photo,
            anchor=tk.CENTER,
        )

    def on_canvas_resize(self, event):
        """Handle canvas resize event"""
        self.update_display()

    def run(self):
        """Start the GUI main loop"""
        # Schedule initial update after window is fully loaded
        self.root.after(100, self.update_display)
        self.root.mainloop()


# ========== Main Entry Point ==========


def log_file_path() -> Path:
    """Return the path of the application log file.

    Logs go to a per-user data directory rather than the current working
    directory, which may well be read-only (or shared) when tpsreg is launched
    from an installed console script.
    """
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))

    return base / "tpsreg" / "tpsreg.log"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure application logging to the console and a rotating log file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    log_path = log_file_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3)
        )
    except OSError as exc:
        # Console logging alone is better than refusing to start.
        print(f"Warning: could not open log file {log_path}: {exc}", file=sys.stderr)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def main() -> None:
    """Console entry point for the ``tpsreg`` command."""
    parser = argparse.ArgumentParser(
        prog="tpsreg",
        description=(
            "Multimodal image registration GUI. Aligns images using a "
            "thin-plate spline fitted to user-selected control points."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug-level logging",
    )
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    logger.info("Starting tpsreg %s", __version__)

    try:
        app = ModernDistortionCorrectionView()
    except tk.TclError as exc:
        # The most common cause by far is a headless machine or a missing
        # Tk installation; a traceback here is pure noise for a scientist.
        print(
            f"Could not start the graphical interface: {exc}\n\n"
            "tpsreg needs a graphical display and a working Tk installation.\n"
            "On Linux, install Tk with your package manager, for example:\n"
            "    sudo apt install python3-tk",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    app.mainloop()


if __name__ == "__main__":
    main()

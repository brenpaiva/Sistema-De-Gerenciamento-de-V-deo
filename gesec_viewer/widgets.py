from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .benchmark import BENCHMARK_PROFILES, BenchmarkResult, BenchmarkWorker, save_benchmark_report
from .camera_store import CameraStore
from .config import AppConfig, CameraConfig
from .fps import FPSCounter
from .frames import FramePacket
from .gpu import DeviceInfo
from .performance import PerformancePolicy
from .ptz import PtzController
from .recording import RecordingManager
from .worker import CameraWorker


logger = logging.getLogger(__name__)


class VideoTile(QWidget):
    """Widget that renders one camera tile, overlay, and status metadata."""
    clicked = pyqtSignal(str)
    double_clicked = pyqtSignal(str)

    def __init__(self, camera_id: str, camera_name: str) -> None:
        super().__init__()
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.status = "Aguardando"
        self.backend = "-"
        self.render_fps = 0.0
        self.recording = False
        self.selected = False
        self._pixmap: QPixmap | None = None
        self._fps_counter = FPSCounter()
        self._frame_sequence = -1
        self._last_counted_frame_sequence = -1
        self.setMinimumSize(300, 190)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_status(self, status: str) -> None:
        self.status = status
        if status in {"Parado", "Erro de conexão"}:
            self._fps_counter.reset()
            self.render_fps = 0.0
        self.update()

    def set_recording(self, recording: bool) -> None:
        self.recording = recording
        self.update()

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self.update()

    def set_frame(self, packet: FramePacket) -> None:
        frame = packet.frame
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self.backend = packet.backend.upper()
        self._frame_sequence = packet.sequence
        self.update()

    def mousePressEvent(self, event: Any) -> None:
        self.clicked.emit(self.camera_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        self.double_clicked.emit(self.camera_id)
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#0b0f12"))

        if self._pixmap is None:
            painter.setPen(QColor("#7d8a8a"))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(self.rect(), Qt.AlignCenter, "Aguardando stream")
        else:
            target = self._pixmap.size()
            target.scale(self.size(), Qt.KeepAspectRatio)
            x = (self.width() - target.width()) // 2
            y = (self.height() - target.height()) // 2
            painter.drawPixmap(x, y, target.width(), target.height(), self._pixmap)
            if self._frame_sequence != self._last_counted_frame_sequence:
                self.render_fps = self._fps_counter.tick()
                self._last_counted_frame_sequence = self._frame_sequence

        self._draw_overlay(painter)
        self._draw_border(painter)
        painter.end()

    def _draw_overlay(self, painter: QPainter) -> None:
        painter.fillRect(0, 0, self.width(), 50, QColor(0, 0, 0, 165))
        painter.setPen(QColor("#f4f7f2"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(14, 20, self.camera_name)

        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor("#b9c3bd"))
        painter.drawText(14, 40, self.status)

        painter.setPen(QColor("#dce8dd"))
        right_text = f"{self.render_fps:04.1f} FPS | {self.backend}"
        painter.drawText(0, 0, self.width() - 14, 34, Qt.AlignRight | Qt.AlignVCenter, right_text)

        if self.recording:
            painter.setBrush(QColor("#e53e3e"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.width() - 86, 36, 10, 10)
            painter.setPen(QColor("#ffecec"))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(self.width() - 70, 45, "REC")

    def _draw_border(self, painter: QPainter) -> None:
        color = QColor("#2f7dff") if self.selected else QColor("#27302c")
        width = 3 if self.selected else 1
        painter.setPen(color)
        painter.drawRect(width // 2, width // 2, self.width() - width, self.height() - width)


class MainWindow(QMainWindow):
    """Main application window with navigation, camera grid, and actions."""
    def __init__(
        self,
        app_config: AppConfig,
        device_info: DeviceInfo,
        camera_store: CameraStore | None = None,
        hardware_decode_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.app_config = app_config
        self.device_info = device_info
        self.camera_store = camera_store or CameraStore(app_config.source_path or "config/demo.yaml")
        self.hardware_decode_mode = hardware_decode_mode
        self.recording_manager = RecordingManager()
        self.ptz_controller = PtzController()
        self.performance_policy = PerformancePolicy()

        self._tiles: dict[str, VideoTile] = {}
        self._workers: dict[str, CameraWorker] = {}
        self._threads: dict[str, Any] = {}
        self._worker_generations: dict[str, int] = {}
        self._latest_frames: dict[str, FramePacket] = {}
        self._last_rendered_sequence: dict[str, int] = {}
        self._camera_status: dict[str, str] = {}
        self._stream_aliases: dict[str, list[str]] = {}
        self._stream_sources: dict[str, tuple[str, str]] = {}
        self._nav_buttons: dict[str, QPushButton] = {}
        self._pages: dict[str, QWidget] = {}
        self._recording_buttons: dict[str, QPushButton] = {}
        self._recording_labels: dict[str, QLabel] = {}
        self._benchmark_thread: Any | None = None
        self._benchmark_worker: Any | None = None
        self._benchmark_result: BenchmarkResult | None = None
        self._benchmark_cancelled = False
        self._paused = False
        self._streaming_requested = False
        self._current_page_key = "monitor"
        self._layout_slots = 4 if app_config.columns <= 2 else 9
        self._selected_camera_id = app_config.cameras[0].id if app_config.cameras else ""

        self.setWindowTitle(app_config.title)
        self.resize(1280, 780)
        logo_icon = Path("assets/logo.ico").resolve()
        if logo_icon.exists():
            self.setWindowIcon(QIcon(str(logo_icon)))

        self._build_ui()
        self._apply_styles()
        self._refresh_all_views()

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(self.performance_policy.render_interval_ms(self._layout_slots))
        self._render_timer.timeout.connect(self._render_latest_frames)
        self._render_timer.start()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_recording_status)
        self._status_timer.start()

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = self._build_sidebar()
        content = self._build_content()
        layout.addWidget(sidebar)
        layout.addWidget(content, stretch=1)

        self.setCentralWidget(root)

    def _build_sidebar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Sidebar")
        frame.setFixedWidth(210)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        logo = QLabel()
        logo.setObjectName("Logo")
        pixmap = self._load_logo_pixmap()
        if pixmap:
            logo.setPixmap(pixmap.scaledToWidth(150, Qt.SmoothTransformation))
            logo.setAlignment(Qt.AlignCenter)
        else:
            logo.setText("GESEC VMS")
            logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)

        menu_items = [
            ("monitor", "Monitor"),
            ("cameras", "Câmeras"),
            ("recording", "Gravação"),
            ("ptz", "PTZ"),
            ("events", "Eventos"),
            ("settings", "Configurações"),
            ("benchmark", "Benchmark"),
            ("about", "Sobre"),
        ]
        for key, label in menu_items:
            button = QPushButton(label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, page_key=key: self._show_page(page_key))
            layout.addWidget(button)
            self._nav_buttons[key] = button

        layout.addStretch(1)
        self.backend_badge = QLabel(f"{self.device_info.backend.upper()}")
        self.backend_badge.setObjectName("BackendBadge")
        self.backend_badge.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.backend_badge)
        return frame

    def _build_content(self) -> QWidget:
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = QHBoxLayout()
        self.title_label = QLabel(self.app_config.title)
        self.title_label.setObjectName("Title")
        self.status_label = QLabel("Pronto")
        self.status_label.setObjectName("StatusLabel")
        header.addWidget(self.title_label, stretch=1)
        header.addWidget(self.status_label)
        layout.addLayout(header)

        self.stack = QStackedWidget()
        self._pages["monitor"] = self._build_monitor_page()
        self._pages["cameras"] = self._build_cameras_page()
        self._pages["recording"] = self._build_recording_page()
        self._pages["ptz"] = self._build_ptz_page()
        self._pages["events"] = self._build_events_page()
        self._pages["settings"] = self._build_settings_page()
        self._pages["benchmark"] = self._build_benchmark_page()
        self._pages["about"] = self._build_about_page()
        for page in self._pages.values():
            self.stack.addWidget(page)
        layout.addWidget(self.stack, stretch=1)
        return frame

    def _build_monitor_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        toolbar = QHBoxLayout()
        self.start_button = QPushButton("Iniciar")
        self.pause_button = QPushButton("Pausar")
        self.stop_button = QPushButton("Parar")
        self.snapshot_button = QPushButton("Snapshot")
        self.fullscreen_button = QPushButton("Tela cheia")
        self.reconnect_button = QPushButton("Reconectar")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_streaming)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_streaming)
        self.snapshot_button.clicked.connect(self.capture_snapshot)
        self.fullscreen_button.clicked.connect(self.show_selected_fullscreen)
        self.reconnect_button.clicked.connect(self.reconnect_streams)

        self.monitor_camera_combo = QComboBox()
        self.monitor_camera_combo.currentIndexChanged.connect(self._select_camera_from_monitor_combo)
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("1 tela", 1)
        self.layout_combo.addItem("2 telas", 2)
        self.layout_combo.addItem("4 telas", 4)
        self.layout_combo.addItem("9 telas", 9)
        default_layout_index = self.layout_combo.findData(self._layout_slots)
        self.layout_combo.setCurrentIndex(default_layout_index if default_layout_index >= 0 else 2)
        self.layout_combo.currentIndexChanged.connect(self._change_layout_slots)

        for widget in (
            self.start_button,
            self.pause_button,
            self.stop_button,
            self.reconnect_button,
            self.snapshot_button,
            self.fullscreen_button,
        ):
            toolbar.addWidget(widget)
        toolbar.addStretch(1)
        toolbar.addWidget(QLabel("Câmera"))
        toolbar.addWidget(self.monitor_camera_combo)
        toolbar.addWidget(QLabel("Layout"))
        toolbar.addWidget(self.layout_combo)
        layout.addLayout(toolbar)

        self.grid_frame = QFrame()
        self.grid_frame.setObjectName("GridFrame")
        self.video_grid = QGridLayout(self.grid_frame)
        self.video_grid.setContentsMargins(0, 0, 0, 0)
        self.video_grid.setSpacing(10)
        layout.addWidget(self.grid_frame, stretch=1)
        return page

    def _build_cameras_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        left = QVBoxLayout()
        self.camera_list = QListWidget()
        self.camera_list.currentItemChanged.connect(self._load_camera_form_from_item)
        left.addWidget(QLabel("Câmeras cadastradas"))
        left.addWidget(self.camera_list, stretch=1)

        list_buttons = QHBoxLayout()
        self.new_camera_button = QPushButton("Nova")
        self.remove_camera_button = QPushButton("Remover")
        self.new_camera_button.clicked.connect(self.new_camera_form)
        self.remove_camera_button.clicked.connect(self.remove_selected_camera)
        list_buttons.addWidget(self.new_camera_button)
        list_buttons.addWidget(self.remove_camera_button)
        left.addLayout(list_buttons)
        layout.addLayout(left, stretch=1)

        form_frame = QFrame()
        form_frame.setObjectName("Panel")
        form_layout_outer = QVBoxLayout(form_frame)
        form_layout_outer.setContentsMargins(16, 16, 16, 16)
        form_layout_outer.setSpacing(12)
        form_layout_outer.addWidget(QLabel("Cadastro e protocolo da câmera"))

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.camera_id_input = QLineEdit()
        self.camera_name_input = QLineEdit()
        self.camera_type_combo = QComboBox()
        self.camera_type_combo.addItems(["synthetic", "rtsp", "file", "device"])
        self.camera_protocol_combo = QComboBox()
        self.camera_protocol_combo.addItems(["synthetic", "rtsp", "onvif", "file", "device"])
        self.camera_url_input = QLineEdit()
        self.camera_active_check = QCheckBox("Ativa")
        self.camera_loop_check = QCheckBox("Loop")
        self.camera_fps_input = QDoubleSpinBox()
        self.camera_fps_input.setRange(1.0, 120.0)
        self.camera_fps_input.setDecimals(1)
        self.camera_width_input = QSpinBox()
        self.camera_width_input.setRange(160, 7680)
        self.camera_height_input = QSpinBox()
        self.camera_height_input.setRange(90, 4320)
        self.onvif_host_input = QLineEdit()
        self.onvif_port_input = QSpinBox()
        self.onvif_port_input.setRange(1, 65535)
        self.onvif_user_input = QLineEdit()
        self.onvif_password_input = QLineEdit()
        self.onvif_password_input.setEchoMode(QLineEdit.Password)
        self.ptz_enabled_check = QCheckBox("Habilitar PTZ")

        form.addRow("ID", self.camera_id_input)
        form.addRow("Nome", self.camera_name_input)
        form.addRow("Tipo", self.camera_type_combo)
        form.addRow("Protocolo", self.camera_protocol_combo)
        form.addRow("URL RTSP/Origem", self.camera_url_input)
        form.addRow("Ativa", self.camera_active_check)
        form.addRow("Loop", self.camera_loop_check)
        form.addRow("FPS alvo", self.camera_fps_input)
        form.addRow("Largura", self.camera_width_input)
        form.addRow("Altura", self.camera_height_input)
        form.addRow("ONVIF host", self.onvif_host_input)
        form.addRow("ONVIF porta", self.onvif_port_input)
        form.addRow("Usuário", self.onvif_user_input)
        form.addRow("Senha", self.onvif_password_input)
        form.addRow("PTZ", self.ptz_enabled_check)
        form_layout_outer.addLayout(form)

        form_buttons = QHBoxLayout()
        self.test_camera_button = QPushButton("Testar conexão")
        self.save_camera_button = QPushButton("Salvar câmera")
        self.test_camera_button.clicked.connect(self.test_camera_form)
        self.save_camera_button.clicked.connect(self.save_camera_form)
        form_buttons.addStretch(1)
        form_buttons.addWidget(self.test_camera_button)
        form_buttons.addWidget(self.save_camera_button)
        form_layout_outer.addLayout(form_buttons)
        layout.addWidget(form_frame, stretch=2)
        return page

    def _build_recording_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Gravação local")
        title.setObjectName("SectionTitle")
        self.storage_label = QLabel("")
        self.storage_label.setObjectName("MutedLabel")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.storage_label)
        layout.addLayout(header)

        self.recording_rows = QVBoxLayout()
        self.recording_rows.setSpacing(10)
        layout.addLayout(self.recording_rows)
        layout.addStretch(1)
        return page

    def _build_ptz_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        header = QHBoxLayout()
        header.addWidget(QLabel("Câmera PTZ"))
        self.ptz_camera_combo = QComboBox()
        self.ptz_camera_combo.currentIndexChanged.connect(self._refresh_ptz_state)
        header.addWidget(self.ptz_camera_combo)
        header.addStretch(1)
        self.ptz_state_label = QLabel("")
        self.ptz_state_label.setObjectName("MutedLabel")
        header.addWidget(self.ptz_state_label)
        layout.addLayout(header)

        controls = QFrame()
        controls.setObjectName("Panel")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setSpacing(12)

        self.ptz_speed_combo = QComboBox()
        self.ptz_speed_combo.addItem("Velocidade baixa", 0.25)
        self.ptz_speed_combo.addItem("Velocidade média", 0.45)
        self.ptz_speed_combo.addItem("Velocidade alta", 0.75)

        arrows = QGridLayout()
        self.ptz_buttons: list[QPushButton] = []
        ptz_specs = [
            ("NW", 0, 0, -1, 1, 0),
            ("UP", 0, 1, 0, 1, 0),
            ("NE", 0, 2, 1, 1, 0),
            ("LEFT", 1, 0, -1, 0, 0),
            ("STOP", 1, 1, 0, 0, 0),
            ("RIGHT", 1, 2, 1, 0, 0),
            ("SW", 2, 0, -1, -1, 0),
            ("DOWN", 2, 1, 0, -1, 0),
            ("SE", 2, 2, 1, -1, 0),
            ("Zoom -", 3, 0, 0, 0, -1),
            ("Zoom +", 3, 2, 0, 0, 1),
        ]
        for label, row, column, pan, tilt, zoom in ptz_specs:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, p=pan, t=tilt, z=zoom: self.send_ptz(p, t, z))
            arrows.addWidget(button, row, column)
            self.ptz_buttons.append(button)

        preset_row = QHBoxLayout()
        self.preset_input = QLineEdit()
        self.preset_input.setPlaceholderText("preset-1")
        self.save_preset_button = QPushButton("Salvar preset")
        self.goto_preset_button = QPushButton("Chamar preset")
        self.save_preset_button.clicked.connect(self.save_ptz_preset)
        self.goto_preset_button.clicked.connect(self.goto_ptz_preset)
        preset_row.addWidget(self.preset_input)
        preset_row.addWidget(self.save_preset_button)
        preset_row.addWidget(self.goto_preset_button)
        self.ptz_buttons.extend([self.save_preset_button, self.goto_preset_button])

        controls_layout.addWidget(self.ptz_speed_combo)
        controls_layout.addLayout(arrows)
        controls_layout.addLayout(preset_row)
        layout.addWidget(controls)
        layout.addStretch(1)
        return page

    def _build_events_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.events_list = QListWidget()
        layout.addWidget(QLabel("Eventos do VMS"))
        layout.addWidget(self.events_list, stretch=1)
        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        panel = QFrame()
        panel.setObjectName("Panel")
        form = QFormLayout(panel)
        form.setContentsMargins(18, 18, 18, 18)
        self.config_path_label = QLabel("")
        self.backend_label = QLabel(f"{self.device_info.backend.upper()} | {self.device_info.reason}")
        self.recording_path_label = QLabel(str(self.recording_manager.root))
        form.addRow("Configuração ativa", self.config_path_label)
        form.addRow("Backend", self.backend_label)
        form.addRow("Pasta de gravação", self.recording_path_label)
        form.addRow("OpenCV", QLabel(self.device_info.opencv_version))
        layout.addWidget(panel)
        layout.addStretch(1)
        return page

    def _build_benchmark_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Benchmark de desempenho")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        note = QLabel(
            "Teste sintético local usando o mesmo processamento CPU/GPU do VMS. "
            "Ele não mede rede RTSP nem decodificação NVDEC real."
        )
        note.setObjectName("MutedLabel")
        note.setWordWrap(True)
        layout.addWidget(note)

        controls = QFrame()
        controls.setObjectName("Panel")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setSpacing(12)

        form = QFormLayout()
        self.benchmark_backend_combo = QComboBox()
        for backend in ("auto", "cpu", "cuda", "opencl", "torch"):
            self.benchmark_backend_combo.addItem(backend, backend)
        self.benchmark_backend_combo.setCurrentText(self.device_info.requested)

        self.benchmark_profile_combo = QComboBox()
        for profile in BENCHMARK_PROFILES.values():
            self.benchmark_profile_combo.addItem(profile.name, profile.key)

        self.benchmark_duration_input = QSpinBox()
        self.benchmark_duration_input.setRange(3, 60)
        self.benchmark_duration_input.setValue(10)
        self.benchmark_duration_input.setSuffix(" s")

        form.addRow("Backend", self.benchmark_backend_combo)
        form.addRow("Perfil", self.benchmark_profile_combo)
        form.addRow("Duração", self.benchmark_duration_input)
        controls_layout.addLayout(form)

        actions = QHBoxLayout()
        self.benchmark_start_button = QPushButton("Iniciar benchmark")
        self.benchmark_stop_button = QPushButton("Parar")
        self.benchmark_save_button = QPushButton("Salvar relatório")
        self.benchmark_stop_button.setEnabled(False)
        self.benchmark_save_button.setEnabled(False)
        self.benchmark_start_button.clicked.connect(self.start_benchmark)
        self.benchmark_stop_button.clicked.connect(self.stop_benchmark)
        self.benchmark_save_button.clicked.connect(self.save_benchmark_report)
        for button in (self.benchmark_start_button, self.benchmark_stop_button, self.benchmark_save_button):
            actions.addWidget(button)
        actions.addStretch(1)
        controls_layout.addLayout(actions)

        self.benchmark_progress = QProgressBar()
        self.benchmark_progress.setRange(0, 1000)
        self.benchmark_progress.setValue(0)
        controls_layout.addWidget(self.benchmark_progress)
        layout.addWidget(controls)

        results = QFrame()
        results.setObjectName("Panel")
        results_form = QFormLayout(results)
        results_form.setContentsMargins(18, 18, 18, 18)
        self.benchmark_status_label = QLabel("Pronto para medir.")
        self.benchmark_status_label.setWordWrap(True)
        self.benchmark_result_labels: dict[str, QLabel] = {}
        for key, label in (
            ("verdict", "Veredito"),
            ("backend", "Backend usado"),
            ("total_fps", "FPS total"),
            ("fps_per_camera", "FPS por câmera"),
            ("latency_avg", "Latência média"),
            ("latency_p95", "Latência p95"),
            ("elapsed", "Tempo total"),
            ("frames", "Frames processados"),
            ("target", "Meta do perfil"),
        ):
            value = QLabel("-")
            value.setWordWrap(True)
            self.benchmark_result_labels[key] = value
            results_form.addRow(label, value)
        results_form.addRow("Estado", self.benchmark_status_label)
        layout.addWidget(results)
        layout.addStretch(1)
        return page

    def _build_about_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop)
        logo = QLabel()
        logo.setAlignment(Qt.AlignLeft)
        pixmap = self._load_logo_pixmap()
        if pixmap:
            logo.setPixmap(pixmap.scaledToWidth(190, Qt.SmoothTransformation))
        else:
            logo.setText("GESEC VMS")
            logo.setObjectName("Title")
        layout.addWidget(logo)
        layout.addWidget(QLabel("Video Management System desktop com RTSP, ONVIF, PTZ e gravação local."))
        layout.addWidget(QLabel("Projeto GESEC Mini VMS Viewer evoluído para operação completa."))
        return page

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #111615;
                color: #eef4ef;
                font-family: "Segoe UI", "Ubuntu", sans-serif;
                font-size: 12px;
            }
            QFrame#Sidebar {
                background: #171d1b;
                border-right: 1px solid #2c3733;
            }
            QLabel#Logo {
                min-height: 64px;
                font-size: 20px;
                font-weight: 800;
            }
            QLabel#Title {
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#SectionTitle {
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#StatusLabel, QLabel#MutedLabel {
                color: #aebbb3;
            }
            QLabel#BackendBadge {
                color: #dff4e8;
                background: #214d3a;
                border: 1px solid #3e765d;
                border-radius: 6px;
                padding: 8px;
                font-weight: 700;
            }
            QPushButton {
                border: 1px solid #3d4a45;
                border-radius: 6px;
                padding: 8px 14px;
                background: #202824;
                color: #eef4ef;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #2b3731;
            }
            QPushButton:disabled {
                color: #67706d;
                border-color: #2b322f;
                background: #171c1a;
            }
            QPushButton#NavButton {
                text-align: left;
                padding: 11px 12px;
                border: 1px solid transparent;
                background: transparent;
            }
            QPushButton#NavButton:checked {
                background: #1f4f72;
                border-color: #3379a8;
            }
            QFrame#GridFrame, QFrame#Panel {
                background: #0f1312;
                border: 1px solid #27302c;
                border-radius: 8px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {
                background: #161d1a;
                border: 1px solid #33413b;
                border-radius: 6px;
                padding: 7px;
                selection-background-color: #256b96;
            }
            QCheckBox {
                spacing: 8px;
            }
            """
        )

    def start_streaming(self) -> None:
        cameras = self._capture_cameras()
        if not cameras:
            self._add_event("Nenhuma câmera ativa para iniciar.")
            return

        self._streaming_requested = True
        self._paused = False
        self.pause_button.setText("Pausar")
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self._restart_capture_workers()

    def _restart_capture_workers(self) -> None:
        if not self._streaming_requested:
            return
        self._stop_capture_workers(clear_frames=False)
        cameras = self._capture_cameras()
        if not cameras:
            return
        self._start_capture_workers(cameras)
        self._add_event(f"Streaming ativo para {len(cameras)} câmera(s) necessária(s).")

    def _start_capture_workers(self, cameras: list[CameraConfig]) -> None:
        from PyQt5.QtCore import QThread

        self._stream_aliases.clear()
        self._stream_sources.clear()
        for camera, aliases in self._capture_plan(cameras):
            existing_thread = self._threads.get(camera.id)
            if existing_thread is not None and existing_thread.isRunning():
                self._add_event(f"Câmera {camera.name} ainda está liberando o backend de vídeo.")
                continue

            source_key = (camera.type, camera.url)
            if aliases:
                self._stream_aliases[camera.id] = aliases
                for alias_id in aliases:
                    self._on_status(alias_id, f"Espelhando {camera.name}")
            effective_camera = self.performance_policy.adapt_camera(camera, self._layout_slots)
            worker = CameraWorker(
                effective_camera,
                self.device_info,
                self.app_config.reconnect_delay_seconds,
                self.hardware_decode_mode,
            )
            thread = QThread(self)
            worker.moveToThread(thread)
            generation = self._worker_generations.get(camera.id, 0) + 1
            self._worker_generations[camera.id] = generation

            thread.started.connect(worker.run)
            worker.frame_ready.connect(self._on_frame)
            worker.status_changed.connect(self._on_status)
            worker.log_message.connect(self._on_log_message)
            worker.finished.connect(lambda camera_id, generation=generation: self._on_worker_finished(camera_id, generation))
            worker.finished.connect(lambda camera_id, thread=thread: thread.quit())
            worker.finished.connect(lambda camera_id, worker=worker: worker.deleteLater())
            thread.finished.connect(thread.deleteLater)

            self._workers[camera.id] = worker
            self._threads[camera.id] = thread
            self._stream_sources[camera.id] = source_key
            thread.start()

    def _capture_plan(self, cameras: list[CameraConfig]) -> list[tuple[CameraConfig, list[str]]]:
        stream_owners: dict[tuple[str, str], int] = {}
        plan: list[tuple[CameraConfig, list[str]]] = []
        for camera in cameras:
            source_key = (camera.type, camera.url)
            owner_index = stream_owners.get(source_key)
            if owner_index is not None and camera.type == "rtsp":
                plan[owner_index][1].append(camera.id)
                continue
            stream_owners[source_key] = len(plan)
            plan.append((camera, []))
        return plan

    def toggle_pause(self) -> None:
        if not self._workers:
            return

        self._paused = not self._paused
        self.pause_button.setText("Continuar" if self._paused else "Pausar")
        for worker in self._workers.values():
            if self._paused:
                worker.pause()
            else:
                worker.resume()
        self._add_event("Streaming pausado." if self._paused else "Streaming retomado.")

    def stop_streaming(self) -> None:
        had_activity = bool(self._workers) or bool(self.recording_manager.active_states())
        for state in self.recording_manager.stop_all(drain=False):
            self._add_event(f"Gravação encerrada: {state.path}")

        self._streaming_requested = False
        self._stop_capture_workers(clear_frames=True)
        for tile in self._tiles.values():
            tile.set_status("Parado")
            tile.set_recording(False)

        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause_button.setText("Pausar")
        self._refresh_recording_rows()
        if had_activity:
            self._add_event("Streaming parado.")

    def _stop_capture_workers(self, clear_frames: bool) -> None:
        for worker in list(self._workers.values()):
            worker.stop()

        still_running: list[str] = []
        for camera_id, thread in list(self._threads.items()):
            if thread.wait(150):
                self._workers.pop(camera_id, None)
                self._threads.pop(camera_id, None)
                continue

            thread.quit()
            if thread.wait(150):
                self._workers.pop(camera_id, None)
                self._threads.pop(camera_id, None)
                continue

            still_running.append(camera_id)
            self._add_event(f"Thread da câmera {camera_id} ainda aguardando liberação do backend de vídeo.")

        if not still_running:
            self._workers.clear()
            self._threads.clear()

        self._stream_aliases.clear()
        self._stream_sources.clear()
        if clear_frames:
            self._latest_frames.clear()
            self._last_rendered_sequence.clear()

    def _on_worker_finished(self, camera_id: str, generation: int) -> None:
        if self._worker_generations.get(camera_id) != generation:
            return

        self._workers.pop(camera_id, None)
        self._threads.pop(camera_id, None)
        self._camera_status[camera_id] = "Parado"
        tile = self._tiles.get(camera_id)
        if tile:
            tile.set_status("Parado")

    def reconnect_streams(self) -> None:
        was_running = bool(self._workers)
        if was_running:
            self._add_event("Reconectando streams visíveis.")
            self._restart_capture_workers()
        else:
            self.start_streaming()

    def capture_snapshot(self) -> None:
        camera = self._selected_camera()
        packet = self._latest_frames.get(camera.id) if camera else None
        if not camera or not packet:
            self._add_event("Snapshot indisponível: selecione uma câmera com frame ativo.")
            return
        try:
            path = self.recording_manager.save_snapshot(camera, packet.frame)
        except Exception as exc:
            logger.exception("Falha ao salvar snapshot")
            self._add_event(f"Falha ao salvar snapshot: {exc}")
            return
        self._add_event(f"Snapshot salvo: {path}")

    def show_selected_fullscreen(self) -> None:
        camera = self._selected_camera()
        packet = self._latest_frames.get(camera.id) if camera else None
        if not camera or not packet:
            self._add_event("Tela cheia indisponível: sem frame ativo.")
            return

        frame = packet.frame
        height, width, channels = frame.shape
        image = QImage(frame.data, width, height, channels * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)

        dialog = QDialog(self)
        dialog.setWindowTitle(camera.name)
        layout = QVBoxLayout(dialog)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        screen = QApplication.primaryScreen().availableGeometry().size()
        label.setPixmap(pixmap.scaled(screen, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(label)
        dialog.showFullScreen()
        dialog.exec_()

    def toggle_recording(self, camera_id: str) -> None:
        camera = self._camera_by_id(camera_id)
        if camera is None:
            return
        if self.recording_manager.is_recording(camera_id):
            state = self.recording_manager.stop(camera_id, drain=False)
            if state:
                self._add_event(f"Gravação encerrada: {state.path}")
        else:
            state = self.recording_manager.start(camera)
            self._add_event(f"Gravação iniciada: {state.path}")
        self._refresh_recording_rows()
        self._refresh_tile_recording(camera_id)
        if self._streaming_requested:
            self._restart_capture_workers()
        elif self.recording_manager.is_recording(camera_id):
            self.start_streaming()

    def start_benchmark(self) -> None:
        from PyQt5.QtCore import QThread

        if self._benchmark_thread is not None and self._benchmark_thread.isRunning():
            return

        profile_key = str(self.benchmark_profile_combo.currentData() or "2")
        profile = BENCHMARK_PROFILES[profile_key]
        backend = str(self.benchmark_backend_combo.currentData() or "auto")
        duration = float(self.benchmark_duration_input.value())
        worker = BenchmarkWorker(profile, backend, duration)
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_benchmark_progress)
        worker.finished.connect(self._on_benchmark_finished)
        worker.failed.connect(self._on_benchmark_failed)
        worker.finished.connect(lambda result, thread=thread: thread.quit())
        worker.failed.connect(lambda message, thread=thread: thread.quit())
        worker.finished.connect(lambda result, worker=worker: worker.deleteLater())
        worker.failed.connect(lambda message, worker=worker: worker.deleteLater())
        thread.finished.connect(self._on_benchmark_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._benchmark_worker = worker
        self._benchmark_thread = thread
        self._benchmark_result = None
        self._benchmark_cancelled = False
        self._set_benchmark_running(True)
        self.benchmark_save_button.setEnabled(False)
        self.benchmark_progress.setValue(0)
        self._set_benchmark_result_text("verdict", "Executando")
        self._set_benchmark_result_text("backend", backend)
        self._set_benchmark_result_text("target", f"{profile.target_total_fps:.1f} FPS total")
        self.benchmark_status_label.setText("Benchmark em execução...")
        self._add_event(f"Benchmark iniciado: {profile.name} em backend {backend}.")
        thread.start()

    def stop_benchmark(self) -> None:
        self._stop_benchmark(cancelled=True)

    def save_benchmark_report(self) -> None:
        if self._benchmark_result is None:
            QMessageBox.information(self, "Benchmark", "Execute um benchmark antes de salvar o relatório.")
            return
        try:
            path = save_benchmark_report(self._benchmark_result)
        except Exception as exc:
            logger.exception("Falha ao salvar relatório de benchmark")
            self._add_event(f"Falha ao salvar relatório de benchmark: {exc}")
            return
        self._add_event(f"Relatório de benchmark salvo: {path}")

    def _stop_benchmark(self, cancelled: bool, wait_ms: int = 1500) -> None:
        worker = self._benchmark_worker
        thread = self._benchmark_thread
        if worker is None and thread is None:
            return

        if worker is not None:
            self._benchmark_cancelled = cancelled
            worker.stop()
        if thread is not None:
            thread.quit()
            thread.wait(wait_ms)
            if thread.isRunning():
                logger.warning("Benchmark thread did not stop within %sms", wait_ms)
                return

        self._benchmark_worker = None
        self._benchmark_thread = None
        self._set_benchmark_running(False)
        if cancelled:
            self.benchmark_status_label.setText("Benchmark cancelado.")
            self._add_event("Benchmark cancelado.")

    def _on_benchmark_progress(self, progress: Any) -> None:
        self.benchmark_progress.setValue(int(progress.percent * 10))
        self._set_benchmark_result_text("backend", progress.backend)
        self._set_benchmark_result_text("total_fps", f"{progress.total_fps:.2f}")
        self._set_benchmark_result_text("fps_per_camera", f"{progress.fps_per_camera:.2f}")
        self._set_benchmark_result_text("elapsed", f"{progress.elapsed_seconds:.1f}s")
        self._set_benchmark_result_text("frames", str(progress.frames))
        self.benchmark_status_label.setText(f"Executando... {progress.percent:.0f}%")

    def _on_benchmark_finished(self, result: BenchmarkResult) -> None:
        if self._benchmark_cancelled:
            return
        self._benchmark_result = result
        self.benchmark_progress.setValue(1000)
        self._set_benchmark_result_text("verdict", result.verdict)
        self._set_benchmark_result_text("backend", result.selected_backend)
        self._set_benchmark_result_text("total_fps", f"{result.total_fps:.2f}")
        self._set_benchmark_result_text("fps_per_camera", f"{result.fps_per_camera:.2f}")
        self._set_benchmark_result_text("latency_avg", f"{result.latency_avg_ms:.2f} ms")
        self._set_benchmark_result_text("latency_p95", f"{result.latency_p95_ms:.2f} ms")
        self._set_benchmark_result_text("elapsed", f"{result.elapsed_seconds:.2f}s")
        self._set_benchmark_result_text("frames", str(result.frames))
        self._set_benchmark_result_text("target", f"{result.profile.target_total_fps:.1f} FPS total")
        self.benchmark_status_label.setText(
            f"Concluído: {result.verdict}. Teste sintético local, sem medir rede RTSP."
        )
        self.benchmark_save_button.setEnabled(True)
        self._add_event(f"Benchmark concluído: {result.verdict}, {result.total_fps:.2f} FPS total.")

    def _on_benchmark_failed(self, message: str) -> None:
        if self._benchmark_cancelled:
            return
        self.benchmark_status_label.setText(f"Falha no benchmark: {message}")
        self._add_event(f"Falha no benchmark: {message}")

    def _on_benchmark_thread_finished(self) -> None:
        self._benchmark_worker = None
        self._benchmark_thread = None
        self._set_benchmark_running(False)

    def _set_benchmark_running(self, running: bool) -> None:
        self.benchmark_start_button.setEnabled(not running)
        self.benchmark_stop_button.setEnabled(running)
        self.benchmark_backend_combo.setEnabled(not running)
        self.benchmark_profile_combo.setEnabled(not running)
        self.benchmark_duration_input.setEnabled(not running)

    def _set_benchmark_result_text(self, key: str, value: str) -> None:
        label = self.benchmark_result_labels.get(key)
        if label is not None:
            label.setText(value)

    def new_camera_form(self) -> None:
        index = len(self.app_config.cameras) + 1
        self._load_camera_form(
            CameraConfig(
                id=f"camera_{index}",
                name=f"Câmera {index}",
                type="rtsp",
                protocol="rtsp",
                url="rtsp://usuario:senha@192.168.0.10:554/stream1",
                loop=False,
                active=True,
                render_size=self.app_config.render_size,
            )
        )
        self.camera_list.clearSelection()

    def save_camera_form(self) -> None:
        try:
            camera = self._camera_from_form()
        except ValueError as exc:
            QMessageBox.warning(self, "Câmera inválida", str(exc))
            return

        if self._workers:
            self.stop_streaming()
        updated = self.camera_store.add_or_update_camera(self.app_config, camera)
        self.app_config = self.camera_store.save(updated)
        self._selected_camera_id = camera.id
        self._refresh_all_views()
        self._add_event(f"Câmera salva: {camera.name}")

    def remove_selected_camera(self) -> None:
        item = self.camera_list.currentItem()
        if item is None:
            return
        if len(self.app_config.cameras) <= 1:
            QMessageBox.warning(self, "Remover câmera", "Mantenha pelo menos uma câmera cadastrada.")
            return
        camera_id = item.data(Qt.UserRole)
        camera = self._camera_by_id(camera_id)
        if camera is None:
            return
        if self._workers:
            self.stop_streaming()
        updated = self.camera_store.remove_camera(self.app_config, camera_id)
        self.app_config = self.camera_store.save(updated)
        self._selected_camera_id = self.app_config.cameras[0].id if self.app_config.cameras else ""
        self._refresh_all_views()
        self._add_event(f"Câmera removida: {camera.name}")

    def test_camera_form(self) -> None:
        try:
            camera = self._camera_from_form()
        except ValueError as exc:
            QMessageBox.warning(self, "Teste de conexão", str(exc))
            return

        if camera.type == "synthetic":
            message = "Fonte sintética pronta."
        elif camera.type == "rtsp":
            message = "RTSP válido. A conexão real será feita ao iniciar o monitor."
        elif camera.type in {"file", "video"}:
            message = "Arquivo informado existe." if Path(camera.url).exists() else "Arquivo não encontrado."
        elif camera.type in {"device", "webcam"}:
            message = "Dispositivo informado. A abertura real será feita ao iniciar."
        else:
            message = "Configuração validada."

        if camera.ptz_enabled or camera.onvif_host:
            result = self.ptz_controller.probe(camera)
            message = f"{message} ONVIF: {result.message}"
        QMessageBox.information(self, "Teste de conexão", message)
        self._add_event(f"Teste de câmera '{camera.name}': {message}")

    def send_ptz(self, pan: float, tilt: float, zoom: float) -> None:
        camera = self._ptz_camera()
        if camera is None:
            self._add_event("PTZ sem câmera selecionada.")
            return
        speed = float(self.ptz_speed_combo.currentData())
        result = self.ptz_controller.nudge(camera, pan=pan, tilt=tilt, zoom=zoom, speed=speed)
        self._add_event(result.message)
        self._refresh_ptz_state()

    def save_ptz_preset(self) -> None:
        camera = self._ptz_camera()
        if camera is None:
            return
        result = self.ptz_controller.set_preset(camera, self.preset_input.text())
        self._add_event(result.message)

    def goto_ptz_preset(self) -> None:
        camera = self._ptz_camera()
        if camera is None:
            return
        result = self.ptz_controller.goto_preset(camera, self.preset_input.text())
        self._add_event(result.message)

    def closeEvent(self, event: Any) -> None:
        self.stop_streaming()
        self._stop_benchmark(cancelled=False)
        event.accept()

    def _on_frame(self, packet: FramePacket) -> None:
        self._store_frame(packet)
        for alias_id in self._stream_aliases.get(packet.camera_id, []):
            alias_camera = self._camera_by_id(alias_id)
            if alias_camera is None:
                continue
            self._store_frame(
                FramePacket(
                    camera_id=alias_camera.id,
                    camera_name=alias_camera.name,
                    sequence=packet.sequence,
                    frame=packet.frame,
                    captured_at=packet.captured_at,
                    processed_at=packet.processed_at,
                    backend=packet.backend,
                )
            )

    def _store_frame(self, packet: FramePacket) -> None:
        self._latest_frames[packet.camera_id] = packet
        try:
            self.recording_manager.write_frame(packet)
        except Exception as exc:
            logger.exception("Falha na gravação de %s", packet.camera_name)
            self._add_event(f"Falha na gravação de {packet.camera_name}: {exc}")
            self.recording_manager.stop(packet.camera_id)
            self._refresh_recording_rows()

    def _on_status(self, camera_id: str, status: str) -> None:
        self._camera_status[camera_id] = status
        tile = self._tiles.get(camera_id)
        if tile:
            tile.set_status(status)
        for alias_id in self._stream_aliases.get(camera_id, []):
            alias_status = status if status != "Rodando" else "Rodando"
            self._camera_status[alias_id] = alias_status
            alias_tile = self._tiles.get(alias_id)
            if alias_tile:
                alias_tile.set_status(alias_status)
        self._refresh_recording_status()

    def _on_log_message(self, message: str) -> None:
        logger.info(message)

    def _render_latest_frames(self) -> None:
        if self._current_page_key != "monitor":
            return
        for camera_id, packet in list(self._latest_frames.items()):
            if self._last_rendered_sequence.get(camera_id) == packet.sequence:
                continue
            tile = self._tiles.get(camera_id)
            if tile is None:
                continue
            tile.set_frame(packet)
            tile.set_recording(self.recording_manager.is_recording(camera_id))
            self._last_rendered_sequence[camera_id] = packet.sequence

    def _show_page(self, key: str) -> None:
        page = self._pages.get(key)
        if page is None:
            return
        self.stack.setCurrentWidget(page)
        for nav_key, button in self._nav_buttons.items():
            button.setChecked(nav_key == key)
        self._current_page_key = key
        page_names = {
            "monitor": "Monitor",
            "cameras": "Câmeras",
            "recording": "Gravação",
            "ptz": "PTZ",
            "events": "Eventos",
            "settings": "Configurações",
            "benchmark": "Benchmark",
            "about": "Sobre",
        }
        self.status_label.setText(page_names.get(key, key.capitalize()))

    def _refresh_all_views(self) -> None:
        self.title_label.setText(self.app_config.title)
        self.config_path_label.setText(self.app_config.source_path or str(self.camera_store.user_path))
        self._refresh_camera_list()
        self._refresh_camera_selectors()
        self._rebuild_video_grid()
        self._refresh_recording_rows()
        self._refresh_ptz_state()
        self._update_storage_label()
        self._show_page("monitor")

    def _refresh_camera_list(self) -> None:
        current = self._selected_camera_id
        self.camera_list.blockSignals(True)
        self.camera_list.clear()
        for camera in self.app_config.cameras:
            label = f"{camera.name} ({camera.type})"
            if not camera.active:
                label += " - inativa"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, camera.id)
            self.camera_list.addItem(item)
            if camera.id == current:
                self.camera_list.setCurrentItem(item)
        self.camera_list.blockSignals(False)
        selected = self._selected_camera() or (self.app_config.cameras[0] if self.app_config.cameras else None)
        if selected:
            self._load_camera_form(selected)

    def _refresh_camera_selectors(self) -> None:
        selectors = [self.monitor_camera_combo, self.ptz_camera_combo]
        for selector in selectors:
            selector.blockSignals(True)
            selector.clear()
            for camera in self.app_config.cameras:
                selector.addItem(camera.name, camera.id)
                if camera.id == self._selected_camera_id:
                    selector.setCurrentIndex(selector.count() - 1)
            selector.blockSignals(False)

    def _rebuild_video_grid(self) -> None:
        self._clear_layout(self.video_grid)
        self._tiles.clear()

        cameras = self._visible_cameras()
        columns = 1 if self._layout_slots == 1 else 2 if self._layout_slots in {2, 4} else 3
        for index, camera in enumerate(cameras):
            tile = VideoTile(camera.id, camera.name)
            tile.set_selected(camera.id == self._selected_camera_id)
            tile.set_recording(self.recording_manager.is_recording(camera.id))
            tile.clicked.connect(self._select_camera)
            tile.double_clicked.connect(lambda camera_id: (self._select_camera(camera_id), self.show_selected_fullscreen()))
            self._tiles[camera.id] = tile
            self.video_grid.addWidget(tile, index // columns, index % columns)

    def _refresh_recording_rows(self) -> None:
        self._clear_layout(self.recording_rows)
        self._recording_buttons.clear()
        self._recording_labels.clear()
        for camera in self._active_cameras():
            row = QFrame()
            row.setObjectName("Panel")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(12, 10, 12, 10)
            name = QLabel(camera.name)
            status = QLabel()
            status.setObjectName("MutedLabel")
            button = QPushButton()
            button.clicked.connect(lambda checked=False, camera_id=camera.id: self.toggle_recording(camera_id))
            row_layout.addWidget(name, stretch=1)
            row_layout.addWidget(status, stretch=2)
            row_layout.addWidget(button)
            self.recording_rows.addWidget(row)
            self._recording_buttons[camera.id] = button
            self._recording_labels[camera.id] = status
        self._refresh_recording_status()
        self._update_storage_label()

    def _refresh_recording_status(self) -> None:
        for camera in self._active_cameras():
            state = next((item for item in self.recording_manager.active_states() if item.camera_id == camera.id), None)
            button = self._recording_buttons.get(camera.id)
            label = self._recording_labels.get(camera.id)
            if not button or not label:
                continue
            if state:
                button.setText("Parar")
                label.setText(f"REC {state.duration_seconds:05.1f}s | {state.frame_count} frames | {state.path}")
            else:
                button.setText("Gravar")
                label.setText(self._camera_status.get(camera.id, "Aguardando"))

    def _refresh_tile_recording(self, camera_id: str) -> None:
        tile = self._tiles.get(camera_id)
        if tile:
            tile.set_recording(self.recording_manager.is_recording(camera_id))

    def _refresh_ptz_state(self) -> None:
        camera = self._ptz_camera()
        enabled = bool(camera and self.ptz_controller.is_configured(camera))
        for button in self.ptz_buttons:
            button.setEnabled(enabled)
        self.ptz_state_label.setText("ONVIF/PTZ pronto" if enabled else "Configure ONVIF e habilite PTZ no cadastro")

    def _update_storage_label(self) -> None:
        if not hasattr(self, "storage_label"):
            return
        size_mb = _directory_size(self.recording_manager.root) / (1024 * 1024)
        self.storage_label.setText(f"{self.recording_manager.root} | {size_mb:.1f} MB usados")

    def _load_camera_form_from_item(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        camera = self._camera_by_id(current.data(Qt.UserRole))
        if camera:
            self._select_camera(camera.id)
            self._load_camera_form(camera)

    def _load_camera_form(self, camera: CameraConfig) -> None:
        self.camera_id_input.setText(camera.id)
        self.camera_name_input.setText(camera.name)
        self.camera_type_combo.setCurrentText(camera.type)
        self.camera_protocol_combo.setCurrentText(camera.protocol)
        self.camera_url_input.setText(camera.url)
        self.camera_active_check.setChecked(camera.active)
        self.camera_loop_check.setChecked(camera.loop)
        self.camera_fps_input.setValue(camera.target_fps)
        self.camera_width_input.setValue(camera.render_size[0])
        self.camera_height_input.setValue(camera.render_size[1])
        self.onvif_host_input.setText(camera.onvif_host)
        self.onvif_port_input.setValue(camera.onvif_port)
        self.onvif_user_input.setText(camera.username)
        self.onvif_password_input.setText(camera.password)
        self.ptz_enabled_check.setChecked(camera.ptz_enabled)

    def _camera_from_form(self) -> CameraConfig:
        camera_id = _slugify(self.camera_id_input.text() or self.camera_name_input.text())
        name = self.camera_name_input.text().strip() or camera_id
        source_type = self.camera_type_combo.currentText().strip().lower()
        protocol = self.camera_protocol_combo.currentText().strip().lower()
        url = self.camera_url_input.text().strip()
        onvif_host = self.onvif_host_input.text().strip()

        if not camera_id:
            raise ValueError("Informe ID ou nome da câmera.")
        if source_type == "rtsp" and not url.lower().startswith("rtsp://"):
            raise ValueError("Câmera RTSP precisa de URL iniciando com rtsp://.")
        if source_type in {"file", "video"} and not url:
            raise ValueError("Câmera de arquivo precisa de caminho de vídeo.")
        if self.ptz_enabled_check.isChecked() and not onvif_host:
            raise ValueError("PTZ exige host ONVIF.")
        if source_type == "synthetic" and not url:
            url = f"synthetic://{camera_id}"

        return CameraConfig(
            id=camera_id,
            name=name,
            url=url,
            type=source_type,
            protocol=protocol,
            onvif_host=onvif_host,
            onvif_port=int(self.onvif_port_input.value()),
            username=self.onvif_user_input.text().strip(),
            password=self.onvif_password_input.text(),
            ptz_enabled=self.ptz_enabled_check.isChecked(),
            active=self.camera_active_check.isChecked(),
            loop=self.camera_loop_check.isChecked(),
            target_fps=float(self.camera_fps_input.value()),
            render_size=(int(self.camera_width_input.value()), int(self.camera_height_input.value())),
        )

    def _select_camera_from_monitor_combo(self) -> None:
        camera_id = self.monitor_camera_combo.currentData()
        if camera_id:
            self._select_camera(camera_id)

    def _select_camera(self, camera_id: str) -> None:
        self._selected_camera_id = camera_id
        for tile_id, tile in self._tiles.items():
            tile.set_selected(tile_id == camera_id)
        self._refresh_camera_selectors()
        self._refresh_ptz_state()
        if self._layout_slots == 1:
            self._rebuild_video_grid()
            if self._streaming_requested:
                self._restart_capture_workers()

    def _change_layout_slots(self) -> None:
        slots = self.layout_combo.currentData()
        self._layout_slots = int(slots or 4)
        self._render_timer.setInterval(self.performance_policy.render_interval_ms(self._layout_slots))
        self._rebuild_video_grid()
        if self._streaming_requested:
            self._restart_capture_workers()

    def _selected_camera(self) -> CameraConfig | None:
        return self._camera_by_id(self._selected_camera_id)

    def _ptz_camera(self) -> CameraConfig | None:
        camera_id = self.ptz_camera_combo.currentData()
        return self._camera_by_id(camera_id) if camera_id else None

    def _camera_by_id(self, camera_id: str) -> CameraConfig | None:
        return self._camera_map().get(camera_id)

    def _camera_map(self) -> dict[str, CameraConfig]:
        return {camera.id: camera for camera in self.app_config.cameras}

    def _active_cameras(self) -> list[CameraConfig]:
        return [camera for camera in self.app_config.cameras if camera.active]

    def _visible_cameras(self) -> list[CameraConfig]:
        cameras = self._active_cameras()
        if self._layout_slots == 1 and self._selected_camera_id:
            selected = self._selected_camera()
            return [selected] if selected and selected.active else cameras[:1]
        return cameras[: self._layout_slots]

    def _capture_cameras(self) -> list[CameraConfig]:
        by_id = {camera.id: camera for camera in self._visible_cameras()}
        for state in self.recording_manager.active_states():
            camera = self._camera_by_id(state.camera_id)
            if camera and camera.active:
                by_id[camera.id] = camera
        return list(by_id.values())

    def _add_event(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        logger.info(message)
        if hasattr(self, "events_list"):
            self.events_list.insertItem(0, line)
        self.status_label.setText(message[:120])

    def _load_logo_pixmap(self) -> QPixmap | None:
        project_root = Path(__file__).resolve().parent.parent
        for path in (project_root / "assets/logo.png", project_root / "logo_t (2).avif"):
            if not path.exists():
                continue
            pixmap = QPixmap(str(path.resolve()))
            if not pixmap.isNull():
                return pixmap
        return None

    def _clear_layout(self, layout: Any) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)


def _slugify(value: str) -> str:
    """Create a filesystem-friendly identifier from a string."""
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _directory_size(path: Path) -> int:
    """Return the total size of regular files inside a directory tree."""
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())

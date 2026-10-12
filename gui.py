import os
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from camera import (
    DEFAULT_FILTER_MS, DEFAULT_VIS_WINDOW_MS, HW_BA_MAX_250US, OFF_FINE_MAX, ON_FINE_MIN, CameraWorker,
)
from session import RecordingSession

PREVIEW_SIZE = (420, 320)  # QLabel 표시 크기 (원본 비율 유지해서 리사이즈)


def np_to_pixmap(img):
    if img is None:
        return None
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.tobytes(), w, h, w, QImage.Format_Grayscale8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        qimg = QImage(img.tobytes(), w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAVIS Event Camera Recorder")

        self.session = None
        self.width = 0
        self.height = 0
        self.camera_name = ""
        self._last_frame_img = None
        self._last_frame_ts = None
        self._record_start_time = None

        self._build_ui()

        self.worker = CameraWorker()
        self.worker.connected.connect(self.on_connected)
        self.worker.frameReady.connect(self.on_frame_ready)
        self.worker.eventVisReady.connect(self.on_event_vis_ready)
        self.worker.livePreview.connect(self.on_live_preview)
        self.worker.statusMessage.connect(self.log)
        self.worker.rateUpdated.connect(self.on_rate_updated)
        self.worker.errorOccurred.connect(self.on_error)
        self.worker.start()

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._elapsed_timer.start(500)

    # ---------------------------------------------------------------- UI

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 미리보기 2개: APS 프레임 / 이벤트 시각화(흰 배경 + 빨강/파랑)
        preview_box = QGridLayout()
        self.frame_label = QLabel("카메라 연결 대기 중...")
        self.frame_label.setFixedSize(*PREVIEW_SIZE)
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("background:#222;color:#aaa;border:1px solid #444;")

        self.event_label = QLabel("이벤트 없음")
        self.event_label.setFixedSize(*PREVIEW_SIZE)
        self.event_label.setAlignment(Qt.AlignCenter)
        self.event_label.setStyleSheet("background:#fff;color:#888;border:1px solid #444;")

        preview_box.addWidget(QLabel("APS Frame"), 0, 0)
        preview_box.addWidget(QLabel("Event Visualization (white bg / red=ON / blue=OFF)"), 0, 1)
        preview_box.addWidget(self.frame_label, 1, 0)
        preview_box.addWidget(self.event_label, 1, 1)
        root.addLayout(preview_box)

        # 녹화 설정
        settings_box = QGroupBox("녹화 설정")
        settings_layout = QGridLayout(settings_box)

        settings_layout.addWidget(QLabel("세션 이름"), 0, 0)
        self.session_name_edit = QLineEdit("session")
        settings_layout.addWidget(self.session_name_edit, 0, 1)

        settings_layout.addWidget(QLabel("저장 폴더"), 1, 0)
        self.output_dir_edit = QLineEdit(os.path.expanduser("~/davis_recordings"))
        settings_layout.addWidget(self.output_dir_edit, 1, 1)
        browse_btn = QPushButton("찾아보기...")
        browse_btn.clicked.connect(self.choose_output_dir)
        settings_layout.addWidget(browse_btn, 1, 2)

        root.addWidget(settings_box)

        # 노이즈 필터 (시각화 전용 - events.bin 원본은 그대로 저장됨)
        filter_box = QGroupBox("이벤트 이미지 설정 (시각화 전용, events.bin 은 원본 전체 저장)")
        filter_layout = QHBoxLayout(filter_box)
        self.filter_check = QCheckBox("배경 노이즈 제거")
        self.filter_check.setChecked(True)
        filter_layout.addWidget(self.filter_check)
        filter_layout.addWidget(QLabel("시간창"))
        self.filter_slider = QSlider(Qt.Horizontal)
        self.filter_slider.setRange(1, 50)
        self.filter_slider.setValue(int(DEFAULT_FILTER_MS))
        filter_layout.addWidget(self.filter_slider, 1)
        self.filter_value_label = QLabel(f"{int(DEFAULT_FILTER_MS)} ms")
        self.filter_value_label.setMinimumWidth(48)
        filter_layout.addWidget(self.filter_value_label)
        filter_layout.addWidget(QLabel("(작을수록 강하게 제거)"))
        filter_layout.addSpacing(16)
        filter_layout.addWidget(QLabel("이미지당 이벤트 시간"))
        self.vis_window_spin = QSpinBox()
        self.vis_window_spin.setRange(1, 300)
        self.vis_window_spin.setSuffix(" ms")
        self.vis_window_spin.setValue(DEFAULT_VIS_WINDOW_MS)
        self.vis_window_spin.valueChanged.connect(lambda v: self.worker.set_vis_window(v))
        filter_layout.addWidget(self.vis_window_spin)
        self.filter_check.toggled.connect(self._on_filter_changed)
        self.filter_slider.valueChanged.connect(self._on_filter_changed)
        root.addWidget(filter_box)

        # 카메라 하드웨어 설정 (카메라가 내보내는 이벤트 자체가 바뀜 -> events.bin 원본에도 영향)
        hw_box = QGroupBox("카메라 하드웨어 설정 (주의: events.bin 원본에도 영향)")
        hw_layout = QGridLayout(hw_box)

        hw_layout.addWidget(QLabel("HW 노이즈 필터"), 0, 0)
        self.hw_ba_slider = QSlider(Qt.Horizontal)
        self.hw_ba_slider.setRange(0, HW_BA_MAX_250US)
        hw_layout.addWidget(self.hw_ba_slider, 0, 1)
        self.hw_ba_label = QLabel("끔")
        self.hw_ba_label.setMinimumWidth(64)
        hw_layout.addWidget(self.hw_ba_label, 0, 2)

        hw_layout.addWidget(QLabel("OFF 민감도"), 1, 0)
        self.hw_off_slider = QSlider(Qt.Horizontal)
        self.hw_off_slider.setRange(1, OFF_FINE_MAX)
        self.hw_off_slider.setValue(1)
        hw_layout.addWidget(self.hw_off_slider, 1, 1)
        self.hw_off_label = QLabel("기본")
        hw_layout.addWidget(self.hw_off_label, 1, 2)

        hw_layout.addWidget(QLabel("ON 민감도"), 2, 0)
        self.hw_on_slider = QSlider(Qt.Horizontal)
        self.hw_on_slider.setRange(0, 255 - ON_FINE_MIN)   # 0 = 기본(fine 255), 클수록 민감
        self.hw_on_slider.setValue(0)
        hw_layout.addWidget(self.hw_on_slider, 2, 1)
        self.hw_on_label = QLabel("기본")
        hw_layout.addWidget(self.hw_on_label, 2, 2)

        self.hw_reset_btn = QPushButton("카메라 설정 기본값으로")
        hw_layout.addWidget(self.hw_reset_btn, 3, 0, 1, 2)
        self.rate_label = QLabel("이벤트율: - Mev/s")
        hw_layout.addWidget(self.rate_label, 3, 2)
        hw_layout.addWidget(QLabel("민감도를 올리면 어두운 물체는 잘 잡히지만 노이즈도 늘어남 -> HW 노이즈 필터와 함께 조절. "
                                   "이벤트가 폭주하면 자동으로 기본값 복귀."), 4, 0, 1, 3)

        self.hw_ba_slider.valueChanged.connect(self._on_hw_ba)
        self.hw_off_slider.valueChanged.connect(self._on_hw_off)
        self.hw_on_slider.valueChanged.connect(self._on_hw_on)
        self.hw_reset_btn.clicked.connect(self._on_hw_reset)
        root.addWidget(hw_box)

        # 녹화 컨트롤 / 상태
        control_layout = QHBoxLayout()
        self.record_btn = QPushButton("● 녹화 시작")
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self.toggle_record)
        control_layout.addWidget(self.record_btn)

        self.status_label = QLabel("프레임: 0   이벤트: 0   경과: 0.0s")
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        root.addLayout(control_layout)

        # 로그
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(120)
        root.addWidget(self.log_view)

        self.resize(920, 940)

    def log(self, msg):
        self.log_view.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _on_filter_changed(self, *_):
        ms = self.filter_slider.value()
        self.filter_value_label.setText(f"{ms} ms")
        self.filter_slider.setEnabled(self.filter_check.isChecked())
        self.worker.set_noise_filter(self.filter_check.isChecked(), ms)

    def _on_hw_ba(self, v):
        self.hw_ba_label.setText("끔" if v == 0 else f"{v * 0.25:g} ms")
        self.worker.request_hw("ba", v)

    def _on_hw_off(self, v):
        self.hw_off_label.setText("기본" if v == 1 else f"+{v - 1}")
        self.worker.request_hw("off_fine", v)

    def _on_hw_on(self, v):
        self.hw_on_label.setText("기본" if v == 0 else f"+{v}")
        self.worker.request_hw("on_fine", 255 - v)

    def _on_hw_reset(self):
        for sl, val in ((self.hw_ba_slider, 0), (self.hw_off_slider, 1), (self.hw_on_slider, 0)):
            sl.blockSignals(True)
            sl.setValue(val)
            sl.blockSignals(False)
        self.hw_ba_label.setText("끔")
        self.hw_off_label.setText("기본")
        self.hw_on_label.setText("기본")
        self.worker.request_hw("reset", True)

    def _on_hw_reset_sliders_only(self):
        # 워커가 이미 bias 를 되돌렸으므로 슬라이더 표시만 맞춤 (BA 필터는 그대로)
        for sl, val in ((self.hw_off_slider, 1), (self.hw_on_slider, 0)):
            sl.blockSignals(True)
            sl.setValue(val)
            sl.blockSignals(False)
        self.hw_off_label.setText("기본")
        self.hw_on_label.setText("기본")

    def on_rate_updated(self, mev):
        px = self.width * self.height
        per_px = mev * 1e6 / px if px else 0
        self.rate_label.setText(f"이벤트율: {mev:.2f} Mev/s ({per_px:.1f}/픽셀/초)")

    def choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", self.output_dir_edit.text())
        if d:
            self.output_dir_edit.setText(d)

    # ----------------------------------------------------------- signals

    def on_connected(self, name, w, h):
        self.camera_name = name
        self.width, self.height = w, h
        self.record_btn.setEnabled(True)
        self.frame_label.setText("카메라 연결됨 - APS 프레임 대기 중...")
        self.log(f"연결됨: {name} ({w}x{h})")

    def on_error(self, msg):
        self.log(f"[오류] {msg}")
        if "자동 원복" in msg:
            self._on_hw_reset_sliders_only()
        if msg.startswith("카메라 연결 실패"):
            hint = ""
            if "LIBUSB_ERROR_ACCESS" in msg or "already in use" in msg:
                hint = "\n\n다른 프로그램(DV.app 등)이 카메라를 쓰는 중.\n그 프로그램을 종료하고 이 앱을 다시 실행하세요."
            self.frame_label.setWordWrap(True)
            self.frame_label.setText("카메라 연결 실패" + hint)

    def on_frame_ready(self, img, ts):
        self._last_frame_img = img
        self._last_frame_ts = ts
        pix = np_to_pixmap(img)
        if pix is not None:
            self.frame_label.setPixmap(pix.scaled(*PREVIEW_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_live_preview(self, vis_img):
        # 아직 프레임이 도착 안 한 사이의 누적 이벤트도 실시간으로 보여줌
        pix = np_to_pixmap(vis_img)
        if pix is not None:
            self.event_label.setPixmap(pix.scaled(*PREVIEW_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_event_vis_ready(self, vis_img, raw_events, ts_start, ts_end, meta=None):
        # 라벨은 livePreview(슬라이딩 윈도우)만 갱신 — 여기서 덮어쓰면 깜빡임이 생김.
        # 이 시그널은 녹화 저장 전용.
        if self.session is not None and self._last_frame_img is not None:
            self.session.write(
                self._last_frame_img, self._last_frame_ts,
                vis_img, raw_events, ts_start, ts_end, meta,
            )
            self._update_status()

    # ------------------------------------------------------------ record

    def toggle_record(self):
        if self.session is None:
            out_dir = self.output_dir_edit.text().strip() or os.path.expanduser("~/davis_recordings")
            os.makedirs(out_dir, exist_ok=True)
            self.session = RecordingSession(
                out_dir, self.session_name_edit.text(),
                self.width, self.height, self.camera_name,
            )
            self._record_start_time = time.time()
            self.record_btn.setText("■ 녹화 중지")
            self.session_name_edit.setEnabled(False)
            self.output_dir_edit.setEnabled(False)
            self.log(f"녹화 시작: {self.session.dir}")
        else:
            path = self.session.dir
            n_frames = self.session.frame_idx
            n_events = self.session.total_events
            self.session.close()
            self.session = None
            self._record_start_time = None
            self.record_btn.setText("● 녹화 시작")
            self.session_name_edit.setEnabled(True)
            self.output_dir_edit.setEnabled(True)
            self.log(f"녹화 종료: {path} (프레임 {n_frames}개, 이벤트 {n_events}개)")

    def _update_status(self):
        if self.session is None:
            return
        elapsed = time.time() - self._record_start_time if self._record_start_time else 0.0
        self.status_label.setText(
            f"프레임: {self.session.frame_idx}   이벤트: {self.session.total_events}   경과: {elapsed:.1f}s"
        )

    def _update_elapsed(self):
        if self.session is not None:
            self._update_status()

    def closeEvent(self, event):
        if self.session is not None:
            self.session.close()
        self.worker.stop()
        event.accept()

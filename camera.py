"""
DAVIS(iniVation) 이벤트 카메라 캡처 스레드.

dv-processing 파이썬 바인딩(https://dv-processing.inivation.com)을 사용해서
카메라에서 APS 프레임과 이벤트 배치를 계속 읽어온다.

주의:
  dv-processing 은 버전에 따라 CameraCapture / Frame / EventStore 의
  메서드 이름이 조금씩 바뀌어 왔다 (예: frame.timestamp 가 속성인지 함수인지).
  이 파일의 _frame_image / _frame_timestamp / _events_to_numpy 세 함수에
  실제 설치된 버전에 맞는 접근 방식을 몰아넣었으니, 실행해보고 에러가 나면
  이 세 함수만 고치면 된다. 설치 후 아래 명령으로 실제 API를 먼저 확인해보는
  것도 좋다:

      python3 -c "import dv_processing as dv; help(dv.io.CameraCapture)"
"""

import threading
import time
from datetime import timedelta

import numpy as np
from PySide6.QtCore import QThread, Signal

# 이벤트 1개를 저장하는 고정 포맷 (raw event 바이너리 파일에 그대로 씀)
#   timestamp: 카메라 기준 마이크로초(us) 타임스탬프
#   x, y      : 픽셀 좌표
#   polarity  : 1 = ON(밝아짐), 0 = OFF(어두워짐)
EVENT_DTYPE = np.dtype([
    ("timestamp", "<i8"),
    ("x", "<u2"),
    ("y", "<u2"),
    ("polarity", "u1"),
])

DEFAULT_RESOLUTION = (346, 260)  # (width, height) - DAVIS346 기본값, 연결되면 실제 값으로 갱신됨

# 미리보기 전용 슬라이딩 윈도우(녹화/매핑과는 무관). 프레임 도착과 상관없이
# 항상 "최근 PREVIEW_WINDOW_US 동안의 이벤트"만 그려서 깜빡임을 없앤다.
PREVIEW_WINDOW_US = 33_000
PREVIEW_MAX_FPS = 60

# 배경 노이즈 필터: 같은 위치 근처(이웃 픽셀)에서 이 시간 안에 다른 이벤트가 없었던
# "고립된" 이벤트를 노이즈로 보고 제거한다. 값이 작을수록 강하게 걸러냄.
DEFAULT_FILTER_MS = 2.0

# 이벤트 시각화 이미지 1장에 그릴 시간 길이(ms). 노출 중앙 기준. events.bin 은 이것과 무관하게 전부 저장.
DEFAULT_VIS_WINDOW_MS = 33

# 카메라 하드웨어 설정 안전 범위 (스윕 테스트로 확인한 값: 이 범위 밖은 이벤트 폭주 위험)
#   Off fine  : 1(기본, 가장 조용) ~ OFF_FINE_MAX  (커질수록 민감 = 이벤트 증가, 20 이면 10Mev/s 폭주)
#   On  fine  : 255(기본, 가장 조용) ~ ON_FINE_MIN (작아질수록 민감 = 이벤트 증가)
OFF_FINE_MAX = 6
ON_FINE_MIN = 150
HW_BA_MAX_250US = 40          # 하드웨어 BA 필터 최대 10ms (1 단위 = 250us, 0 = 끔)
CONNECT_RETRIES = 8            # 카메라 열기 재시도 횟수 / 간격(초)
CONNECT_RETRY_SEC = 1.5
FLOOD_MEV_PER_S = 6.0         # 이 이상이 2초 연속이면 bias 를 자동 원복


def _frame_image(frame):
    """dv.Frame -> numpy 이미지(BGR or GRAY)."""
    img = getattr(frame, "image", None)
    if img is None:
        raise AttributeError("Frame 객체에서 .image 를 찾을 수 없음 (dv-processing API 확인 필요)")
    return np.asarray(img)


def _frame_timestamp(frame):
    """dv.Frame -> 마이크로초 타임스탬프(int)."""
    ts = getattr(frame, "timestamp", None)
    if callable(ts):
        ts = ts()
    if ts is None:
        raise AttributeError("Frame 객체에서 timestamp 를 찾을 수 없음 (dv-processing API 확인 필요)")
    return int(ts)


def _events_to_numpy(events):
    """dv.EventStore -> EVENT_DTYPE 구조화 배열."""
    # 최신 dv-processing 은 .numpy() 를 지원함
    numpy_fn = getattr(events, "numpy", None)
    if callable(numpy_fn):
        arr = numpy_fn()
        out = np.empty(len(arr), dtype=EVENT_DTYPE)
        out["timestamp"] = arr["timestamp"]
        out["x"] = arr["x"]
        out["y"] = arr["y"]
        out["polarity"] = arr["polarity"]
        return out

    # 폴백: 이벤트 하나씩 순회 (구버전 바인딩 대응)
    n = len(events)
    out = np.empty(n, dtype=EVENT_DTYPE)
    for i, e in enumerate(events):
        out[i] = (e.timestamp(), e.x(), e.y(), 1 if e.polarity() else 0)
    return out


def render_event_visualization(events_arr, width, height, on_color=(0, 0, 255), off_color=(255, 0, 0)):
    """
    이벤트 배열을 흰 배경 위에 그린다.
    OpenCV(BGR) 기준: on_color 기본값 = 빨강(ON/positive), off_color 기본값 = 파랑(OFF/negative).
    같은 픽셀에 여러 이벤트가 겹치면 가장 마지막(시간상 최신) 이벤트 색이 남는다.
    """
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if events_arr is None or len(events_arr) == 0:
        return canvas

    xs = events_arr["x"].astype(np.int32)
    ys = events_arr["y"].astype(np.int32)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys = xs[valid], ys[valid]
    pol = events_arr["polarity"][valid]

    canvas[ys[pol == 0], xs[pol == 0]] = off_color
    canvas[ys[pol == 1], xs[pol == 1]] = on_color
    return canvas


class CameraWorker(QThread):
    """카메라를 열고 백그라운드에서 계속 프레임/이벤트를 읽는 스레드."""

    connected = Signal(str, int, int)          # device_name, width, height
    frameReady = Signal(object, object)            # frame_image(np.ndarray), timestamp_us
    eventVisReady = Signal(object, object, object, object, object)  # vis_image, raw_events(np.ndarray), ts_start, ts_end, meta(dict)
    livePreview = Signal(object)                 # 프레임 사이 실시간 누적 이벤트 미리보기 이미지
    statusMessage = Signal(str)
    rateUpdated = Signal(float)                  # 초당 이벤트 수 (Mev/s, 필터 전)
    errorOccurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._stop_requested = False
        self._capture = None
        self._width, self._height = DEFAULT_RESOLUTION
        # 노이즈 필터 설정 (GUI 스레드에서 바꾸고 워커 루프가 폴링해서 반영)
        self.filter_enabled = True
        self.filter_ms = DEFAULT_FILTER_MS
        self.vis_window_ms = DEFAULT_VIS_WINDOW_MS
        # GUI 스레드 -> 워커 스레드 카메라 설정 요청 (카메라 객체는 워커 스레드에서만 만짐)
        self._pending = {}
        self._pending_lock = threading.Lock()

    def request_hw(self, key, value):
        """key: 'ba'(HW BA 필터, 250us 단위) | 'off_fine' | 'on_fine' | 'reset'"""
        with self._pending_lock:
            self._pending[key] = value

    def set_noise_filter(self, enabled, duration_ms):
        self.filter_enabled = bool(enabled)
        self.filter_ms = float(duration_ms)

    def set_vis_window(self, window_ms):
        self.vis_window_ms = float(window_ms)

    def stop(self):
        self._stop_requested = True
        self._running = False
        self.wait(2000)

    def run(self):
        try:
            import dv_processing as dv
        except ImportError:
            self.errorOccurred.emit(
                "dv-processing 모듈이 없음. `pip install dv-processing` 로 설치 필요."
            )
            return

        # dv.io.camera.DAVIS() 는 연결된 첫 번째 DAVIS 카메라를 연다.
        # 직전에 종료한 프로그램이 카메라를 아직 놓지 않았을 수 있어서 몇 번 재시도한다.
        last_exc = None
        for attempt in range(1, CONNECT_RETRIES + 1):
            if self._stop_requested:
                return
            try:
                self._capture = dv.io.camera.DAVIS()
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - 카메라 연결 문제를 그대로 사용자에게 보여줌
                last_exc = exc
                self.statusMessage.emit(f"카메라 연결 재시도 {attempt}/{CONNECT_RETRIES} ...")
                time.sleep(CONNECT_RETRY_SEC)
        if last_exc is not None:
            self.errorOccurred.emit(f"카메라 연결 실패: {last_exc}")
            return

        try:
            self._width, self._height = self._capture.getEventResolution()
        except Exception:
            try:
                self._width, self._height = self._capture.getFrameResolution()
            except Exception:
                self._width, self._height = DEFAULT_RESOLUTION

        name = ""
        try:
            name = self._capture.getCameraName()
        except Exception:
            pass

        extra = []
        try:
            extra.append(f"serial={self._capture.getSerialNumber()}")
        except Exception:
            pass
        try:
            extra.append(f"colorMode={self._capture.getColorMode()}")
        except Exception:
            pass

        self.connected.emit(name or "DAVIS", self._width, self._height)
        self.statusMessage.emit(
            f"카메라 연결됨: {name or 'DAVIS'} ({self._width}x{self._height}) " + " ".join(extra)
        )

        event_buffer = []         # 원본 이벤트 (events.bin 저장용, 필터 안 거침)
        vis_buffer = []           # 필터 통과 이벤트 (events_vis 이미지용)
        buffer_ts_start = None
        preview_chunks = []       # 미리보기용 슬라이딩 윈도우 (녹화 버퍼와 별개)
        last_preview_emit = 0.0

        noise_filter = None
        applied_ms = None
        try:
            noise_filter = dv.noise.BackgroundActivityNoiseFilter(
                (self._width, self._height),
                backgroundActivityDuration=timedelta(milliseconds=self.filter_ms),
            )
            applied_ms = self.filter_ms
        except Exception as exc:  # noqa: BLE001
            self.errorOccurred.emit(f"노이즈 필터 생성 실패(필터 없이 진행): {exc}")
        last_filter_log = time.monotonic()

        # --- 하드웨어 설정 (bias / HW 필터): 시작 시 값을 기억해뒀다가 종료/리셋 때 원복 ---
        cap = self._capture
        BiasCF = dv.io.camera.DAVIS.Davis346BiasCF
        orig_on = orig_off = None
        try:
            orig_on = cap.getDavis346BiasCoarseFine(BiasCF.On)
            orig_off = cap.getDavis346BiasCoarseFine(BiasCF.Off)
        except Exception as exc:  # noqa: BLE001
            self.errorOccurred.emit(f"bias 읽기 실패(bias 조절 비활성): {exc}")
        hw = {"bias_tuned": False, "flood_secs": 0}
        rate_count = 0
        rate_t0 = time.monotonic()

        def reset_bias():
            if orig_on is not None:
                cap.setDavis346BiasCoarseFine(BiasCF.On, *orig_on)
                cap.setDavis346BiasCoarseFine(BiasCF.Off, *orig_off)
            hw["bias_tuned"] = False

        self._running = True
        while self._running:
            got_something = False

            # --- GUI 에서 온 카메라 설정 요청 반영 (안전 범위로 clamp) ---
            if self._pending:
                with self._pending_lock:
                    cmds, self._pending = self._pending, {}
                for key, val in cmds.items():
                    try:
                        if key == "reset":
                            reset_bias()
                            cap.setBackgroundActivityFilter(0)
                        elif key == "ba":
                            cap.setBackgroundActivityFilter(int(min(max(val, 0), HW_BA_MAX_250US)))
                        elif key == "off_fine" and orig_off is not None:
                            cap.setDavis346BiasCoarseFine(
                                BiasCF.Off, orig_off[0], int(min(max(val, orig_off[1]), OFF_FINE_MAX)))
                            hw["bias_tuned"] = True
                        elif key == "on_fine" and orig_on is not None:
                            cap.setDavis346BiasCoarseFine(
                                BiasCF.On, orig_on[0], int(min(max(val, ON_FINE_MIN), orig_on[1])))
                            hw["bias_tuned"] = True
                    except Exception as exc:  # noqa: BLE001
                        self.errorOccurred.emit(f"카메라 설정 적용 실패({key}={val}): {exc}")

            # --- 1초마다 이벤트율 보고 + 폭주 시 bias 자동 원복 ---
            now_m = time.monotonic()
            if now_m - rate_t0 >= 1.0:
                rate = rate_count / (now_m - rate_t0) / 1e6
                self.rateUpdated.emit(rate)
                rate_count, rate_t0 = 0, now_m
                if hw["bias_tuned"] and rate > FLOOD_MEV_PER_S:
                    hw["flood_secs"] += 1
                    if hw["flood_secs"] >= 2:
                        try:
                            reset_bias()
                            self.errorOccurred.emit(
                                f"이벤트 폭주({rate:.1f} Mev/s) 감지 - bias 를 기본값으로 자동 원복함")
                        except Exception as exc:  # noqa: BLE001
                            self.errorOccurred.emit(f"bias 원복 실패: {exc}")
                        hw["flood_secs"] = 0
                else:
                    hw["flood_secs"] = 0

            # --- 이벤트 배치 ---
            try:
                is_evt_avail = getattr(self._capture, "isEventStreamAvailable", lambda: True)()
            except Exception:
                is_evt_avail = True

            if is_evt_avail:
                try:
                    events = self._capture.getNextEventBatch()
                except Exception as exc:
                    self.errorOccurred.emit(f"이벤트 읽기 오류: {exc}")
                    events = None

                if events is not None and len(events) > 0:
                    got_something = True
                    arr = _events_to_numpy(events)
                    rate_count += len(arr)
                    if buffer_ts_start is None and len(arr) > 0:
                        buffer_ts_start = int(arr["timestamp"][0])
                    event_buffer.append(arr)

                    # 시각화용: 노이즈 필터 통과분만 (필터는 상태를 가지므로 항상 모든 배치를 먹여야 함)
                    vis_arr = arr
                    if noise_filter is not None:
                        if applied_ms != self.filter_ms:
                            noise_filter.setBackgroundActivityDuration(
                                timedelta(milliseconds=self.filter_ms))
                            applied_ms = self.filter_ms
                        noise_filter.accept(events)
                        filtered = _events_to_numpy(noise_filter.generateEvents())
                        if self.filter_enabled:
                            vis_arr = filtered
                    vis_buffer.append(vis_arr)

                    # 미리보기: 최근 PREVIEW_WINDOW_US 만 유지, 최대 PREVIEW_MAX_FPS 로 갱신
                    t_latest = int(arr["timestamp"][-1])
                    if len(vis_arr) > 0:
                        preview_chunks.append(vis_arr)
                    cutoff = t_latest - PREVIEW_WINDOW_US
                    preview_chunks = [c for c in preview_chunks if int(c["timestamp"][-1]) >= cutoff]
                    now = time.monotonic()
                    if now - last_preview_emit >= 1.0 / PREVIEW_MAX_FPS:
                        last_preview_emit = now
                        if preview_chunks:
                            win = np.concatenate(preview_chunks)
                            win = win[win["timestamp"] >= cutoff]
                        else:
                            win = np.empty(0, dtype=EVENT_DTYPE)
                        self.livePreview.emit(render_event_visualization(win, self._width, self._height))

            # --- APS 프레임 ---
            try:
                is_frame_avail = getattr(self._capture, "isFrameStreamAvailable", lambda: True)()
            except Exception:
                is_frame_avail = True

            if is_frame_avail:
                try:
                    frame = self._capture.getNextFrame()
                except Exception as exc:
                    self.errorOccurred.emit(f"프레임 읽기 오류: {exc}")
                    frame = None

                if frame is not None:
                    got_something = True
                    try:
                        img = _frame_image(frame)
                        ts = _frame_timestamp(frame)
                    except Exception as exc:
                        self.errorOccurred.emit(f"프레임 파싱 오류: {exc}")
                        img = ts = None

                    if img is not None:
                        self.frameReady.emit(img, ts)

                        if event_buffer:
                            merged = np.concatenate(event_buffer)
                        else:
                            merged = np.empty(0, dtype=EVENT_DTYPE)
                        vis_merged = np.concatenate(vis_buffer) if vis_buffer else merged

                        # 이미지 1장에는 "노출 중앙" 기준 vis_window_ms 만큼의 이벤트만 그린다.
                        # (프레임 timestamp = 노출 시작. 노출/프레임 간격이 길어도 이미지가 포화되지 않게)
                        try:
                            exposure_us = int(frame.exposure.total_seconds() * 1e6)
                        except Exception:
                            exposure_us = 0
                        half = int(self.vis_window_ms * 1000 / 2)
                        center = ts + exposure_us // 2
                        lo, hi = center - half, center + half

                        vis_win = vis_merged[(vis_merged["timestamp"] >= lo) & (vis_merged["timestamp"] <= hi)]
                        vis = render_event_visualization(vis_win, self._width, self._height)

                        # 같은 구간의 원본 이벤트가 merged(=events.bin 조각) 안에서 어디인지
                        if len(merged) > 0:
                            ts_arr = merged["timestamp"]
                            i0 = int(np.searchsorted(ts_arr, lo, side="left"))
                            i1 = int(np.searchsorted(ts_arr, hi, side="right"))
                            ts_start, ts_end = int(ts_arr[0]), int(ts_arr[-1])
                        else:
                            i0 = i1 = 0
                            ts_start = ts_end = ts
                        meta = {
                            "exposure_us": exposure_us,
                            "vis_ts_start_us": lo, "vis_ts_end_us": hi,
                            "vis_raw_start_idx": i0, "vis_raw_end_idx": i1,
                            "vis_shown_count": len(vis_win),
                        }
                        self.eventVisReady.emit(vis, merged, ts_start, ts_end, meta)

                        event_buffer = []
                        vis_buffer = []
                        buffer_ts_start = None

            # 5초마다 필터가 얼마나 걸러내는지 로그
            if noise_filter is not None and time.monotonic() - last_filter_log >= 5.0:
                last_filter_log = time.monotonic()
                try:
                    self.statusMessage.emit(
                        f"노이즈 필터 {'ON' if self.filter_enabled else 'OFF(표시만 원본)'} "
                        f"{applied_ms:g}ms: 이벤트 {noise_filter.getReductionFactor() * 100:.1f}% 제거"
                    )
                except Exception:
                    pass

            if not got_something:
                # 바쁜 루프 방지 (카메라 SDK가 blocking 콜을 지원하면 그쪽으로 바꿔도 됨)
                time.sleep(0.001)

        # 종료 시 카메라를 기본 상태로 되돌림 (다음 프로그램이 이상한 bias 를 물려받지 않게)
        try:
            reset_bias()
            cap.setBackgroundActivityFilter(0)
        except Exception:
            pass
        self.statusMessage.emit("카메라 스레드 종료")

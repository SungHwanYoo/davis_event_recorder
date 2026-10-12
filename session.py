"""
녹화 세션: 프레임 PNG, 이벤트 시각화 PNG, raw 이벤트 바이너리, 매핑 CSV/메타를
디스크에 순서대로 기록한다.

세션 폴더 구조
--------------
<output_dir>/<session_name>_<YYYYmmdd_HHMMSS>/
    meta.json               # 카메라 정보, 시작 시각, 컬럼 설명
    frames.csv              # 프레임<->이벤트 매핑 테이블 (아래 컬럼 참고)
    frames/
        frame_000000.png     # APS 원본 프레임
        frame_000001.png
        ...
    events_vis/
        frame_000000.png     # 같은 인덱스 프레임 직전~해당 프레임 사이 이벤트를
        frame_000001.png     # 흰 배경 + 빨강(ON)/파랑(OFF) 점으로 그린 시각화 이미지
        ...
    events.bin               # 전체 raw 이벤트 (timestamp us, x, y, polarity) 연속 기록

frames.csv 컬럼:
    frame_idx, frame_timestamp_us, frame_file, event_vis_file,
    event_ts_start_us, event_ts_end_us, event_count,
    events_byte_offset, events_byte_length

즉 frames.csv 한 행만 보면 해당 프레임에 대응하는 이벤트 구간을
events.bin 에서 byte offset/length 로 바로 slice 해서 읽을 수 있고,
동시에 이미 렌더링된 events_vis/frame_XXXXXX.png 로 바로 눈으로도 확인 가능하다.
"""

import csv
import json
import os
import time

import cv2
import numpy as np

from camera import EVENT_DTYPE


class RecordingSession:
    def __init__(self, output_dir, session_name, width, height, camera_name=""):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = session_name.strip() or "session"
        self.dir = os.path.join(output_dir, f"{safe_name}_{stamp}")
        self.frames_dir = os.path.join(self.dir, "frames")
        self.vis_dir = os.path.join(self.dir, "events_vis")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.width = width
        self.height = height
        self.frame_idx = 0
        self.total_events = 0
        self._events_byte_offset = 0

        self._events_file = open(os.path.join(self.dir, "events.bin"), "ab")
        self._csv_file = open(os.path.join(self.dir, "frames.csv"), "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "frame_idx", "frame_timestamp_us", "frame_file", "event_vis_file",
            "event_ts_start_us", "event_ts_end_us", "event_count",
            "events_byte_offset", "events_byte_length",
            "exposure_us", "vis_ts_start_us", "vis_ts_end_us",
            "vis_raw_byte_offset", "vis_raw_event_count", "vis_shown_count",
        ])

        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump({
                "camera_name": camera_name,
                "width": width,
                "height": height,
                "started_at": stamp,
                "event_dtype": "int64 timestamp_us, uint16 x, uint16 y, uint8 polarity (little-endian, packed)",
                "polarity_color": {"on(1)": "red(BGR 0,0,255)", "off(0)": "blue(BGR 255,0,0)", "background": "white"},
                "note": "events_vis/frame_XXXXXX.png 는 노출 중앙 기준 vis_ts_start~vis_ts_end 구간의 이벤트를 흰 배경에 그린 것(노이즈 필터 적용 가능). events.bin 은 필터 없는 원본 전체.",
            }, f, ensure_ascii=False, indent=2)

    def write(self, frame_img, frame_ts, vis_img, raw_events, ts_start, ts_end, meta=None):
        frame_name = f"frame_{self.frame_idx:06d}.png"
        vis_name = f"frame_{self.frame_idx:06d}.png"

        cv2.imwrite(os.path.join(self.frames_dir, frame_name), frame_img)
        cv2.imwrite(os.path.join(self.vis_dir, vis_name), vis_img)

        n_events = 0 if raw_events is None else len(raw_events)
        byte_len = 0
        if n_events > 0:
            packed = np.asarray(raw_events, dtype=EVENT_DTYPE)
            self._events_file.write(packed.tobytes())
            byte_len = packed.nbytes

        self._csv_writer.writerow([
            self.frame_idx, frame_ts, f"frames/{frame_name}", f"events_vis/{vis_name}",
            ts_start, ts_end, n_events,
            self._events_byte_offset, byte_len,
            *self._vis_columns(meta),
        ])
        self._csv_file.flush()

        self._events_byte_offset += byte_len
        self.total_events += n_events
        self.frame_idx += 1

    def _vis_columns(self, meta):
        """이미지에 그려진 구간이 events.bin 의 어느 byte 범위인지 계산해서 CSV 컬럼으로."""
        if not meta:
            return ["", "", "", "", "", ""]
        i0, i1 = meta["vis_raw_start_idx"], meta["vis_raw_end_idx"]
        rec = EVENT_DTYPE.itemsize
        return [
            meta["exposure_us"], meta["vis_ts_start_us"], meta["vis_ts_end_us"],
            self._events_byte_offset + i0 * rec, i1 - i0, meta["vis_shown_count"],
        ]

    def close(self):
        self._events_file.close()
        self._csv_file.close()

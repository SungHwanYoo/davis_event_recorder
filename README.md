# DAVIS Event Camera Recorder

DAVIS(iniVation) 컬러 이벤트 카메라로 APS 프레임 + 이벤트 데이터를 동시에 녹화하고,
프레임마다 자동으로 매핑된 파일로 저장해주는 간단한 PySide6 GUI.

## 실행

```bash
cd ~/davis_event_recorder
source .venv/bin/activate   # 이미 dv-processing / PySide6 / opencv-python / numpy 설치돼 있음
python3 main.py
```

카메라를 USB로 연결한 뒤 실행하면 자동으로 첫 번째 DAVIS 장치를 연다.
연결에 실패하면 창 하단 로그에 에러 메시지가 뜬다(예: 권한 문제, 다른 프로그램이 이미 점유 중 등).

`.venv`가 없거나 새로 세팅해야 하면:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
(이 Mac은 Apple Silicon + Python 3.13이라 `dv-processing`이 pip에서 바로 wheel로 설치됨. 확인 완료.)

## 화면 구성

- 좌측 미리보기: APS 프레임 (흑백/컬러 그대로)
- 우측 미리보기: **흰 배경 위에 이벤트를 빨강(ON)/파랑(OFF)으로 찍은 시각화** — 프레임 도착 전에도
  누적 중인 이벤트가 실시간으로 계속 갱신되고, 프레임이 도착하는 순간 "그 프레임 구간"으로 확정되어
  저장됨(직전 프레임 ~ 이번 프레임 사이 이벤트만 누적, 겹치지 않게 초기화됨)
- 세션 이름 / 저장 폴더 지정 후 "● 녹화 시작"

## 저장되는 파일 구조

```
<저장폴더>/<세션이름>_<YYYYmmdd_HHMMSS>/
    meta.json              # 카메라 정보, 컬러 컨벤션 등
    frames.csv             # 프레임<->이벤트 매핑 테이블 (아래 참고)
    frames/
        frame_000000.png   # APS 원본 프레임
        frame_000001.png
        ...
    events_vis/
        frame_000000.png   # 같은 인덱스: 흰 배경 + 빨강(ON)/파랑(OFF) 이벤트 시각화 이미지
        frame_000001.png   # (frames/ 와 파일명이 1:1로 매칭됨)
        ...
    events.bin              # 전체 raw 이벤트 연속 기록 (timestamp_us, x, y, polarity)
```

`frames.csv` 컬럼:

| 컬럼 | 설명 |
|---|---|
| frame_idx | 프레임 순번 (frames/, events_vis/ 파일명과 동일) |
| frame_timestamp_us | 해당 APS 프레임의 카메라 타임스탬프(us) |
| frame_file / event_vis_file | 상대 경로 |
| event_ts_start_us / event_ts_end_us | 이 구간에 누적된 이벤트의 시간 범위 |
| event_count | 이 구간 이벤트 개수 |
| events_byte_offset / events_byte_length | `events.bin`에서 이 구간 raw 이벤트를 바로 slice할 수 있는 byte 위치 |

즉 `frames.csv` 한 줄만 봐도:
- 어느 이미지(`frames/frame_XXXXXX.png`)인지
- 같은 구간 이벤트를 빨강/파랑으로 그린 시각화 이미지(`events_vis/frame_XXXXXX.png`)
- 그 구간의 raw 이벤트를 `events.bin`에서 정확히 어디부터 몇 바이트 읽으면 되는지

가 전부 매핑되어 있다. raw 이벤트를 파이썬에서 통째로 읽고 싶으면:

```python
import numpy as np
EVENT_DTYPE = np.dtype([("timestamp","<i8"),("x","<u2"),("y","<u2"),("polarity","u1")])
events = np.fromfile("events.bin", dtype=EVENT_DTYPE)
```

## 색 컨벤션

- 배경: 흰색
- ON(밝아짐, polarity=1): 빨강
- OFF(어두워짐, polarity=0): 파랑

`camera.py`의 `render_event_visualization()` 상단 인자로 색을 바꿀 수 있음.

## 참고/주의

- `dv-processing` 파이썬 API는 실제 이 Mac에 설치해서 (`dv.io.camera.DAVIS()`, `EventStore.numpy()`,
  `Frame.image/.timestamp` 등) 직접 확인하고 맞춘 것이라 이름이 틀릴 걱정은 없음. 다만 카메라가 실제로
  붙어있지 않은 상태에서 테스트해서, 실제 프레임 이미지 채널 순서(BGR 가정) 등은 카메라 연결 후 첫 실행에서
  눈으로 한번 확인 권장.
- 실행 시 터미널에 `objc[...]: Class ... is implemented in both ... libavdevice ...` 경고가 뜰 수 있는데,
  opencv-python과 dv-processing이 각자 내장 ffmpeg/opencv 라이브러리를 갖고 있어서 나는 경고이고 동작에는
  영향 없음(무시해도 됨).
- 컬러 모드(GRAYSCALE/ORIGINAL 등)는 카메라에 저장된 현재 설정을 그대로 씀. 흑백으로 나오면
  `camera.py`의 `CameraWorker.run()`에서 카메라 연결 직후에
  `self._capture.setColorMode(dv.io.camera.DAVIS.ColorMode.ORIGINAL)` 한 줄 추가하면 됨.

# MediaPipe 실습

Google **MediaPipe**와 **OpenCV**를 이용해 손 감지, 얼굴 감지, 배경 분리(Selfie Segmentation)를 해 보는 인공지능 기초 실습 프로젝트입니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `hand_detector.py` | 손 감지: 손마다 21개 랜드마크와 연결선을 표시 (`hand.mp4` 또는 웹캠) |
| `face_detector.py` | 얼굴 감지: 얼굴 박스와 6개 키포인트(눈·코·입·귀), 신뢰도, 얼굴 수를 표시 (`face.mp4` 또는 웹캠) |
| `hand_detector_youtube.py` | YouTube 영상 주소를 입력으로 받아 손 감지 (`yt-dlp` 사용) |
| `selfie_segmentation.py` | 사람과 배경을 분리해 배경만 블러 처리 (블러를 기본보다 강하게 수정) |
| `requirements.txt` | 필요한 패키지 목록 |

## 실행 환경

- Python 3.12 (MediaPipe는 Python 3.9~3.12에서 동작)
- `mediapipe==0.10.21`: 0.10.30 이후 버전부터는 이 코드가 쓰는 `mp.solutions` API가 빠져서 버전을 고정함
- `opencv-python==4.11.0.86`, `numpy<2`: 위 mediapipe 버전과 호환되는 버전

## 설치

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

## 실행

```bash
python hand_detector.py
python face_detector.py
python selfie_segmentation.py
python hand_detector_youtube.py                 # 기본 YouTube 영상
python hand_detector_youtube.py "<YouTube URL>" # 다른 영상
```

- 실행 창은 `ESC` 키를 누르면 종료됩니다.
- 웹캠을 쓰려면 코드에서 `cv2.VideoCapture(0)`의 주석을 풀고 mp4 줄을 주석 처리하세요. 카메라가 안 잡히면 번호를 1, 2, 3으로 바꿔 보세요.
- `*.mp4` 파일은 `.gitignore`에 들어 있어 저장소에 올라가지 않습니다. 실행하려면 제공된 `hand.mp4`, `face.mp4`를 프로젝트 폴더에 넣으세요.

## 코드 설명

### hand_detector.py와 face_detector.py의 차이
- 사용하는 솔루션: `mp.solutions.hands` / `mp.solutions.face_detection`
- 결과: 손은 `multi_hand_landmarks`(손마다 21개 3D 랜드마크), 얼굴은 `detections`(박스, 6개 키포인트, 신뢰도 점수)
- 그리기: 손은 `draw_landmarks()`와 `HAND_CONNECTIONS`, 얼굴은 `draw_detection()`
- 옵션: Hands는 `max_num_hands`, `min_tracking_confidence`(추적) 사용, FaceDetection은 `model_selection`(0=근거리, 1=원거리) 사용

### hand_detector_youtube.py
`yt-dlp`로 YouTube 주소에서 실제 영상 스트림 주소(mp4/H.264, 720p 이하)를 꺼낸 뒤 `cv2.VideoCapture()`에 넘깁니다. 셀카 영상이 아니므로 좌우 반전은 하지 않습니다.

### selfie_segmentation.py
1. `segmentation_mask`: 픽셀마다 사람일 확률(0~1)을 담은 마스크를 0.5 기준으로 이진화
2. `GaussianBlur`로 흐린 배경 이미지 생성
3. `frame * mask + blur * (1 - mask)`: 사람 부분은 원본, 배경은 블러 이미지로 합성

배경을 더 흐리게 하기 위해 블러 커널을 `(31, 31)`에서 `(99, 99)`로 키우고 두 번 적용했습니다. 커널 크기는 홀수여야 하고, 클수록 더 흐려집니다.

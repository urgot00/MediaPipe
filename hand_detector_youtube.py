#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 사용법: python hand_detector_youtube.py [YouTube URL]
# 사전 설치: pip install yt-dlp

import sys
import cv2
import mediapipe as mp
import yt_dlp

# =========================================
# 🎬 YouTube 주소 → 실제 영상 스트림 주소 변환
# =========================================
YOUTUBE_URL = "https://youtu.be/udeQhZHx-00"
if len(sys.argv) > 1:
    YOUTUBE_URL = sys.argv[1]


def get_stream_url(youtube_url):
    ydl_opts = {
        # OpenCV가 읽을 수 있는 mp4(H.264) 영상, 720p 이하
        "format": "bestvideo[ext=mp4][vcodec^=avc1][height<=720]/best[ext=mp4][height<=720]/best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        print(f"🎬 영상 제목: {info.get('title')}")
        return info["url"]


# =========================================
# 🧩 Mediapipe 초기화
# =========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,       # 동영상 입력
    max_num_hands=2,              # 최대 손 개수
    min_detection_confidence=0.5, # 탐지 신뢰도
    min_tracking_confidence=0.5   # 추적 신뢰도
)

# =========================================
# 📸 YouTube 스트림 연결
# =========================================
print(f"🔗 YouTube 영상 불러오는 중: {YOUTUBE_URL}")
cap = cv2.VideoCapture(get_stream_url(YOUTUBE_URL))

print("📷 스트림 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 프레임을 읽지 못했습니다. (영상 끝 또는 스트림 오류)")
        break

    # BGR → RGB 변환 (YouTube 영상은 셀카가 아니므로 좌우 반전 안 함)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 손 검출 수행
    result = hands.process(image_rgb)

    # 🖐️ 손 랜드마크 표시
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    # 화면 표시
    cv2.imshow('🖐️ MediaPipe Hand Detector (YouTube)', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()

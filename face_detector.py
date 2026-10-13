#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

# =========================================
# 🧩 Mediapipe 초기화
# =========================================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,            # 0=근거리(2m 이내), 1=원거리(5m 이내)
    min_detection_confidence=0.5  # 탐지 신뢰도
)

# =========================================
# 📸 카메라 연결
# =========================================
# cap = cv2.VideoCapture(0)            # 기본 카메라 사용시
cap = cv2.VideoCapture("face.mp4")   # 동영상 파일 사용 시

print("📷 카메라 스트림 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 프레임을 읽지 못했습니다. 카메라 연결을 확인하세요.")
        break

    # 좌우 반전 (셀카 뷰)
    image = cv2.flip(image, 1)

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    result = face_detection.process(image_rgb)

    # 🙂 얼굴 박스 + 6개 키포인트(눈, 코, 입, 귀) 표시
    if result.detections:
        h, w, _ = image.shape
        for detection in result.detections:
            mp_drawing.draw_detection(image, detection)

            # 신뢰도 점수 표시
            box = detection.location_data.relative_bounding_box
            x, y = int(box.xmin * w), int(box.ymin * h)
            score = detection.score[0]
            cv2.putText(image, f"{score * 100:.0f}%", (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(image, f"Faces: {len(result.detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # 화면 표시
    cv2.imshow('🙂 MediaPipe Face Detector', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()

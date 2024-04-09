# 라이브러리
import cv2, dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model
import winsound

# 추가한 라이브러리
# import tkinter as tk
# from PIL import Image, ImageTk

# 이미지 크기
IMG_SIZE = (64,56)
# 눈 크기
B_SIZE = (34, 26)
# 여백
margin = 95
class_labels = ['center','left', 'right'] 
# 감지기 초기화
detector = dlib.get_frontal_face_detector()
# 예측기 초기화
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# font 설정
font_letter = cv2.FONT_HERSHEY_PLAIN
# 눈동자 위치 예측 모델 불러오기
model = load_model('models/gazev3.1.h5')
# 눈 깜빡임 감지 모델 불러오기
model_b = load_model('models/blinkdetection.h5')

# 눈동자의 위치를 예측하는 함수 정의
def detect_gaze(eye_img):
    pred_l = model.predict(eye_img)
    # 모델이 예측한 결과를 numpy 배열로 변환하여
    # 배열 내의 가장 큰 값(가장 높은 확률) 선택
    # 백분율로 변환(100을 곱한 이유), 계산된 확률을 정수형으로 변환
    accuracy = int(np.array(pred_l).max() * 100)
    # 예측된 결과 중에서 가장 높은 확률을 가진 클래스의 인덱스 반환
    # class_labels 리스트에서 해당 인덱스에 해당하는 클래스 선택
    gaze = class_labels[np.argmax(pred_l)]
    # 예측된 눈동자의 위치에 대한 클래스 저장('center','left', 'right' 중 하나)
    return gaze

# 눈 깜빡임을 감지하는 함수 정의
def detect_blink(eye_img):
    # 눈이 감겼는지 여부에 대한 확률 반환
    pred_B = model_b.predict(eye_img)
    # 모델의 예측 결과 중 첫번째 값을 변수 status에 저장
    status = pred_B[0][0]
    # 확률 값을 백분율로 변환
    status = status*100
    # 소수점 아래 세자리까지 반올림
    status = round(status,3)
    # 눈을 감은 정도를 백분율로 반환
    return  status

# 눈 영역을 잘라내는 함수 정의
def crop_eye(img, eye_points):
    # 눈 영역의 좌상단 모서리
    # 눈의 윤곽 점(eye_points) x, y 좌표의 최소값을 찾는다
    x1, y1 = np.amin(eye_points, axis=0)
    # 눈 영역의 우하단 모서리
    # 눈의 윤곽 점 x, y 좌표의 최대값 찾는다
    x2, y2 = np.amax(eye_points, axis=0)
    # 최소 좌표와 최대 좌표 사이의 중심점
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    # 눈의 너비 계산(눈의 영역 확장->더 많은 정보 포착위해 1.2 곱함)
    w = (x2 - x1) * 1.2
    # 눈의 높이 계산(IMG_SIZE 비율 적용 for 이미지 비율 유지)
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    # 눈 영역 주변 여백 계산
    margin_x, margin_y = w / 2, h / 2

    # 눈 영역의 최소 x,y 좌표
    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    # 눈 영역의 최대 x,y 좌표
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    # 눈 영역의 경계 상자 좌표 반올림하여 정수형으로 변환
    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    # 계산된 경계 상자 좌표를 이용하여 주어진 이미지에서 눈 영역 잘라냄
    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    # 잘려진 눈 이미지, 눈 영역 경계 상자 좌표 반환
    return eye_img, eye_rect

# main
# -------- 웹캠에서 프레임을 읽고 화면에 표시하기 위한 초기 설정 수행 ---------------
# 시스템에 연결된 첫번째(인덱스 0) 비디오 장치 사용해서 캡처하는 객체 생성
# ↪ 웹캠
# 카메라 열기
cap = cv2.VideoCapture(0)
# pattern = []
# frames = 10 # 눈깜빡임 인식하기 위해 필요한 프레임 수
# pattern_length = 0
# 눈 깜빡임 인지하기 위해 필요한 프레임수
# 눈 깜빡임을 감지하는 데 사용되는 임계값
frames_to_blink = 6
# 현재까지의 눈 깜빡임 프레임 수
blinking_frames = 0
blinking_frames_total = 0  # 수정된 부분: 눈 깜빡임 횟수를 누적할 변수
blinking_detected = False  # 수정된 부분:눈을 감았다가 떴을 때 한 번의 깜빡임으로 인식하기 위한 플래그

# 영상 처리 while 루프 시작
while cap.isOpened(): # 웹캠이 열려있는 동안 무한 루프 실행(isOpened함수: 웹캠 올바르게 열려있는지 여부 확인)
    # 빈화면 생성(후속 프레임에서 실제 영상 프레임 위에 덮어씌워 결과 표시)
    output = np.zeros((900,820,3), dtype="uint8") # 크기 (900, 820), 채널 3(RGB)
    # cap 객체 사용하여 웹캠에서 다음 프레임을 읽어옴
    # ret:프레임 제대로 읽어왔는지 나타내는 부울 값
    # img:실제 프레임 이미지
    ret, img = cap.read()
    # 필요 시 이미지를 수평으로 뒤집음(원래의 웹캠 이미지와 동일한 방향으로 출력)
    # flipCode=1: 수평으로 뒤집는 옵션
    img = cv2.flip(img,flipCode = 1)
    # 눈 영역을 자르고 크기를 조정할 때 사용(높이 112, 너비 128)
    h,w = (112,128)	
    # ------------------------------------------
    # 이미지 가져오지 못한 경우 루프 종료
    if not ret: # True:성공, False:실패
        break

    # 이미지를 흑백으로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    # 얼굴 검출
    faces = detector(gray)

    # 얼굴에서 눈의 위치 검출
    # -------얼굴에서 눈, 코, 입과 같은 주요 특징점 감지, 해당 특징점을 연결하는 선을 그림---------
    # 감지된 얼굴들에 대한 루프
    # faces: dlib 얼굴 감지기로 감지된 얼굴들의 목록
    for face in faces:
        # 현재 얼굴에 대해 얼굴 특징점 예측
        # predictor:얼굴 영역과 함께 입력 이미지를 받아 얼굴 특징점을 예측하는 데 사용되는 dlib 모델
        shapes = predictor(gray, face)
        
        # 눈 주위의 특징점에 대한 선을 그리는 부분
        for n in range(36,42): #왼쪽 눈에 해당하는 특징점 범위, 36~41 인덱스 사용
            x= shapes.part(n).x # 특정 인덱스의 x 좌표 가져옴
            y = shapes.part(n).y # 특정 인덱스의 y 좌표 가져옴
            next_point = n+1 # 다음 점의 인덱스 설정
            # 마지막 점인 경우 다음 점을 첫 번째 점으로 설정
            if n==41:
                next_point = 36 
            
            # 다음 점의 x좌표 가져옴
            x2 = shapes.part(next_point).x
            # 다음 점의 y좌표 가져옴
            y2 = shapes.part(next_point).y
            # 눈 주위 특징점 연결선 그림, 빨간색(0, 69, 255)으로 표시, 두께 2
            cv2.line(img,(x,y),(x2,y2),(0,69,255),2)

        for n in range(42,48): # 오른쪽 눈 특징점 선 그리는 부분, 42~47 인덱스 사용
            x= shapes.part(n).x
            y = shapes.part(n).y
            next_point = n+1
            if n==47:
                next_point = 42 
            
            x2 = shapes.part(next_point).x
            y2 = shapes.part(next_point).y
            cv2.line(img,(x,y),(x2,y2),(153,0,153),2) # 파란색선
        # dlib의 얼굴 특징점 객체를 NumPy 배열로 변환(특징점 좌표 접근 쉬워짐)
        shapes = face_utils.shape_to_np(shapes)
    # -------------------------------------
        #~~~~~~~~~~~~~~~~~56,64 EYE IMAGE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 왼쪽 눈, 오른쪽 눈 이미지 crop
        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])
        #~~~~~~~~~~~~~~~~FOR THE EYE FINAL_WINDOW~~~~~~~~~~~~~~~~~~~~~~#
        eye_img_l_view = cv2.resize(eye_img_l, dsize=(128,112))
        eye_img_l_view = cv2.cvtColor(eye_img_l_view,cv2.COLOR_BGR2RGB)
        eye_img_r_view = cv2.resize(eye_img_r, dsize=(128,112))
        eye_img_r_view = cv2.cvtColor(eye_img_r_view, cv2.COLOR_BGR2RGB)
        #~~~~~~~~~~~~~~~~~FOR THE BLINK DETECTION~~~~~~~~~~~~~~~~~~~~~~~
        # 눈 깜빡임 감지를 위해 눈 이미지 크기 조정, 정규화
        # copy()해서 원본 이미지 보존
        # B_SIZE 현재(34, 26)으로 설정 (width, height) -> 이미지 크기 조정할 때 사용
        eye_blink_left = cv2.resize(eye_img_l.copy(), B_SIZE)
        eye_blink_right = cv2.resize(eye_img_r.copy(), B_SIZE)

        # 1차원 배열로 평탄화된 이미지를 다시 (1, height, width, 1) 형태로 변형 -> 각 눈 이미지에 대해 배치 차원, 채널 차원 추가(for 딥러닝 모델 사용)
        # 0에서 1 사이의 값으로 정규화하기 위해 255로 나눔
        eye_blink_left_i = eye_blink_left.reshape((1, B_SIZE[1], B_SIZE[0], 1)).astype(np.float32) / 255.
        eye_blink_right_i = eye_blink_right.reshape((1, B_SIZE[1], B_SIZE[0], 1)).astype(np.float32) / 255.
        #~~~~~~~~~~~~~~~~FOR THE GAZE DETECTIOM~~~~~~~~~~~~~~~~~~~~~~~~#
        # 눈의 시선 추적을 위한 데이터를 준비하는 단계
        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_input_g = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

        #----추가한 부분
        # 오른쪽 눈 이미지 크기 조정
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        # 오른쪽 눈 이미지 형태 변형 및 정규화
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
        #----

        #~~~~~~~~~~~~~~~~~~PREDICTION PROCESS~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        # 왼쪽 눈 깜빡임 상태 감지
        status_l = detect_blink(eye_blink_left_i)
        #----추가한 부분
        # 오른쪽 눈 깜빡임 상태 감지
        status_r = detect_blink(eye_blink_right_i)
        #----
        # -> 0에 가까울수록 눈이 감겨 있는 정도가 크다

        # 눈동자 위치 감지
        gaze =  detect_gaze(eye_input_g)
        #---추가
        # 오른쪽 눈동자 위치 감지
        gaze_r =  detect_gaze(eye_input_r)
        #---

        # 왼쪽 눈에 대해
        if gaze == class_labels[1]:
            blinking_frames += 1
            #blinking_frames_total += 1  # 수정된 부분: 눈 깜빡임 횟수 누적
            # #---추가
            # # blinking 인식
            # if not blinking_detected:
            #     blinking_detected = True
            #     blinking_frames_total += 1

            # elif status_l < 1 and status_r < 1: # 양쪽 눈이 감긴 경우
            #     blinking_detected = False  # 수정된 부분:눈을 감았다가 떴을 때까지 플래그 유지
            #     if blinking_frames == frames_to_blink:
            #         # pattern_length +=1
            #         winsound.Beep(1000,250)
            #         # pattern.append(1)
            # #---

            #추가한 부분 변경1
            if status_l < 1 and status_r < 1: 
                blinking_detected = False
                if not blinking_detected:
                    blinking_detected = True
                    blinking_frames_total += 1 
                if blinking_frames == frames_to_blink:
                    winsound.Beep(1000,250)

            # # 추가한 부분 변경2
            # blinking_detected = False  # 눈 깜빡임 감지 여부
            # prev_status_l = 0  # 이전 프레임의 왼쪽 눈 상태
            # prev_status_r = 0  # 이전 프레임의 오른쪽 눈 상태

            # if status_l >= 50 and status_r >= 50:  # 두 눈이 모두 열렸다면
            #     if prev_status_l < 1 and prev_status_r < 1:  # 이전에 두 눈이 모두 닫혀 있었다면
            #         if not blinking_detected:  # 눈 깜빡임이 감지되지 않았다면
            #             blinking_detected = True  # 눈 깜빡임 감지
            #             blinking_frames_total += 1  # 눈 깜빡임 횟수 증가
            #             winsound.Beep(1000, 250)  # 비프음 재생

            # prev_status_l = status_l  # 현재 프레임의 왼쪽 눈 상태 저장
            # prev_status_r = status_r  # 현재 프레임의 오른쪽 눈 상태 저장

                
            # --- 변경:주석처리함    
            # if blinking_frames == frames_to_blink:
            #     # pattern_length +=1
            #     winsound.Beep(1000,250)
            #     # pattern.append(1)
            # ---

        elif gaze == class_labels[2]:
            blinking_frames += 1
            #blinking_frames_total += 1  # 수정된 부분: 눈 깜빡임 횟수 누적
            #---추가
            # blinking 인식
            if status_l < 1 and status_r < 1: 
                blinking_detected = False
                if not blinking_detected:
                    blinking_detected = True
                    blinking_frames_total += 1 
                if blinking_frames == frames_to_blink:
                    winsound.Beep(1000,250)
            #---
                    
            #---변경:주석처리함
            # if blinking_frames == frames_to_blink:
            #     # pattern_length +=1
            #     winsound.Beep(1000,250)
            #     # pattern.append(2)
            #---

        # 오른쪽 눈동자        
        # if gaze_r == class_labels[1]:
        #     blinking_frames += 1
        #     #blinking_frames_total += 1  # 수정된 부분: 눈 깜빡임 횟수 누적
        #     if blinking_frames == frames_to_blink:
        #         # pattern_length +=1
        #         winsound.Beep(1000,250)
        #         # pattern.append(1)
        # elif gaze_r == class_labels[2]:
        #     blinking_frames += 1
        #     #blinking_frames_total += 1  # 수정된 부분: 눈 깜빡임 횟수 누적
        #     if blinking_frames == frames_to_blink:
        #         # pattern_length +=1
        #         winsound.Beep(1000,250)
        #         # pattern.append(2)


        # --- 변경: 주석처리함
        # elif status_l < 0.1:
        #     blinking_frames += 1
        #     if blinking_frames == frames_to_blink:
        #         # pattern_length +=1
        #         # pattern.append(3)
        #         winsound.Beep(1000,250)
        # ---

        # 변경3
        # else:
        #         blinking_detected = False  # 눈이 감겨 있지 않다면 눈 깜빡임 감지 해제

        else:
            #---추가
            # blinking 인식
            if status_l < 1 and status_r < 1: 
                blinking_detected = False
                if not blinking_detected:
                    blinking_detected = True
                    blinking_frames_total += 1 
                blinking_frames = 0
                
            #---

            
            # blinking_frames = 0
        
        # --- 추가
        # 오른쪽 눈에 대해
        if gaze_r == class_labels[1]:
            blinking_frames += 1

            if status_l < 1 and status_r < 1: 
                blinking_detected = False
                if not blinking_detected:
                    blinking_detected = True
                    #blinking_frames_total += 1 
                if blinking_frames == frames_to_blink:
                    winsound.Beep(1000,250)

        elif gaze_r == class_labels[2]:
            blinking_frames += 1
            if status_l < 1 and status_r < 1: 
                blinking_detected = False
                if not blinking_detected:
                    blinking_detected = True
                    #blinking_frames_total += 1 
                if blinking_frames == frames_to_blink:
                    winsound.Beep(1000,250)
                
        else:
            if status_l < 1 and status_r < 1: 
                blinking_detected = False
                if not blinking_detected:
                    blinking_detected = True
                    #blinking_frames_total += 1 
                blinking_frames = 0
        #---
        
        # 출력 결과
        #~~~~~~~~~~~~~~~~~~~~~~~FINAL_WINDOWS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        output = cv2.line(output,(400,200), (400,0),(0,255,0),thickness=2)
        cv2.putText(output,"LEFT EYE GAZE",(10,180), font_letter,1, (255,255,51),1)
        cv2.putText(output,"LEFT EYE OPENING %",(200,180), font_letter,1, (255,255,51),1)
        cv2.putText(output,"RIGHT EYE GAZE",(440,180), font_letter,1, (255,255,51),1)
        cv2.putText(output,"RIGHT EYE OPENING %",(621,180), font_letter,1, (255,255,51),1)
        # 양쪽 눈에 대해서 변경
        # if status_l < 10:		
        if status_l < 10 and status_r < 10 : # 눈을 감은 정도가 매우 높을 때 
            cv2.putText(output,"---BLINKING----",(250,300), font_letter,2, (153,153,255),2)

        # 왼쪽 눈 결과
        output[0:112, 0:128] = eye_img_l_view
        cv2.putText(output, gaze,(30,150), font_letter,2, (0,255,0),2)
        output[0:112, margin+w:(margin+w)+w] = eye_img_l_view
        cv2.putText(output,(str(status_l)+"%"),((margin+w),150), font_letter,2, (0,0,255),2)

        # 오른쪽 눈 결과
        output[0:112, 2*margin+2*w:(2*margin+2*w)+w] = eye_img_r_view
        # gaze_r로 변경
        # cv2.putText(output, gaze,((2*margin+2*w)+30,150), font_letter,2, (0,0,255),2)
        cv2.putText(output, gaze_r,((2*margin+2*w)+30,150), font_letter,2, (0,0,255),2)
        output[0:112, 3*margin+3*w:(3*margin+3*w)+w] = eye_img_r_view
        # status_r로 변경
        # cv2.putText(output, (str(status_l)+"%"),((3*margin+3*w),150), font_letter,2, (0,0,255),2)
        cv2.putText(output, (str(status_r)+"%"),((3*margin+3*w),150), font_letter,2, (0,0,255),2)
        output[235+100:715+100, 80:720] = img

        # 변경한 부분
        # 화면에 눈 깜빡임 횟수 표시
        cv2.putText(output, "Blink Count: " + str(blinking_frames_total), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #
        # 오류
        # 정면인 경우에만 눈 깜빡임 카운트됨
        # 눈 감고 있을 때 카운트 올라감
        # 

        #cv2.imshow('result',output)
        
        # 변경한 부분
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)  # 창의 이름과 창의 속성을 설정합니다.
        cv2.resizeWindow('result', 1280, 720)  # 창의 크기를 원하는 크기로 조정합니다.

        cv2.imshow('result', output)  # 이미지를 디스플레이합니다.
        #

    # q 로 종료
    if cv2.waitKey(1) == ord('q') : 
        break
cap.release()
cv2.destroyAllWindows()    
# print(pattern)

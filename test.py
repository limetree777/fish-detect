import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

app = Flask(__name__)

# 객체 감지(Detection)용 YOLO 모델 로드
model = YOLO('best.pt') 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # 1. 이미지 받기
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "이미지가 없습니다."}), 400

    in_memory_file = file.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. YOLO 객체 감지 수행
    results = model(img)
    result = results[0]
    
    # 박스와 라벨이 그려진 이미지 생성
    annotated_img = result.plot()

    detected_objects = []

    # 3. 감지된 객체 정보 추출
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0]) * 100
        species = result.names[cls_id]
        
        # 한국어 이름으로 변환
        species = change_fish_name_to_korean(species)

        detected_objects.append({
            "species": species,
            "confidence": round(conf, 1)
        })

    # 4. 분석이 끝난 이미지를 Base64로 인코딩
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # 결과 반환 (길이 계산은 프론트엔드로 위임)
    return jsonify({
        "status": "success",
        "detected_objects": detected_objects,
        "result_image": f"data:image/jpeg;base64,{img_base64}"
    })


def change_fish_name_to_korean(species):
    fish_name = {
        "Olive flounder": "넙치(광어)",
        "Korea rockfish": "조피볼락(우럭)",
        "Red seabream": "참돔",
        "Black porgy": "감성돔",
        "Rock bream": "돌돔",
        "Scomber japonicus	": "고등어",
        "Snakehead": "가물치",
        "Pseudocaranx dentex": "흑점줄전갱이",
        "Mugil cephalus": "숭어",
        "Freshwater Eel": "민물장어, 뱀장어",
        "Belone belone": "학꽁치",
    }
    return fish_name.get(species, species)  # 매핑이 없으면 원래 이름 반환

if __name__ == '__main__':
    app.run(port=11111)
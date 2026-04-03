from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# 1. 모델 로드
model = YOLO('best.pt') 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image'].read()
        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "이미지 디코딩 실패"}), 400

        results = model.predict(img, conf=0.25, verbose=False)
        
        annotated_img = img.copy()
        predictions = []

        for r in results:
            for box in r.boxes:
                # 상자 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = r.names[cls]
                
                # 상자 및 텍스트 그리기
                # 파란색 상자 (255, 0, 0), 두께 2
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # 텍스트 내용: [어종 확신도%] (예: fish 85%)
                label_str = f"{label} {round(conf * 100)}%"
                
                # 텍스트 그리기 (상자 좌상단 위)
                cv2.putText(annotated_img, label_str, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                predictions.append({
                    "species": label,
                    "confidence": round(conf, 2)
                })

        # 4. 결과 이미지를 base64로 변환
        _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        img_as_text = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "image_data": f"data:image/jpeg;base64,{img_as_text}",
            "predictions": predictions
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 외부 접속 할때는 0.0.0.0으로 실행
    app.run(port=11111)
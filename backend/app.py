"""
AI 검사 시스템 - Flask 백엔드 서버
작성자: AI 시스템 개발팀
버전: 1.0
기능: 이미지 업로드, YOLO 불량 검증, 결과 반환
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import logging
import base64
from pathlib import Path
import json
import sqlite3

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flask_app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 설정으로 frontend와 연결

# 설정 변수
UPLOAD_FOLDER = './inspectionmodule/appset'
MODEL_PATH = './inspectionmodule/yolo_dataset'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 데이터베이스 초기화
def init_database():
    """SQLite 데이터베이스 초기화"""
    conn = sqlite3.connect('inspection_logs.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            processed_filename TEXT NOT NULL,
            file_size INTEGER,
            is_defective BOOLEAN NOT NULL,
            confidence REAL NOT NULL,
            defect_count INTEGER DEFAULT 0,
            details TEXT,
            result_image TEXT,
            processing_time REAL,
            model_used TEXT DEFAULT 'YOLOv8'
        )
    ''')
    
    conn.commit()
    conn.close()

def save_inspection_log(log_data):
    """검사 로그를 데이터베이스에 저장"""
    try:
        conn = sqlite3.connect('inspection_logs.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO inspection_logs 
            (timestamp, original_filename, processed_filename, file_size, 
             is_defective, confidence, defect_count, details, result_image, 
             processing_time, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            log_data['timestamp'],
            log_data['original_filename'],
            log_data['processed_filename'],
            log_data['file_size'],
            log_data['is_defective'],
            log_data['confidence'],
            log_data['defect_count'],
            log_data['details'],
            log_data['result_image'],
            log_data['processing_time'],
            log_data['model_used']
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"로그 저장 실패: {str(e)}")
        return False

def get_inspection_logs(limit=100):
    """검사 로그 조회"""
    try:
        conn = sqlite3.connect('inspection_logs.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM inspection_logs 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        # 딕셔너리 형태로 변환
        logs = []
        for row in rows:
            logs.append({
                'id': row[0],
                'timestamp': row[1],
                'original_filename': row[2],
                'processed_filename': row[3],
                'file_size': row[4],
                'is_defective': bool(row[5]),
                'confidence': row[6],
                'defect_count': row[7],
                'details': row[8],
                'result_image': row[9],
                'processing_time': row[10],
                'model_used': row[11]
            })
        
        return logs
    except Exception as e:
        logger.error(f"로그 조회 실패: {str(e)}")
        return []

# YOLO 모델 전역 변수
yolo_model = None

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_best_model():
    """최적의 훈련된 모델 찾기 (test.py와 동일한 로직)"""
    import glob
    
    # 훈련된 모델들 찾기 (우선순위 순서) - test.py와 동일
    model_candidates = [
        # 1. 실제 훈련된 최종 모델
        "./inspectionmodule/defect_detection_model_20250719_115333.pt",
        # 2. 다른 타임스탬프 모델들 (패턴 매칭)
    ]
    
    # defect_detection_model_*.pt 패턴으로 모든 모델 찾기
    pattern_models = glob.glob("./inspectionmodule/defect_detection_model_*.pt")
    pattern_models.sort(reverse=True)  # 최신순 정렬
    model_candidates.extend(pattern_models)
    
    # yolo_dataset 폴더 내 모델들도 확인
    yolo_dataset_models = glob.glob("./inspectionmodule/yolo_dataset/defect_detection_model_*.pt")
    yolo_dataset_models.sort(reverse=True)
    model_candidates.extend(yolo_dataset_models)
    
    # runs/detect 폴더의 best 모델들 (최신순)
    runs_candidates = [
        "./inspectionmodule/runs/detect/train2222/weights/best.pt",
        "./inspectionmodule/runs/detect/train222/weights/best.pt", 
        "./inspectionmodule/runs/detect/train22/weights/best.pt",
        "./inspectionmodule/runs/detect/train2/weights/best.pt",
        "./inspectionmodule/runs/detect/train/weights/best.pt",
    ]
    model_candidates.extend(runs_candidates)
    
    # 기본 모델 (마지막 옵션)
    model_candidates.append("./inspectionmodule/yolov8n.pt")
    
    # 중복 제거하면서 순서 유지
    seen = set()
    unique_candidates = []
    for candidate in model_candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    
    logger.info("모델 검색 순서:")
    for i, candidate in enumerate(unique_candidates[:10]):  # 상위 10개만 표시
        logger.info(f"  {i+1}. {candidate}")
    
    for candidate in unique_candidates:
        if os.path.exists(candidate):
            logger.info(f"✅ 모델 발견: {candidate}")
            return candidate
    
    raise FileNotFoundError("사용 가능한 YOLO 모델을 찾을 수 없습니다.")

def load_yolo_model():
    """YOLO 모델 로드"""
    global yolo_model
    try:
        # test.py와 동일한 모델 찾기 로직 사용
        model_path = find_best_model()
        yolo_model = YOLO(model_path)
        logger.info(f"YOLO 모델 로드 성공: {model_path}")
        
        # 모델 정보 출력
        logger.info(f"모델 클래스: {yolo_model.names}")
        logger.info(f"모델 클래스 개수: {len(yolo_model.names) if yolo_model.names else 'Unknown'}")
        
        return True
            
    except Exception as e:
        logger.error(f"YOLO 모델 로드 실패: {str(e)}")
        return False

def preprocess_image(image_path):
    """이미지 전처리"""
    try:
        # OpenCV로 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("이미지를 읽을 수 없습니다.")
        
        # RGB로 변환 (YOLO는 RGB 입력을 기대)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 이미지 크기 확인
        height, width = image_rgb.shape[:2]
        logger.info(f"이미지 크기: {width}x{height}")
        
        return image_rgb
        
    except Exception as e:
        logger.error(f"이미지 전처리 실패: {str(e)}")
        return None

def analyze_detection_results(results):
    """YOLO 검출 결과 분석 (test.py와 동일한 로직)"""
    try:
        if not results or len(results) == 0:
            return {
                'is_defective': False,
                'confidence': 0.0,
                'defect_count': 0,
                'details': "검출된 객체가 없습니다."
            }
        
        result = results[0]  # 첫 번째 결과 사용
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return {
                'is_defective': False,
                'confidence': 0.0,
                'defect_count': 0,
                'details': "불량이 검출되지 않았습니다."
            }
        
        # 모든 검출 정보 수집
        all_detections = []
        defect_detections = []
        max_confidence = 0.0
        
        for box in boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id] if yolo_model.names else f"class_{class_id}"
            
            # 바운딩 박스 크기 계산 (너무 작은 박스 필터링)
            if hasattr(box, 'xyxy'):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                # 너무 작은 박스는 노이즈로 간주 (이미지 크기의 0.1% 미만)
                if box_area < 500:  # 약 22x22 픽셀 미만은 제외
                    continue
            else:
                box_area = 0
            
            detection_info = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2] if hasattr(box, 'xyxy') else None,
                'area': box_area
            }
            all_detections.append(detection_info)
            
            # 불량 클래스 판정
            is_defect = (
                class_id == 0 or  # 일반적으로 첫 번째 클래스
                class_name.lower() in ['defect', 'defective', 'bad', 'fault'] or
                (yolo_model.names and len(yolo_model.names) == 1)  # 단일 클래스 모델
            )
            
            if is_defect:
                defect_detections.append(detection_info)
                max_confidence = max(max_confidence, confidence)
        
        # 중복/겹치는 박스 제거 (NMS 후처리)
        if len(defect_detections) > 1:
            # 신뢰도가 높은 상위 N개만 사용
            defect_detections = sorted(defect_detections, 
                                     key=lambda x: x['confidence'], 
                                     reverse=True)[:10]  # 최대 10개로 제한
        
        # 신뢰도 임계값 기준으로 불량 판정 (test.py와 동일하게 0.3 사용)
        confidence_threshold = 0.3
        is_defective = len(defect_detections) > 0 and max_confidence >= confidence_threshold
        
        details = f"불량 검출 개수: {len(defect_detections)}, 최대 신뢰도: {max_confidence:.3f}"
        if is_defective:
            details += f" (임계값 {confidence_threshold} 이상으로 불량 판정)"
        else:
            details += f" (임계값 {confidence_threshold} 미만으로 정상 판정)"
        
        logger.info(f"검출 결과: {len(all_detections)}개 객체, 불량: {len(defect_detections)}개, 최대신뢰도: {max_confidence:.3f}")
        
        return {
            'is_defective': is_defective,
            'confidence': max_confidence,
            'defect_count': len(defect_detections),
            'details': details,
            'detections': defect_detections
        }
        
    except Exception as e:
        logger.error(f"검출 결과 분석 실패: {str(e)}")
        return {
            'is_defective': False,
            'confidence': 0.0,
            'defect_count': 0,
            'details': f"분석 오류: {str(e)}"
        }

def create_result_image(image_path, results):
    """검출 결과가 표시된 이미지 생성"""
    try:
        if not results or len(results) == 0:
            return None
        
        # 원본 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        result = results[0]
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # 불량 검출시 빨간색 박스 그리기 (test.py와 동일한 임계값 0.3 사용)
                if class_id == 0 and confidence >= 0.3:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Defect: {confidence:.2f}"
                    cv2.putText(image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 결과 이미지 저장
        result_filename = f"result_{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)
        cv2.imwrite(result_path, image)
        
        return result_filename
        
    except Exception as e:
        logger.error(f"결과 이미지 생성 실패: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': yolo_model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/inspect', methods=['POST'])
def inspect_image():
    """이미지 불량 검사 API"""
    start_time = datetime.now()
    
    try:
        logger.info("이미지 검사 요청 수신")
        
        # 모델 확인
        if yolo_model is None:
            return jsonify({
                'success': False,
                'error': 'YOLO 모델이 로드되지 않았습니다.'
            }), 500
        
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '파일이 업로드되지 않았습니다.'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '파일이 선택되지 않았습니다.'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'지원되지 않는 파일 형식입니다. 지원 형식: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # 파일 저장
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        file.save(file_path)
        file_size = os.path.getsize(file_path)
        logger.info(f"파일 저장 완료: {file_path}")
        
        # 이미지 전처리
        processed_image = preprocess_image(file_path)
        if processed_image is None:
            return jsonify({
                'success': False,
                'error': '이미지 전처리에 실패했습니다.'
            }), 500
        
        # YOLO 검출 실행 (test.py와 동일하게 낮은 임계값으로 시작)
        logger.info("YOLO 검출 시작")
        results = yolo_model(file_path, verbose=False, conf=0.1)
        
        # 결과 분석
        analysis = analyze_detection_results(results)
        
        # 결과 이미지 생성
        result_image_filename = create_result_image(file_path, results)
        
        # 처리 시간 계산
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 응답 데이터 구성
        response_data = {
            'success': True,
            'inspection_result': {
                'is_defective': analysis['is_defective'],
                'confidence': analysis['confidence'],
                'defect_count': analysis['defect_count'],
                'details': analysis['details'],
                'timestamp': start_time.isoformat(),
                'original_filename': filename,
                'processed_filename': unique_filename,
                'processing_time': processing_time
            }
        }
        
        # 검출 상세 정보 추가
        if 'detections' in analysis:
            response_data['inspection_result']['detections'] = analysis['detections']
        
        # 결과 이미지 경로 추가
        if result_image_filename:
            response_data['inspection_result']['result_image'] = result_image_filename
        
        # 로그 저장
        log_data = {
            'timestamp': start_time.isoformat(),
            'original_filename': filename,
            'processed_filename': unique_filename,
            'file_size': file_size,
            'is_defective': analysis['is_defective'],
            'confidence': analysis['confidence'],
            'defect_count': analysis['defect_count'],
            'details': analysis['details'],
            'result_image': result_image_filename,
            'processing_time': processing_time,
            'model_used': 'YOLOv8'
        }
        
        if save_inspection_log(log_data):
            logger.info("검사 로그 저장 완료")
        else:
            logger.warning("검사 로그 저장 실패")
        
        logger.info(f"검사 완료: {'불량' if analysis['is_defective'] else '정상'} (신뢰도: {analysis['confidence']:.3f}, 처리시간: {processing_time:.2f}초)")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"이미지 검사 오류: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'서버 오류가 발생했습니다: {str(e)}'
        }), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    """검사 로그 조회 API"""
    try:
        # 쿼리 파라미터에서 제한 개수 가져오기
        limit = request.args.get('limit', 100, type=int)
        limit = min(limit, 1000)  # 최대 1000개로 제한
        
        logs = get_inspection_logs(limit)
        
        return jsonify({
            'success': True,
            'logs': logs,
            'count': len(logs)
        })
        
    except Exception as e:
        logger.error(f"로그 조회 오류: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'로그 조회에 실패했습니다: {str(e)}'
        }), 500

@app.route('/logs/stats', methods=['GET'])
def get_log_stats():
    """검사 통계 정보 API"""
    try:
        conn = sqlite3.connect('inspection_logs.db')
        cursor = conn.cursor()
        
        # 전체 검사 횟수
        cursor.execute('SELECT COUNT(*) FROM inspection_logs')
        total_inspections = cursor.fetchone()[0]
        
        # 불량 검출 횟수
        cursor.execute('SELECT COUNT(*) FROM inspection_logs WHERE is_defective = 1')
        defective_count = cursor.fetchone()[0]
        
        # 정상 검출 횟수
        normal_count = total_inspections - defective_count
        
        # 평균 신뢰도
        cursor.execute('SELECT AVG(confidence) FROM inspection_logs')
        avg_confidence = cursor.fetchone()[0] or 0
        
        # 평균 처리 시간
        cursor.execute('SELECT AVG(processing_time) FROM inspection_logs')
        avg_processing_time = cursor.fetchone()[0] or 0
        
        # 최근 7일간 검사 횟수
        cursor.execute('''
            SELECT COUNT(*) FROM inspection_logs 
            WHERE datetime(timestamp) >= datetime('now', '-7 days')
        ''')
        recent_inspections = cursor.fetchone()[0]
        
        conn.close()
        
        stats = {
            'total_inspections': total_inspections,
            'defective_count': defective_count,
            'normal_count': normal_count,
            'defective_rate': (defective_count / total_inspections * 100) if total_inspections > 0 else 0,
            'avg_confidence': round(avg_confidence * 100, 2) if avg_confidence else 0,
            'avg_processing_time': round(avg_processing_time, 3) if avg_processing_time else 0,
            'recent_inspections': recent_inspections
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"통계 조회 오류: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'통계 조회에 실패했습니다: {str(e)}'
        }), 500

@app.route('/results/<filename>', methods=['GET'])
def get_result_image(filename):
    """결과 이미지 다운로드"""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({
                'success': False,
                'error': '파일을 찾을 수 없습니다.'
            }), 404
    except Exception as e:
        logger.error(f"파일 전송 오류: {str(e)}")
        return jsonify({
            'success': False,
            'error': '파일 전송에 실패했습니다.'
        }), 500

@app.route('/model/status', methods=['GET'])
def model_status():
    """모델 상태 확인"""
    return jsonify({
        'model_loaded': yolo_model is not None,
        'model_info': {
            'type': 'YOLOv8',
            'classes': yolo_model.names if yolo_model and yolo_model.names else ['defect'],
            'threshold': 0.3  # test.py와 동일한 임계값
        } if yolo_model is not None else None
    })

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("AI 검사 시스템 Flask 서버 시작")
    logger.info("=" * 60)
    
    # 데이터베이스 초기화
    logger.info("데이터베이스 초기화 중...")
    init_database()
    logger.info("데이터베이스 초기화 완료")
    
    # YOLO 모델 로드
    logger.info("YOLO 모델 초기화 중...")
    if load_yolo_model():
        logger.info("YOLO 모델 로드 성공")
    else:
        logger.warning("YOLO 모델 로드 실패 - 기본 모델로 대체")
    
    # Flask 서버 시작
    logger.info("Flask 서버 시작...")
    logger.info("API 엔드포인트:")
    logger.info("  POST /inspect - 이미지 불량 검사")
    logger.info("  GET /health - 서버 상태 확인")
    logger.info("  GET /model/status - 모델 상태 확인")
    logger.info("  GET /logs - 검사 로그 조회")
    logger.info("  GET /logs/stats - 검사 통계 정보")
    logger.info("  GET /results/<filename> - 결과 이미지 다운로드")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

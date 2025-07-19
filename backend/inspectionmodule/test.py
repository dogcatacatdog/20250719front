"""
YOLO 모델 성능 테스트 및 검증 시스템
작성자: AI 시스템 개발팀
버전: 1.0
기능: 훈련된 YOLO 모델의 성능을 testset으로 검증
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import json
from datetime import datetime
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YOLOModelTester:
    """
    YOLO 모델 성능 테스트 클래스
    """
    
    def __init__(self, model_path=None, testset_dir="./testset", confidence_threshold=0.3):
        self.testset_dir = testset_dir
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.test_results = []
        self.ground_truth = []
        self.predictions = []
        self.confidences = []
        
        # 최적 모델 경로 찾기
        self.model_path = self.find_best_model(model_path)
        
        logger.info(f"테스트 설정:")
        logger.info(f"  - 모델 경로: {self.model_path}")
        logger.info(f"  - 테스트셋 경로: {self.testset_dir}")
        logger.info(f"  - 신뢰도 임계값: {self.confidence_threshold}")
        logger.info(f"  - 주의: 과검출 문제로 인해 적절한 임계값 사용")
    
    def find_best_model(self, model_path=None):
        """최적의 훈련된 모델 찾기"""
        if model_path and os.path.exists(model_path):
            return model_path
        
        # 훈련된 모델들 찾기 (우선순위 순서)
        model_candidates = [
            # 1. 실제 훈련된 최종 모델
            "./defect_detection_model_20250719_115333.pt",
            # 2. 다른 타임스탬프 모델들 (패턴 매칭)
        ]
        
        # defect_detection_model_*.pt 패턴으로 모든 모델 찾기
        import glob
        pattern_models = glob.glob("./defect_detection_model_*.pt")
        pattern_models.sort(reverse=True)  # 최신순 정렬
        model_candidates.extend(pattern_models)
        
        # yolo_dataset 폴더 내 모델들도 확인
        yolo_dataset_models = glob.glob("./yolo_dataset/defect_detection_model_*.pt")
        yolo_dataset_models.sort(reverse=True)
        model_candidates.extend(yolo_dataset_models)
        
        # runs/detect 폴더의 best 모델들 (최신순)
        runs_candidates = [
            "./runs/detect/train2222/weights/best.pt",
            "./runs/detect/train222/weights/best.pt", 
            "./runs/detect/train22/weights/best.pt",
            "./runs/detect/train2/weights/best.pt",
            "./runs/detect/train/weights/best.pt",
        ]
        model_candidates.extend(runs_candidates)
        
        # 기본 모델 (마지막 옵션)
        model_candidates.append("./yolov8n.pt")
        
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
    
    def load_model(self):
        """YOLO 모델 로드"""
        try:
            self.model = YOLO(self.model_path)
            logger.info("YOLO 모델 로드 성공")
            
            # 모델 정보 출력
            logger.info(f"모델 클래스: {self.model.names}")
            logger.info(f"모델 클래스 개수: {len(self.model.names) if self.model.names else 'Unknown'}")
            
            return True
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            return False
    
    def load_ground_truth(self):
        """Labels.txt에서 정답 라벨 로드"""
        labels_path = os.path.join(self.testset_dir, "Labels.txt")
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels.txt 파일을 찾을 수 없습니다: {labels_path}")
        
        ground_truth_dict = {}
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 첫 번째 줄은 헤더이므로 스킵
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                file_num = parts[0]
                label = int(parts[1])  # 0=정상, 1=불량
                filename = parts[2]
                ground_truth_dict[filename] = label
        
        logger.info(f"정답 라벨 로드 완료: {len(ground_truth_dict)}개")
        return ground_truth_dict
    
    def predict_single_image(self, image_path):
        """단일 이미지에 대한 예측 수행"""
        try:
            # YOLO 검출 실행 (기본 confidence 0.25 사용)
            results = self.model(image_path, verbose=False, conf=0.1)
            
            if not results or len(results) == 0:
                return False, 0.0, 0, []  # 검출 없음
            
            result = results[0]
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                return False, 0.0, 0, []  # 검출 없음
            
            # 모든 검출 정보 수집
            all_detections = []
            defect_detections = []
            max_confidence = 0.0
            
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id] if self.model.names else f"class_{class_id}"
                
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
                    (self.model.names and len(self.model.names) == 1)  # 단일 클래스 모델
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
            
            # 디버깅 정보 로그
            if all_detections:
                filename = os.path.basename(image_path)
                logger.debug(f"{filename}: 총 {len(all_detections)}개 검출 "
                           f"(필터링 후), 불량 {len(defect_detections)}개, "
                           f"최대 신뢰도: {max_confidence:.3f}")
                
                # 처음 몇 개 이미지는 상세 정보 출력
                if hasattr(self, '_debug_count'):
                    self._debug_count += 1
                else:
                    self._debug_count = 1
                
                if self._debug_count <= 5:  # 처음 5개 이미지만 상세 출력
                    logger.info(f"  {filename}: {len(all_detections)}개 검출 "
                              f"(불량: {len(defect_detections)}개, "
                              f"최대신뢰도: {max_confidence:.3f})")
            
            # 신뢰도 임계값 기준으로 불량 판정
            is_defective = len(defect_detections) > 0 and max_confidence >= self.confidence_threshold
            
            return is_defective, max_confidence, len(defect_detections), all_detections
            
        except Exception as e:
            logger.error(f"이미지 예측 실패 ({image_path}): {str(e)}")
            return False, 0.0, 0, []
    
    def run_test(self):
        """전체 테스트 실행"""
        logger.info("=" * 60)
        logger.info("YOLO 모델 성능 테스트 시작")
        logger.info("=" * 60)
        
        # 모델 로드
        if not self.load_model():
            logger.error("모델 로드 실패로 테스트 중단")
            return False
        
        # 정답 라벨 로드
        try:
            ground_truth_dict = self.load_ground_truth()
        except Exception as e:
            logger.error(f"정답 라벨 로드 실패: {str(e)}")
            return False
        
        # 테스트 이미지들 가져오기
        image_files = [f for f in os.listdir(self.testset_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files = sorted(image_files)  # 정렬
        
        logger.info(f"테스트 이미지 개수: {len(image_files)}")
        
        # 각 이미지에 대해 예측 수행
        correct_predictions = 0
        total_predictions = 0
        detection_summary = {
            'total_detections': 0,
            'defect_detections': 0,
            'no_detection': 0,
            'confidence_distribution': []
        }
        
        for filename in tqdm(image_files, desc="테스트 진행"):
            if filename not in ground_truth_dict:
                logger.warning(f"정답 라벨이 없는 파일: {filename}")
                continue
            
            image_path = os.path.join(self.testset_dir, filename)
            ground_truth_label = ground_truth_dict[filename]
            
            # 예측 수행
            is_defective, confidence, defect_count, all_detections = self.predict_single_image(image_path)
            predicted_label = 1 if is_defective else 0
            
            # 검출 통계 업데이트
            if all_detections:
                detection_summary['total_detections'] += len(all_detections)
                detection_summary['defect_detections'] += defect_count
                detection_summary['confidence_distribution'].extend([d['confidence'] for d in all_detections])
            else:
                detection_summary['no_detection'] += 1
            
            # 결과 저장
            result = {
                'filename': filename,
                'ground_truth': ground_truth_label,
                'predicted': predicted_label,
                'confidence': confidence,
                'defect_count': defect_count,
                'all_detections': all_detections,
                'correct': ground_truth_label == predicted_label
            }
            self.test_results.append(result)
            
            # 메트릭 계산용 리스트 업데이트
            self.ground_truth.append(ground_truth_label)
            self.predictions.append(predicted_label)
            self.confidences.append(confidence)
            
            if ground_truth_label == predicted_label:
                correct_predictions += 1
            total_predictions += 1
            
            # 진행상황 로그 (100개마다)
            if total_predictions % 100 == 0:
                accuracy = correct_predictions / total_predictions * 100
                logger.info(f"진행: {total_predictions}/{len(image_files)} - 현재 정확도: {accuracy:.2f}%")
        
        # 검출 통계 출력
        logger.info("-" * 40)
        logger.info("검출 통계:")
        logger.info(f"전체 이미지: {total_predictions}개")
        logger.info(f"검출된 객체 총합: {detection_summary['total_detections']}개")
        logger.info(f"불량으로 분류된 검출: {detection_summary['defect_detections']}개")
        logger.info(f"아무것도 검출되지 않은 이미지: {detection_summary['no_detection']}개")
        
        if detection_summary['confidence_distribution']:
            confidences = detection_summary['confidence_distribution']
            logger.info(f"신뢰도 분포 - 평균: {np.mean(confidences):.3f}, "
                       f"최대: {np.max(confidences):.3f}, "
                       f"최소: {np.min(confidences):.3f}")
        
        logger.info(f"테스트 완료: {total_predictions}개 이미지 처리")
        return True
    
    def calculate_metrics(self):
        """성능 메트릭 계산"""
        if not self.test_results:
            logger.error("테스트 결과가 없습니다.")
            return None
        
        # 기본 메트릭
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        y_scores = np.array(self.confidences)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 성능 메트릭 계산
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # ROC AUC (신뢰도 점수 사용)
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall AUC
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)
        else:
            roc_auc = 0.0
            pr_auc = 0.0
            fpr, tpr = [], []
            precision_vals, recall_vals = [], []
        
        metrics = {
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'total_samples': len(y_true),
            'defective_samples': int(np.sum(y_true)),
            'normal_samples': int(len(y_true) - np.sum(y_true)),
            'curves': {
                'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
                'precision_recall': {
                    'precision': precision_vals.tolist(), 
                    'recall': recall_vals.tolist()
                }
            }
        }
        
        return metrics
    
    def generate_report(self, metrics):
        """성능 리포트 생성"""
        if not metrics:
            return
        
        logger.info("=" * 60)
        logger.info("YOLO 모델 성능 테스트 결과")
        logger.info("=" * 60)
        
        # 기본 정보
        logger.info(f"모델: {os.path.basename(self.model_path)}")
        logger.info(f"테스트 이미지: {metrics['total_samples']}개")
        logger.info(f"불량 이미지: {metrics['defective_samples']}개")
        logger.info(f"정상 이미지: {metrics['normal_samples']}개")
        logger.info(f"신뢰도 임계값: {self.confidence_threshold}")
        
        logger.info("-" * 40)
        logger.info("성능 메트릭:")
        logger.info(f"정확도 (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"정밀도 (Precision): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        logger.info(f"재현율 (Recall): {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        logger.info(f"특이도 (Specificity): {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        logger.info(f"F1 점수: {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
        
        logger.info("-" * 40)
        logger.info("혼동 행렬 (Confusion Matrix):")
        cm = metrics['confusion_matrix']
        logger.info(f"True Negative (정상→정상): {cm['true_negative']}")
        logger.info(f"False Positive (정상→불량): {cm['false_positive']}")
        logger.info(f"False Negative (불량→정상): {cm['false_negative']}")
        logger.info(f"True Positive (불량→불량): {cm['true_positive']}")
        
        # 에러 분석
        logger.info("-" * 40)
        logger.info("에러 분석:")
        if cm['false_positive'] > 0:
            logger.info(f"과검출 (False Positive): {cm['false_positive']}건 - 정상을 불량으로 잘못 판정")
        if cm['false_negative'] > 0:
            logger.info(f"미검출 (False Negative): {cm['false_negative']}건 - 불량을 정상으로 잘못 판정")
    
    def save_results(self, metrics):
        """결과를 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. JSON 결과 파일
        results_data = {
            'test_info': {
                'model_path': self.model_path,
                'testset_dir': self.testset_dir,
                'confidence_threshold': self.confidence_threshold,
                'timestamp': timestamp,
                'total_images': len(self.test_results)
            },
            'metrics': metrics,
            'detailed_results': self.test_results
        }
        
        results_filename = f"test_results_{timestamp}.json"
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        logger.info(f"상세 결과 저장: {results_filename}")
        
        # 2. CSV 결과 파일
        df = pd.DataFrame(self.test_results)
        csv_filename = f"test_results_{timestamp}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        logger.info(f"CSV 결과 저장: {csv_filename}")
        
        # 3. 성능 메트릭만 따로 저장
        metrics_filename = f"test_metrics_{timestamp}.json"
        with open(metrics_filename, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"메트릭 저장: {metrics_filename}")
    
    def create_visualizations(self, metrics):
        """시각화 생성"""
        try:
            # 한글 폰트 설정
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.style.use('default')
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'YOLO Model Test Results\nModel: {os.path.basename(self.model_path)}', 
                        fontsize=16, fontweight='bold')
            
            # 1. 혼동 행렬
            cm_data = metrics['confusion_matrix']
            cm_matrix = np.array([[cm_data['true_negative'], cm_data['false_positive']],
                                 [cm_data['false_negative'], cm_data['true_positive']]])
            
            sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Predicted Normal', 'Predicted Defective'],
                       yticklabels=['Actual Normal', 'Actual Defective'],
                       ax=axes[0,0])
            axes[0,0].set_title('Confusion Matrix')
            
            # 2. 성능 메트릭 바 차트
            metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
            metric_values = [metrics['accuracy'], metrics['precision'], 
                           metrics['recall'], metrics['specificity'], metrics['f1_score']]
            
            bars = axes[0,1].bar(metric_names, metric_values, 
                                color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
            axes[0,1].set_title('Performance Metrics')
            axes[0,1].set_ylim(0, 1)
            axes[0,1].set_ylabel('Score')
            
            # 값 표시
            for bar, value in zip(bars, metric_values):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{value:.3f}', ha='center', va='bottom')
            
            plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45)
            
            # 3. ROC 곡선
            if len(metrics['curves']['roc']['fpr']) > 0:
                axes[1,0].plot(metrics['curves']['roc']['fpr'], 
                              metrics['curves']['roc']['tpr'], 
                              'b-', label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
                axes[1,0].plot([0, 1], [0, 1], 'r--', label='Random')
                axes[1,0].set_xlabel('False Positive Rate')
                axes[1,0].set_ylabel('True Positive Rate')
                axes[1,0].set_title('ROC Curve')
                axes[1,0].legend()
                axes[1,0].grid(True)
            else:
                axes[1,0].text(0.5, 0.5, 'ROC Curve\nNot Available', 
                              ha='center', va='center', transform=axes[1,0].transAxes)
                axes[1,0].set_title('ROC Curve')
            
            # 4. Precision-Recall 곡선
            if len(metrics['curves']['precision_recall']['precision']) > 0:
                axes[1,1].plot(metrics['curves']['precision_recall']['recall'],
                              metrics['curves']['precision_recall']['precision'],
                              'g-', label=f'PR Curve (AUC = {metrics["pr_auc"]:.3f})')
                axes[1,1].set_xlabel('Recall')
                axes[1,1].set_ylabel('Precision')
                axes[1,1].set_title('Precision-Recall Curve')
                axes[1,1].legend()
                axes[1,1].grid(True)
            else:
                axes[1,1].text(0.5, 0.5, 'PR Curve\nNot Available', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Precision-Recall Curve')
            
            plt.tight_layout()
            
            # 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"test_visualization_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            logger.info(f"시각화 저장: {plot_filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"시각화 생성 실패: {str(e)}")
    
    def test_multiple_thresholds(self):
        """다양한 신뢰도 임계값으로 테스트"""
        if not self.test_results:
            logger.error("테스트 결과가 없습니다. 먼저 run_test()를 실행하세요.")
            return
        
        logger.info("=" * 60)
        logger.info("다양한 신뢰도 임계값 성능 분석")
        logger.info("=" * 60)
        
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
        best_f1 = 0
        best_threshold = 0
        
        results_summary = []
        
        for threshold in thresholds:
            # 현재 임계값으로 재평가
            y_true = []
            y_pred = []
            
            for result in self.test_results:
                y_true.append(result['ground_truth'])
                # 신뢰도 기준으로 재판정
                is_defective = result['confidence'] >= threshold and result['defect_count'] > 0
                y_pred.append(1 if is_defective else 0)
            
            # 메트릭 계산
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                results_summary.append({
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
                })
                
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_threshold = threshold
                
                logger.info(f"임계값 {threshold:.2f}: 정확도={accuracy:.3f}, "
                           f"정밀도={precision:.3f}, 재현율={recall:.3f}, F1={f1_score:.3f}")
        
        logger.info("-" * 40)
        logger.info(f"최적 임계값: {best_threshold} (F1 점수: {best_f1:.3f})")
        
        return results_summary, best_threshold
    
    def analyze_errors(self):
        """에러 케이스 분석"""
        if not self.test_results:
            return
        
        # 에러 케이스 추출
        false_positives = [r for r in self.test_results 
                          if r['ground_truth'] == 0 and r['predicted'] == 1]
        false_negatives = [r for r in self.test_results 
                          if r['ground_truth'] == 1 and r['predicted'] == 0]
        
        logger.info("-" * 40)
        logger.info("에러 케이스 분석:")
        
        if false_positives:
            logger.info(f"\n과검출 (False Positive) - {len(false_positives)}건:")
            for i, fp in enumerate(false_positives[:10]):  # 최대 10개만 표시
                logger.info(f"  {i+1}. {fp['filename']} (신뢰도: {fp['confidence']:.3f})")
            if len(false_positives) > 10:
                logger.info(f"  ... 외 {len(false_positives)-10}건")
        
        if false_negatives:
            logger.info(f"\n미검출 (False Negative) - {len(false_negatives)}건:")
            for i, fn in enumerate(false_negatives[:10]):  # 최대 10개만 표시
                logger.info(f"  {i+1}. {fn['filename']} (신뢰도: {fn['confidence']:.3f})")
            if len(false_negatives) > 10:
                logger.info(f"  ... 외 {len(false_negatives)-10}건")

def main():
    """메인 실행 함수"""
    logger.info("YOLO 모델 테스트 시작")
    
    # 테스터 초기화 (적절한 신뢰도 임계값으로 시작)
    tester = YOLOModelTester(confidence_threshold=0.3)
    
    try:
        # 테스트 실행
        if not tester.run_test():
            logger.error("테스트 실행 실패")
            return
        
        # 다양한 임계값으로 성능 분석
        threshold_results, best_threshold = tester.test_multiple_thresholds()
        
        # 최적 임계값으로 테스터 업데이트
        tester.confidence_threshold = best_threshold
        logger.info(f"최적 임계값 {best_threshold}로 최종 평가 수행")
        
        # 메트릭 계산 (최적 임계값 기준)
        metrics = tester.calculate_metrics()
        if not metrics:
            logger.error("메트릭 계산 실패")
            return
        
        # 결과 리포트
        tester.generate_report(metrics)
        
        # 에러 분석
        tester.analyze_errors()
        
        # 결과 저장
        tester.save_results(metrics)
        
        # 시각화 생성
        tester.create_visualizations(metrics)
        
        logger.info("=" * 60)
        logger.info("YOLO 모델 테스트 완료")
        logger.info(f"권장 신뢰도 임계값: {best_threshold}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()

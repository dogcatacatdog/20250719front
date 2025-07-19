"""
AI 검사 시스템 - YOLO8 어노테이션 모듈
작성자: AI 시스템 개발팀
버전: 1.0
기능: 이미지 어노테이션, 바운딩 박스 설정, 추론
"""

import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_annotation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DefectAnnotator:
    """
    불량 이미지에 바운딩 박스를 설정하는 어노테이션 도구
    """
    
    def __init__(self, image_path, output_dir="annotations", auto_save=True):
        self.image_path = image_path
        self.output_dir = output_dir
        self.auto_save = auto_save
        self.image = None
        self.annotations = []
        self.current_bbox = None
        self.fig = None
        self.ax = None
        self.annotation_complete = False
        
        # 출력 디렉터리 생성
        os.makedirs(output_dir, exist_ok=True)
        
    def load_image(self):
        """이미지 로드"""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {self.image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return self.image
    
    def on_select(self, eclick, erelease):
        """바운딩 박스 선택 시 호출되는 콜백"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # YOLO 형식으로 변환 (정규화된 좌표)
        h, w = self.image.shape[:2]
        x_center = (x1 + x2) / 2 / w
        y_center = (y1 + y2) / 2 / h
        width = abs(x2 - x1) / w
        height = abs(y2 - y1) / h
        
        # 어노테이션 추가 (클래스 0: 불량)
        annotation = {
            'class': 0,
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height
        }
        
        self.annotations.append(annotation)
        logger.info(f"바운딩 박스 추가: {annotation}")
        
        # 바운딩 박스 시각화
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
        self.ax.add_patch(rect)
        self.fig.canvas.draw()
        
        # 자동 저장 및 완료 처리
        if self.auto_save:
            self.save_annotation()
            self.annotation_complete = True
            logger.info("어노테이션 자동 저장 완료 - 다음 이미지로 이동")
            plt.close(self.fig)  # 창 자동 닫기
    
    def on_key_press(self, event):
        """키보드 이벤트 처리"""
        if event.key == 'enter':
            # 엔터키로 현재 어노테이션 저장하고 종료
            if self.annotations:
                self.save_annotation()
            self.annotation_complete = True
            plt.close(self.fig)
            logger.info("엔터키로 어노테이션 완료")
        elif event.key == 'escape':
            # ESC키로 어노테이션 없이 종료
            self.annotation_complete = True
            plt.close(self.fig)
            logger.info("ESC키로 어노테이션 취소")
        elif event.key == 'c':
            # C키로 어노테이션 초기화
            self.clear_annotations()
    
    def annotate_image(self):
        """이미지 어노테이션 인터페이스"""
        if self.image is None:
            self.load_image()
            
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.ax.imshow(self.image)
        
        # 제목과 사용법 표시
        title = f"불량 영역을 드래그로 선택하세요: {os.path.basename(self.image_path)}"
        subtitle = "사용법: 드래그로 박스 그리기 | Enter: 저장&다음 | ESC: 건너뛰기 | C: 초기화"
        self.ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        
        # 바운딩 박스 선택기
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1],  # 왼쪽 마우스 버튼
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        # 키보드 이벤트 연결
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 창 제목 설정 (지원되는 경우에만)
        try:
            self.fig.canvas.manager.set_window_title('AI 검사 시스템 - 불량 어노테이션')
        except AttributeError:
            # 창 제목 설정이 지원되지 않는 경우 무시
            pass
        
        plt.show()
        
        return self.annotation_complete
        
    def save_annotation(self, filename=None):
        """YOLO 형식으로 어노테이션 저장"""
        if filename is None:
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            filename = f"{base_name}.txt"
            
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            for ann in self.annotations:
                f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                       f"{ann['width']:.6f} {ann['height']:.6f}\n")
        
        logger.info(f"어노테이션 저장: {output_path}")
        return output_path
    
    def clear_annotations(self):
        """모든 어노테이션 초기화"""
        self.annotations = []
        if self.ax:
            # 기존 바운딩 박스 제거
            for patch in self.ax.patches:
                patch.remove()
            self.fig.canvas.draw()
        logger.info("어노테이션 초기화 완료")

class BatchAnnotator:
    """
    여러 이미지를 일괄 어노테이션하는 클래스
    """
    
    def __init__(self, json_file, base_dir="./trainset", output_dir="annotations"):
        self.json_file = os.path.join(base_dir, json_file) if not os.path.isabs(json_file) else json_file
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.images_data = self.load_images_data()
        
    def load_images_data(self):
        """JSON 파일에서 이미지 데이터 로드"""
        if not os.path.exists(self.json_file):
            raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {self.json_file}")
        
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['images']
    
    def annotate_defective_images(self, max_images=10, auto_mode=True):
        """불량 이미지들을 순차적으로 어노테이션"""
        logger.info(f"최대 {max_images}개 이미지 어노테이션 시작")
        
        annotated_count = 0
        skipped_count = 0
        
        for i, img_info in enumerate(self.images_data):
            if annotated_count >= max_images:
                break
                
            image_path = os.path.join(self.base_dir, img_info['filename'])
            
            if not os.path.exists(image_path):
                logger.warning(f"이미지 파일을 찾을 수 없습니다: {image_path}")
                continue
            
            print(f"\n=== 이미지 {i+1}/{len(self.images_data)}: {img_info['filename']} ===")
            
            if auto_mode:
                # 자동 모드: 바운딩 박스 그리면 자동으로 다음 이미지
                print("🎯 불량 영역을 드래그로 선택하세요 (자동 저장 모드)")
                print("   - 바운딩 박스 그리기: 마우스로 드래그")
                print("   - 저장 후 다음: Enter 키")
                print("   - 건너뛰기: ESC 키")
                print("   - 초기화: C 키")
                
                annotator = DefectAnnotator(image_path, self.output_dir, auto_save=True)
                completed = annotator.annotate_image()
                
                if completed and annotator.annotations:
                    annotated_count += 1
                    print(f"✅ 어노테이션 완료: {len(annotator.annotations)}개 바운딩 박스")
                else:
                    skipped_count += 1
                    print("⏭️ 건너뛰기")
                    
            else:
                # 수동 모드: 사용자가 직접 선택
                print("1. 어노테이션 시작")
                print("2. 건너뛰기")
                print("3. 종료")
                
                choice = input("선택 (1/2/3): ").strip()
                
                if choice == '1':
                    annotator = DefectAnnotator(image_path, self.output_dir, auto_save=False)
                    annotator.annotate_image()
                    annotator.save_annotation()
                    annotated_count += 1
                    
                elif choice == '2':
                    skipped_count += 1
                    continue
                    
                elif choice == '3':
                    break
                    
                else:
                    print("잘못된 선택입니다. 건너뛰기합니다.")
                    skipped_count += 1
                    continue
        
        print(f"\n🎉 어노테이션 세션 완료!")
        print(f"   - 완료된 이미지: {annotated_count}개")
        print(f"   - 건너뛴 이미지: {skipped_count}개")
        logger.info(f"어노테이션 완료: {annotated_count}개, 건너뛰기: {skipped_count}개")

class YOLOInferencer:
    """
    학습된 YOLO 모델을 사용한 추론 클래스
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """학습된 모델 로드"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"모델 로드 완료: {self.model_path}")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            # 기본 모델로 대체
            self.model = YOLO("yolov8n.pt")
            logger.info("기본 YOLOv8n 모델을 사용합니다.")
    
    def predict_image(self, image_path, conf_threshold=0.5):
        """단일 이미지 추론"""
        if not os.path.exists(image_path):
            raise ValueError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        results = self.model(image_path, conf=conf_threshold)
        return results
    
    def predict_batch(self, image_paths, conf_threshold=0.5):
        """여러 이미지 일괄 추론"""
        results = []
        for image_path in image_paths:
            if os.path.exists(image_path):
                result = self.predict_image(image_path, conf_threshold)
                results.append({
                    'image_path': image_path,
                    'result': result
                })
        return results
    
    def visualize_results(self, image_path, results, save_path=None):
        """추론 결과 시각화"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # 바운딩 박스 그리기
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    
                    # 바운딩 박스 그리기
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # 라벨 추가
                    ax.text(x1, y1-10, f'Defect: {conf:.2f}', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                           fontsize=10, color='white')
        
        ax.set_title(f"불량 탐지 결과: {os.path.basename(image_path)}")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"결과 이미지 저장: {save_path}")
        
        plt.show()

def run_annotation_tool():
    """어노테이션 도구 실행"""
    print("=" * 60)
    print("🎯 AI 검사 시스템 어노테이션 도구")
    print("=" * 60)
    
    print("\n1. 단일 이미지 어노테이션")
    print("2. 불량 이미지 일괄 어노테이션")
    print("3. 모델 추론 테스트")
    
    choice = input("\n선택하세요 (1/2/3): ").strip()
    
    if choice == '1':
        image_path = input("이미지 경로를 입력하세요: ").strip()
        if os.path.exists(image_path):
            annotator = DefectAnnotator(image_path)
            annotator.annotate_image()
            annotator.save_annotation()
        else:
            print("이미지 파일을 찾을 수 없습니다.")
    
    elif choice == '2':
        # trainset 폴더 내의 defective_images.json 파일 확인
        defective_json = "trainset/defective_images.json"
        if os.path.exists(defective_json):
            max_images = int(input("최대 어노테이션할 이미지 수: ") or "5")
            
            print("\n어노테이션 모드 선택:")
            print("1. 자동 모드 (바운딩 박스 그리면 자동으로 다음 이미지)")
            print("2. 수동 모드 (사용자가 직접 진행 제어)")
            mode_choice = input("선택 (1/2): ").strip()
            
            auto_mode = mode_choice == '1'
            
            batch_annotator = BatchAnnotator("defective_images.json", base_dir="trainset")
            batch_annotator.annotate_defective_images(max_images, auto_mode=auto_mode)
        else:
            print(f"defective_images.json 파일을 찾을 수 없습니다: {defective_json}")
            print("현재 작업 디렉터리:", os.getcwd())
    
    elif choice == '3':
        model_path = input("모델 경로 (엔터: 기본 모델): ").strip() or "yolov8n.pt"
        image_path = input("테스트할 이미지 경로: ").strip()
        
        if os.path.exists(image_path):
            inferencer = YOLOInferencer(model_path)
            results = inferencer.predict_image(image_path)
            inferencer.visualize_results(image_path, results)
        else:
            print("이미지 파일을 찾을 수 없습니다.")
    
    else:
        print("잘못된 선택입니다.")

def main():
    """메인 실행 함수"""
    run_annotation_tool()

if __name__ == "__main__":
    main()
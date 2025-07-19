"""
AI 검사 시스템 - YOLO8 학습 모듈
작성자: AI 시스템 개발팀
버전: 1.0
기능: 데이터셋 준비, 모델 학습, 검증
"""

import json
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm
import logging
from datetime import datetime

# 로깅 설정 (이모지 제거하여 인코딩 문제 해결)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YOLODatasetPreparer:
    """
    YOLO 학습용 데이터셋 준비 클래스
    """
    
    def __init__(self, json_files, base_dir="./trainset"):
        self.json_files = json_files
        self.base_dir = base_dir
        self.dataset_dir = "yolo_dataset"
        self.classes = ["defect"]  # 불량 클래스
        
    def create_yolo_structure(self):
        """YOLO 데이터셋 폴더 구조 생성"""
        dirs = [
            f"{self.dataset_dir}/images/train",
            f"{self.dataset_dir}/images/val",
            f"{self.dataset_dir}/labels/train",
            f"{self.dataset_dir}/labels/val"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info("YOLO 데이터셋 폴더 구조 생성 완료")
    
    def load_dataset(self, json_file):
        """JSON 파일에서 데이터셋 로드"""
        # 절대 경로가 아닌 경우 base_dir와 결합
        if not os.path.isabs(json_file):
            json_file = os.path.join(self.base_dir, json_file)
            
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_file}")
            
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def prepare_defective_images(self):
        """불량 이미지 준비 및 어노테이션 (기존 어노테이션 파일 사용)"""
        # 불량 이미지 JSON 로드
        defective_data = self.load_dataset("defective_images.json")
        defective_images = defective_data['images']
        
        logger.info(f"불량 이미지 {len(defective_images)}개 처리 시작")
        
        # 기존 어노테이션 파일들을 사용
        annotated_images = []
        # inspectionmodule/annotations 폴더의 절대 경로
        current_dir = os.path.dirname(os.path.abspath(__file__))
        annotation_dir = os.path.join(current_dir, "annotations")
        
        if not os.path.exists(annotation_dir):
            logger.warning(f"어노테이션 폴더를 찾을 수 없습니다: {annotation_dir}")
            # 어노테이션 파일이 없으면 기본 어노테이션 생성
            for img_info in tqdm(defective_images[:50], desc="기본 어노테이션 생성"):
                image_path = os.path.join(self.base_dir, img_info['filename'])
                if os.path.exists(image_path):
                    annotation_path = self.create_default_annotation(img_info['filename'])
                    annotated_images.append({
                        'image_path': image_path,
                        'annotation_path': annotation_path,
                        'info': img_info
                    })
        else:
            # 기존 어노테이션 파일들 사용
            for img_info in tqdm(defective_images, desc="어노테이션 파일 확인"):
                image_path = os.path.join(self.base_dir, img_info['filename'])
                base_name = os.path.splitext(img_info['filename'])[0]
                annotation_path = os.path.join(annotation_dir, f"{base_name}.txt")
                
                if os.path.exists(image_path) and os.path.exists(annotation_path):
                    annotated_images.append({
                        'image_path': image_path,
                        'annotation_path': annotation_path,
                        'info': img_info
                    })
                    logger.debug(f"어노테이션 파일 발견: {annotation_path}")
            
            logger.info(f"기존 어노테이션 파일 {len(annotated_images)}개 발견")
        
        return annotated_images
    
    def prepare_normal_images(self):
        """정상 이미지 준비 (배경 클래스로 사용)"""
        normal_data = self.load_dataset("normal_images.json")
        normal_images = normal_data['images']
        
        logger.info(f"정상 이미지 {len(normal_images)}개 처리 시작")
        
        normal_image_list = []
        
        # 불량 이미지와 균형을 맞추기 위해 더 많은 정상 이미지 사용
        # 불량:정상 = 1:2~3 비율로 설정 (YOLO 학습에서 배경 이미지가 중요)
        sample_count = min(450, len(normal_images))  # 최대 450개 사용 (불량 150개의 3배)
        sampled_normal = normal_images[:sample_count]
        
        logger.info(f"정상 이미지 {sample_count}개 선택 (전체 {len(normal_images)}개 중)")
        
        for img_info in tqdm(sampled_normal, desc="정상 이미지 준비"):
            image_path = os.path.join(self.base_dir, img_info['filename'])
            
            if os.path.exists(image_path):
                # 정상 이미지는 빈 어노테이션 파일 생성
                annotation_path = self.create_empty_annotation(img_info['filename'])
                normal_image_list.append({
                    'image_path': image_path,
                    'annotation_path': annotation_path,
                    'info': img_info
                })
        
        logger.info(f"정상 이미지 {len(normal_image_list)}개 준비 완료")
        logger.info(f"데이터 비율 - 불량:정상 = 150:{len(normal_image_list)} (약 1:{len(normal_image_list)//150})")
        return normal_image_list
    
    def create_default_annotation(self, filename):
        """기본 어노테이션 생성 (전체 이미지를 불량으로 간주)"""
        base_name = os.path.splitext(filename)[0]
        annotation_dir = "annotations"
        os.makedirs(annotation_dir, exist_ok=True)
        
        annotation_path = os.path.join(annotation_dir, f"{base_name}.txt")
        
        # 이미지 중앙의 80% 영역을 불량으로 가정
        with open(annotation_path, 'w') as f:
            f.write("0 0.5 0.5 0.8 0.8\n")  # 클래스 0, 중앙 80% 영역
        
        return annotation_path
    
    def create_empty_annotation(self, filename):
        """정상 이미지용 빈 어노테이션 생성"""
        base_name = os.path.splitext(filename)[0]
        annotation_dir = "annotations"
        os.makedirs(annotation_dir, exist_ok=True)
        
        annotation_path = os.path.join(annotation_dir, f"{base_name}.txt")
        
        # 빈 파일 생성 (정상 이미지이므로 객체 없음)
        with open(annotation_path, 'w') as f:
            pass  # 빈 파일
        
        return annotation_path
    
    def split_dataset(self, all_images, train_ratio=0.8):
        """데이터셋을 훈련/검증용으로 분할"""
        train_imgs, val_imgs = train_test_split(
            all_images, 
            train_size=train_ratio, 
            random_state=42,
            stratify=None  # 불량/정상 비율 고려하지 않음
        )
        
        logger.info(f"데이터셋 분할: 훈련 {len(train_imgs)}개, 검증 {len(val_imgs)}개")
        return train_imgs, val_imgs
    
    def copy_files_to_yolo_format(self, train_imgs, val_imgs):
        """YOLO 형식으로 파일 복사"""
        # 훈련 데이터 복사
        for img_data in tqdm(train_imgs, desc="훈련 데이터 복사"):
            src_img = img_data['image_path']
            src_ann = img_data['annotation_path']
            
            dst_img = os.path.join(self.dataset_dir, "images/train", 
                                 os.path.basename(src_img))
            dst_ann = os.path.join(self.dataset_dir, "labels/train", 
                                 os.path.basename(src_ann))
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            if os.path.exists(src_ann):
                shutil.copy2(src_ann, dst_ann)
        
        # 검증 데이터 복사
        for img_data in tqdm(val_imgs, desc="검증 데이터 복사"):
            src_img = img_data['image_path']
            src_ann = img_data['annotation_path']
            
            dst_img = os.path.join(self.dataset_dir, "images/val", 
                                 os.path.basename(src_img))
            dst_ann = os.path.join(self.dataset_dir, "labels/val", 
                                 os.path.basename(src_ann))
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            if os.path.exists(src_ann):
                shutil.copy2(src_ann, dst_ann)
    
    def create_yaml_config(self):
        """YOLO 학습용 YAML 설정 파일 생성"""
        config = {
            'path': os.path.abspath(self.dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        config_path = f"{self.dataset_dir}/dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"YAML 설정 파일 생성: {config_path}")
        return config_path

class YOLOTrainer:
    """
    YOLO8 모델 학습 클래스
    """
    
    def __init__(self, dataset_config, model_name="yolov8n.pt"):
        self.dataset_config = dataset_config
        self.model_name = model_name
        self.model = None
        
    def initialize_model(self):
        """YOLO 모델 초기화"""
        self.model = YOLO(self.model_name)
        logger.info(f"YOLO 모델 로드: {self.model_name}")
        
    def train(self, epochs=100, batch_size=16, imgsz=640):
        """모델 학습"""
        if self.model is None:
            self.initialize_model()
            
        logger.info("YOLO 모델 학습 시작")
        logger.info(f"설정: epochs={epochs}, batch_size={batch_size}, imgsz={imgsz}")
        
        # GPU 사용 가능 여부 확인
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"사용 디바이스: {device}")
        
        # 학습 실행
        results = self.model.train(
            data=self.dataset_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            save=True,
            plots=True,
            device=device,
            patience=10,  # 조기 종료 설정
            save_period=10  # 10 에포크마다 저장
        )
        
        logger.info("모델 학습 완료")
        return results
    
    def validate(self):
        """모델 검증"""
        if self.model is None:
            raise ValueError("모델이 초기화되지 않았습니다.")
            
        results = self.model.val()
        logger.info("모델 검증 완료")
        return results
    
    def save_model(self, path="best_defect_model.pt"):
        """학습된 모델 저장"""
        if self.model is None:
            raise ValueError("모델이 초기화되지 않았습니다.")
            
        self.model.save(path)
        logger.info(f"모델 저장: {path}")

class BatchTrainer:
    """
    배치 단위 학습 관리 클래스
    """
    
    def __init__(self, preparer, trainer):
        self.preparer = preparer
        self.trainer = trainer
        
    def run_batch_training(self, batch_size=50, epochs_per_batch=25):
        """배치 단위로 점진적 학습 실행"""
        logger.info("배치 단위 학습 시작")
        
        # 1단계: 불량 이미지 집중 학습
        logger.info("1단계: 불량 이미지 집중 학습")
        defective_images = self.preparer.prepare_defective_images()
        logger.info(f"불량 이미지 {len(defective_images)}개로 초기 학습")
        
        if defective_images:
            train_defect, val_defect = self.preparer.split_dataset(defective_images)
            self.preparer.copy_files_to_yolo_format(train_defect, val_defect)
            config_path = self.preparer.create_yaml_config()
            
            # 불량 이미지만으로 초기 학습
            self.trainer.dataset_config = config_path
            results1 = self.trainer.train(epochs=epochs_per_batch, batch_size=8)
            logger.info("불량 이미지 학습 완료")
        
        # 2단계: 정상 이미지 추가 학습 (균형잡힌 데이터셋)
        logger.info("2단계: 정상 + 불량 통합 학습")
        normal_images = self.preparer.prepare_normal_images()
        
        if normal_images:
            # 불량 + 정상 이미지 통합
            all_images = defective_images + normal_images
            total_count = len(all_images)
            defect_ratio = len(defective_images) / total_count * 100
            normal_ratio = len(normal_images) / total_count * 100
            
            logger.info(f"통합 데이터셋: 총 {total_count}개")
            logger.info(f"  - 불량: {len(defective_images)}개 ({defect_ratio:.1f}%)")
            logger.info(f"  - 정상: {len(normal_images)}개 ({normal_ratio:.1f}%)")
            
            train_all, val_all = self.preparer.split_dataset(all_images)
            
            # 기존 데이터셋 정리 후 새로 복사
            if os.path.exists(self.preparer.dataset_dir):
                shutil.rmtree(self.preparer.dataset_dir)
            self.preparer.create_yolo_structure()
            self.preparer.copy_files_to_yolo_format(train_all, val_all)
            config_path = self.preparer.create_yaml_config()
            
            # 통합 데이터셋으로 추가 학습 (배치 크기 증가)
            self.trainer.dataset_config = config_path 
            results2 = self.trainer.train(epochs=epochs_per_batch, batch_size=16)  # 더 많은 데이터로 배치 크기 증가
            logger.info("통합 데이터셋 학습 완료")
        
        # 3단계: 최종 미세 조정
        logger.info("3단계: 최종 미세 조정")
        results3 = self.trainer.train(epochs=epochs_per_batch//2, batch_size=8)
        
        # 최종 검증
        validation_results = self.trainer.validate()
        
        # 모델 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"defect_detection_model_{timestamp}.pt"
        self.trainer.save_model(model_path)
        
        logger.info("배치 학습 완료!")
        logger.info(f"최종 데이터셋 구성:")
        logger.info(f"  - 불량 이미지: {len(defective_images)}개")
        logger.info(f"  - 정상 이미지: {len(normal_images) if normal_images else 0}개")
        logger.info(f"  - 총 학습 데이터: {len(defective_images) + len(normal_images) if normal_images else len(defective_images)}개")
        
        return {
            'defect_training': results1,
            'combined_training': results2,
            'fine_tuning': results3,
            'validation': validation_results,
            'model_path': model_path,
            'dataset_summary': {
                'defective_count': len(defective_images),
                'normal_count': len(normal_images) if normal_images else 0,
                'total_count': len(defective_images) + (len(normal_images) if normal_images else 0)
            }
        }

def main():
    """메인 실행 함수"""
    logger.info("=" * 60)
    logger.info("AI 검사 시스템 YOLO8 배치 학습 시작")
    logger.info("=" * 60)
    
    try:
        # 어노테이션 폴더 상태 확인
        annotation_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotations")
        if os.path.exists(annotation_dir):
            annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]
            logger.info(f"기존 어노테이션 파일 {len(annotation_files)}개 발견")
        else:
            logger.info("어노테이션 폴더가 없습니다. 기본 어노테이션을 생성합니다.")
        
        # 1. 데이터셋 준비자 초기화
        preparer = YOLODatasetPreparer([
            "defective_images.json",
            "normal_images.json"
        ])
        
        # YOLO 폴더 구조 생성
        preparer.create_yolo_structure()
        
        # 2. 학습자 초기화
        trainer = YOLOTrainer("", "yolov8n.pt")  # 설정 파일은 나중에 설정
        
        # 3. 배치 학습 실행
        batch_trainer = BatchTrainer(preparer, trainer)
        results = batch_trainer.run_batch_training(
            batch_size=50,
            epochs_per_batch=20
        )
        
        logger.info("=" * 60)
        logger.info("YOLO8 배치 학습 완료!")
        logger.info(f"저장된 모델: {results['model_path']}")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
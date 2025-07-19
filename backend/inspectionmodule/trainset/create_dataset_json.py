import json
import os

def create_dataset_json():
    """
    Labels.txt 파일을 읽어서 이미지 파일명과 라벨(정상/불량)을 매핑한 JSON 파일을 생성합니다.
    """
    
    # 파일 경로 설정
    labels_file = "Labels.txt"
    output_file = "dataset_labels.json"
    
    # 결과를 저장할 딕셔너리
    dataset = {
        "metadata": {
            "description": "AI 검사 시스템 훈련 데이터셋",
            "total_images": 0,
            "normal_count": 0,
            "defective_count": 0,
            "created_by": "Dataset Labeler",
            "version": "1.0"
        },
        "images": [],
        "labels": {
            "0": "정상",
            "1": "불량"
        }
    }
    
    try:
        # Labels.txt 파일 읽기
        with open(labels_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 첫 번째 줄은 헤더이므로 건너뛰기
        for line in lines[1:]:
            line = line.strip()
            if not line:  # 빈 줄 건너뛰기
                continue
                
            # 탭으로 분리된 데이터 파싱
            parts = line.split('\t')
            
            if len(parts) >= 3:
                image_id = parts[0].strip()
                label = int(parts[1].strip())  # 0: 정상, 1: 불량
                filename = parts[2].strip()
                
                # 이미지 정보 추가
                image_info = {
                    "id": image_id,
                    "filename": filename,
                    "label": label,
                    "status": "정상" if label == 0 else "불량",
                    "path": f"./trainset/{filename}"
                }
                
                dataset["images"].append(image_info)
                
                # 카운트 업데이트
                if label == 0:
                    dataset["metadata"]["normal_count"] += 1
                else:
                    dataset["metadata"]["defective_count"] += 1
        
        # 총 이미지 수 업데이트
        dataset["metadata"]["total_images"] = len(dataset["images"])
        
        # JSON 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        # 결과 출력
        print(f"✅ 데이터셋 JSON 파일이 성공적으로 생성되었습니다: {output_file}")
        print(f"📊 통계:")
        print(f"   - 총 이미지 수: {dataset['metadata']['total_images']:,}")
        print(f"   - 정상 이미지: {dataset['metadata']['normal_count']:,}")
        print(f"   - 불량 이미지: {dataset['metadata']['defective_count']:,}")
        print(f"   - 불량률: {(dataset['metadata']['defective_count'] / dataset['metadata']['total_images'] * 100):.2f}%")
        
        # 불량 이미지만 따로 추출한 JSON도 생성
        create_defective_only_json(dataset)
        
        return dataset
        
    except FileNotFoundError:
        print(f"❌ 오류: {labels_file} 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {str(e)}")
        return None

def create_defective_only_json(dataset):
    """
    불량 이미지만 따로 추출한 JSON 파일을 생성합니다.
    """
    defective_dataset = {
        "metadata": {
            "description": "AI 검사 시스템 불량 이미지 데이터셋",
            "total_images": dataset["metadata"]["defective_count"],
            "defective_count": dataset["metadata"]["defective_count"],
            "created_by": "Dataset Labeler",
            "version": "1.0"
        },
        "images": [img for img in dataset["images"] if img["label"] == 1],
        "labels": {
            "1": "불량"
        }
    }
    
    output_file = "defective_images.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(defective_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"📁 불량 이미지 전용 JSON 파일도 생성되었습니다: {output_file}")

def create_normal_only_json(dataset):
    """
    정상 이미지만 따로 추출한 JSON 파일을 생성합니다.
    """
    normal_dataset = {
        "metadata": {
            "description": "AI 검사 시스템 정상 이미지 데이터셋",
            "total_images": dataset["metadata"]["normal_count"],
            "normal_count": dataset["metadata"]["normal_count"],
            "created_by": "Dataset Labeler",
            "version": "1.0"
        },
        "images": [img for img in dataset["images"] if img["label"] == 0],
        "labels": {
            "0": "정상"
        }
    }
    
    output_file = "normal_images.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(normal_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"📁 정상 이미지 전용 JSON 파일도 생성되었습니다: {output_file}")

if __name__ == "__main__":
    print("🚀 AI 검사 시스템 데이터셋 라벨링 시작...")
    print("=" * 50)
    
    # 현재 작업 디렉터리 확인
    current_dir = os.getcwd()
    print(f"📁 작업 디렉터리: {current_dir}")
    
    # JSON 파일 생성
    dataset = create_dataset_json()
    
    if dataset:
        # 정상 이미지 전용 JSON도 생성
        create_normal_only_json(dataset)
        
        print("=" * 50)
        print("✨ 모든 작업이 완료되었습니다!")
        print("\n생성된 파일들:")
        print("  1. dataset_labels.json - 전체 데이터셋")
        print("  2. defective_images.json - 불량 이미지만")
        print("  3. normal_images.json - 정상 이미지만")

"""
간단한 YOLO 모델 검출 테스트
실제로 모델이 무엇을 검출하는지 확인
"""

from ultralytics import YOLO
import os
import cv2

def quick_test():
    print("=" * 50)
    print("YOLO 모델 빠른 검출 테스트")
    print("=" * 50)
    
    # 모델 경로들 시도
    model_paths = [
        "./runs/detect/train/weights/best.pt",
        "./runs/detect/train2/weights/best.pt", 
        "./defect_detection_model_final.pt",
        "./yolov8n.pt"
    ]
    
    model = None
    model_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = YOLO(path)
                model_path = path
                print(f"✅ 모델 로드 성공: {path}")
                break
            except Exception as e:
                print(f"❌ 모델 로드 실패 ({path}): {e}")
                continue
    
    if model is None:
        print("❌ 사용 가능한 모델이 없습니다.")
        return
    
    # 모델 정보 출력
    print(f"모델 클래스: {model.names}")
    print(f"클래스 개수: {len(model.names) if model.names else 'Unknown'}")
    
    # 테스트 이미지들 확인
    testset_dir = "./testset"
    if not os.path.exists(testset_dir):
        print(f"❌ 테스트셋 폴더가 없습니다: {testset_dir}")
        return
    
    # 처음 10개 이미지로 테스트
    image_files = [f for f in os.listdir(testset_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:10]
    
    print(f"\n처음 {len(image_files)}개 이미지로 테스트:")
    print("-" * 30)
    
    total_detections = 0
    
    for i, filename in enumerate(image_files):
        image_path = os.path.join(testset_dir, filename)
        
        try:
            # 매우 낮은 confidence로 검출
            results = model(image_path, conf=0.01, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    print(f"{i+1:2d}. {filename}: {len(boxes)}개 검출")
                    
                    for j, box in enumerate(boxes[:3]):  # 최대 3개만 표시
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id] if model.names else f"class_{class_id}"
                        print(f"    -> {class_name} (ID:{class_id}) 신뢰도:{confidence:.3f}")
                        total_detections += 1
                else:
                    print(f"{i+1:2d}. {filename}: 검출 없음")
            else:
                print(f"{i+1:2d}. {filename}: 결과 없음")
                
        except Exception as e:
            print(f"{i+1:2d}. {filename}: 오류 - {e}")
    
    print("-" * 30)
    print(f"총 검출 개수: {total_detections}")
    
    if total_detections == 0:
        print("\n🚨 문제 진단:")
        print("1. 모델이 아무것도 검출하지 못했습니다.")
        print("2. 가능한 원인:")
        print("   - 모델이 제대로 훈련되지 않음")
        print("   - 클래스 설정 문제")
        print("   - 이미지 전처리 문제")
        print("   - 신뢰도 임계값 문제")
        print("\n💡 해결 방안:")
        print("1. 훈련 로그 확인")
        print("2. 다른 confidence 값 시도")
        print("3. 모델 재훈련 고려")
    else:
        print(f"\n✅ 모델이 정상적으로 작동합니다!")
        print(f"   권장 confidence 임계값: 0.1~0.3")

if __name__ == "__main__":
    quick_test()

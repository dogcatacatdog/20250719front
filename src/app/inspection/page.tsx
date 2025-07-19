"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

export default function InspectionPage() {
  const [isInspecting, setIsInspecting] = useState(false);
  const [inspectionResult, setInspectionResult] = useState<any>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [serverStatus, setServerStatus] = useState<
    "checking" | "online" | "offline"
  >("checking");

  // 서버 상태 확인
  useEffect(() => {
    checkServerStatus();
  }, []);

  const checkServerStatus = async () => {
    try {
      const response = await fetch("http://localhost:5000/health");
      if (response.ok) {
        const data = await response.json();
        setServerStatus(data.status === "healthy" ? "online" : "offline");
      } else {
        setServerStatus("offline");
      }
    } catch (error) {
      setServerStatus("offline");
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleInspection = async () => {
    if (!selectedFile) return;

    setIsInspecting(true);
    setInspectionResult(null);

    try {
      // FormData로 파일 준비
      const formData = new FormData();
      formData.append("file", selectedFile);

      // Flask 백엔드로 POST 요청
      const response = await fetch("http://localhost:5000/inspect", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        // Flask 응답을 프론트엔드 형식으로 변환
        const inspectionData = result.inspection_result;
        const formattedResult = {
          result: inspectionData.is_defective ? "fail" : "pass",
          confidence: Math.round(inspectionData.confidence * 100 * 10) / 10,
          details: inspectionData.details,
          timestamp: new Date(inspectionData.timestamp).toLocaleString("ko-KR"),
          defect_count: inspectionData.defect_count,
          detections: inspectionData.detections || [],
          result_image: inspectionData.result_image || null,
          original_filename: inspectionData.original_filename,
        };
        setInspectionResult(formattedResult);
      } else {
        throw new Error(result.error || "검사에 실패했습니다.");
      }
    } catch (error) {
      console.error("검사 오류:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "알 수 없는 오류가 발생했습니다.";
      const errorResult = {
        result: "error",
        confidence: 0,
        details: `오류가 발생했습니다: ${errorMessage}`,
        timestamp: new Date().toLocaleString("ko-KR"),
        defect_count: 0,
        detections: [],
      };
      setInspectionResult(errorResult);
    } finally {
      setIsInspecting(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <Link
                href="/"
                className="flex items-center space-x-2 hover:text-gray-300"
              >
                <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                  <span className="text-black text-sm font-bold">AI</span>
                </div>
                <span className="text-lg font-semibold">← 홈으로</span>
              </Link>
            </div>
            <h1 className="text-xl font-bold">AI 검사하기</h1>
            <div className="flex items-center space-x-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  serverStatus === "online"
                    ? "bg-green-500"
                    : serverStatus === "offline"
                    ? "bg-red-500"
                    : "bg-yellow-500 animate-pulse"
                }`}
              ></div>
              <span className="text-sm">
                {serverStatus === "online"
                  ? "AI 서버 연결됨"
                  : serverStatus === "offline"
                  ? "AI 서버 연결 안됨"
                  : "서버 상태 확인 중..."}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4">AI 품질 검사</h2>
          <p className="text-xl text-gray-300">
            이미지를 업로드하여 AI 검사를 시작하세요
          </p>
        </div>

        <div className="bg-white rounded-xl p-8 text-black">
          {/* File Upload Section */}
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center mb-8">
            <div className="mb-4">
              <span className="text-4xl">📁</span>
            </div>
            <h3 className="text-lg font-semibold mb-2">
              검사할 이미지를 선택하세요
            </h3>
            <p className="text-gray-600 mb-4">
              JPG, PNG, GIF, BMP, TIFF, WebP 파일을 지원합니다 (최대 50MB)
            </p>

            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="bg-blue-600 text-white px-6 py-3 rounded-lg cursor-pointer hover:bg-blue-700 transition-colors inline-block"
            >
              파일 선택하기
            </label>

            {selectedFile && (
              <div className="mt-4 p-4 bg-gray-100 rounded-lg">
                <div className="flex items-center justify-center mb-2">
                  <span className="text-2xl mr-2">🖼️</span>
                  <p className="font-medium">
                    선택된 파일: {selectedFile.name}
                  </p>
                </div>
                <p className="text-sm text-gray-600">
                  크기: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
                <p className="text-sm text-gray-600">
                  타입: {selectedFile.type}
                </p>
                {/* 이미지 미리보기 */}
                <div className="mt-3">
                  <img
                    src={URL.createObjectURL(selectedFile)}
                    alt="미리보기"
                    className="max-w-xs max-h-40 mx-auto rounded border border-gray-300"
                    onLoad={(e) => {
                      // 메모리 누수 방지를 위해 URL 해제는 컴포넌트 언마운트시에 처리
                    }}
                  />
                  <p className="text-xs text-gray-500 mt-1">미리보기</p>
                </div>
              </div>
            )}
          </div>

          {/* Inspection Button */}
          <div className="text-center mb-8">
            {serverStatus === "offline" && (
              <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-700">
                  ⚠️ AI 서버에 연결할 수 없습니다. 서버가 실행 중인지
                  확인해주세요.
                </p>
                <button
                  onClick={checkServerStatus}
                  className="mt-2 text-sm bg-red-600 text-white px-3 py-1 rounded hover:bg-red-700"
                >
                  다시 연결 시도
                </button>
              </div>
            )}

            <button
              onClick={handleInspection}
              disabled={
                !selectedFile || isInspecting || serverStatus !== "online"
              }
              className={`px-8 py-4 rounded-lg font-semibold text-lg transition-all ${
                !selectedFile || isInspecting || serverStatus !== "online"
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-black text-white hover:bg-gray-800"
              }`}
            >
              {isInspecting
                ? "AI 검사 중..."
                : serverStatus !== "online"
                ? "서버 연결 필요"
                : "AI 검사 시작하기"}
            </button>

            {serverStatus !== "online" && (
              <p className="mt-2 text-sm text-gray-500">
                Flask 서버 (http://localhost:5000)가 실행되어야 합니다.
              </p>
            )}
          </div>

          {/* Loading Animation */}
          {isInspecting && (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
              <p className="mt-4 text-gray-600">
                AI가 이미지를 분석하고 있습니다...
              </p>
              <p className="mt-2 text-sm text-gray-500">
                YOLO 모델로 불량 검출 중입니다. 잠시만 기다려주세요.
              </p>
            </div>
          )}

          {/* Results Section */}
          {inspectionResult && (
            <div className="border-t pt-8">
              <h3 className="text-2xl font-bold mb-6 text-center">검사 결과</h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-50 rounded-lg p-6">
                  <h4 className="font-semibold text-gray-700 mb-2">
                    검사 결과
                  </h4>
                  <div
                    className={`text-2xl font-bold ${
                      inspectionResult.result === "pass"
                        ? "text-green-600"
                        : inspectionResult.result === "fail"
                        ? "text-red-600"
                        : "text-orange-600"
                    }`}
                  >
                    {inspectionResult.result === "pass"
                      ? "✅ 정상 (합격)"
                      : inspectionResult.result === "fail"
                      ? "❌ 불량 (불합격)"
                      : "⚠️ 검사 오류"}
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-6">
                  <h4 className="font-semibold text-gray-700 mb-2">신뢰도</h4>
                  <div className="text-2xl font-bold text-blue-600">
                    {inspectionResult.confidence}%
                  </div>
                </div>
              </div>

              {/* 불량 검출 정보 추가 */}
              {inspectionResult.result === "fail" &&
                inspectionResult.defect_count > 0 && (
                  <div className="mt-6 bg-red-50 rounded-lg p-6 border border-red-200">
                    <h4 className="font-semibold text-red-700 mb-2">
                      불량 검출 정보
                    </h4>
                    <p className="text-red-800">
                      검출된 불량 개수:{" "}
                      <span className="font-bold">
                        {inspectionResult.defect_count}개
                      </span>
                    </p>
                    {inspectionResult.detections &&
                      inspectionResult.detections.length > 0 && (
                        <div className="mt-3">
                          <p className="text-sm text-red-600 mb-2">
                            검출 상세:
                          </p>
                          {inspectionResult.detections.map(
                            (detection: any, index: number) => (
                              <div
                                key={index}
                                className="text-sm bg-white p-2 rounded border border-red-100 mb-1"
                              >
                                불량 #{index + 1}: 신뢰도{" "}
                                {(detection.confidence * 100).toFixed(1)}%
                              </div>
                            )
                          )}
                        </div>
                      )}
                  </div>
                )}

              <div className="mt-6 bg-gray-50 rounded-lg p-6">
                <h4 className="font-semibold text-gray-700 mb-2">세부 분석</h4>
                <p className="text-gray-800">{inspectionResult.details}</p>
                <p className="text-sm text-gray-500 mt-2">
                  검사 시간: {inspectionResult.timestamp}
                </p>
                {inspectionResult.original_filename && (
                  <p className="text-sm text-gray-500">
                    원본 파일: {inspectionResult.original_filename}
                  </p>
                )}
              </div>

              {/* 결과 이미지 표시 */}
              {inspectionResult.result_image && (
                <div className="mt-6 bg-gray-50 rounded-lg p-6">
                  <h4 className="font-semibold text-gray-700 mb-4">
                    검사 결과 이미지
                  </h4>
                  <div className="text-center">
                    <img
                      src={`http://localhost:5000/results/${inspectionResult.result_image}`}
                      alt="검사 결과"
                      className="max-w-full h-auto rounded-lg border border-gray-300"
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                        console.error("결과 이미지 로드 실패");
                      }}
                    />
                    <p className="text-sm text-gray-500 mt-2">
                      불량 영역이 빨간색 박스로 표시됩니다
                    </p>
                  </div>
                </div>
              )}

              <div className="text-center mt-8">
                <button
                  onClick={() => {
                    setInspectionResult(null);
                    setSelectedFile(null);
                    // 파일 입력 초기화
                    const fileInput = document.getElementById(
                      "file-upload"
                    ) as HTMLInputElement;
                    if (fileInput) fileInput.value = "";
                  }}
                  className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors mr-4"
                >
                  새로운 검사
                </button>
                <Link
                  href="/logpage"
                  className="bg-gray-600 text-white px-6 py-3 rounded-lg hover:bg-gray-700 transition-colors inline-block"
                >
                  검사 로그 보기
                </Link>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

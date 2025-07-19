"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

interface InspectionLog {
  id: number;
  timestamp: string;
  original_filename: string;
  processed_filename: string;
  file_size: number;
  is_defective: boolean;
  confidence: number;
  defect_count: number;
  details: string;
  result_image: string | null;
  processing_time: number;
  model_used: string;
}

interface LogStats {
  total_inspections: number;
  defective_count: number;
  normal_count: number;
  defective_rate: number;
  avg_confidence: number;
  avg_processing_time: number;
  recent_inspections: number;
}

export default function LogPage() {
  const [logs, setLogs] = useState<InspectionLog[]>([]);
  const [stats, setStats] = useState<LogStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedLog, setSelectedLog] = useState<InspectionLog | null>(null);
  const [filterType, setFilterType] = useState<"all" | "defective" | "normal">(
    "all"
  );
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetchLogs();
    fetchStats();
  }, []);

  const fetchLogs = async () => {
    try {
      setLoading(true);
      const response = await fetch("http://localhost:5000/logs?limit=200");

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        setLogs(data.logs);
        setError(null);
      } else {
        throw new Error(data.error || "로그를 가져오는데 실패했습니다.");
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.";
      setError(errorMessage);
      console.error("로그 조회 오류:", err);
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch("http://localhost:5000/logs/stats");

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setStats(data.stats);
        }
      }
    } catch (err) {
      console.error("통계 조회 오류:", err);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString("ko-KR", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const filteredLogs = logs.filter((log) => {
    const matchesFilter =
      filterType === "all" ||
      (filterType === "defective" && log.is_defective) ||
      (filterType === "normal" && !log.is_defective);

    const matchesSearch =
      searchTerm === "" ||
      log.original_filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.details.toLowerCase().includes(searchTerm.toLowerCase());

    return matchesFilter && matchesSearch;
  });

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
            <h1 className="text-xl font-bold">검사 로그</h1>
            <div className="flex space-x-2">
              <Link
                href="/inspection"
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors"
              >
                새 검사하기
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 통계 카드 */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-lg p-6 text-black">
              <h3 className="text-lg font-semibold text-gray-700 mb-2">
                전체 검사
              </h3>
              <div className="text-3xl font-bold text-blue-600">
                {stats.total_inspections}
              </div>
              <p className="text-sm text-gray-500 mt-1">총 검사 횟수</p>
            </div>

            <div className="bg-white rounded-lg p-6 text-black">
              <h3 className="text-lg font-semibold text-gray-700 mb-2">
                불량 검출
              </h3>
              <div className="text-3xl font-bold text-red-600">
                {stats.defective_count}
              </div>
              <p className="text-sm text-gray-500 mt-1">
                불량률: {stats.defective_rate.toFixed(1)}%
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 text-black">
              <h3 className="text-lg font-semibold text-gray-700 mb-2">
                평균 신뢰도
              </h3>
              <div className="text-3xl font-bold text-green-600">
                {stats.avg_confidence}%
              </div>
              <p className="text-sm text-gray-500 mt-1">전체 평균</p>
            </div>

            <div className="bg-white rounded-lg p-6 text-black">
              <h3 className="text-lg font-semibold text-gray-700 mb-2">
                평균 처리시간
              </h3>
              <div className="text-3xl font-bold text-purple-600">
                {stats.avg_processing_time}s
              </div>
              <p className="text-sm text-gray-500 mt-1">
                최근 7일: {stats.recent_inspections}건
              </p>
            </div>
          </div>
        )}

        {/* 필터 및 검색 */}
        <div className="bg-white rounded-lg p-6 mb-6 text-black">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
            <div className="flex space-x-4">
              <select
                value={filterType}
                onChange={(e) =>
                  setFilterType(
                    e.target.value as "all" | "defective" | "normal"
                  )
                }
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">전체</option>
                <option value="defective">불량만</option>
                <option value="normal">정상만</option>
              </select>

              <button
                onClick={fetchLogs}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                새로고침
              </button>
            </div>

            <div className="flex-1 max-w-md">
              <input
                type="text"
                placeholder="파일명 또는 상세 내용으로 검색..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
        </div>

        {/* 로딩 상태 */}
        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
            <p className="mt-4">로그를 불러오는 중...</p>
          </div>
        )}

        {/* 에러 상태 */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
            <div className="flex items-center">
              <span className="text-red-600 text-xl mr-3">⚠️</span>
              <div>
                <h3 className="text-lg font-semibold text-red-800">
                  오류 발생
                </h3>
                <p className="text-red-700">{error}</p>
                <p className="text-sm text-red-600 mt-2">
                  Flask 서버가 실행 중인지 확인해주세요. (http://localhost:5000)
                </p>
              </div>
            </div>
          </div>
        )}

        {/* 로그 테이블 */}
        {!loading && !error && (
          <div className="bg-white rounded-lg overflow-hidden text-black">
            <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
              <h2 className="text-xl font-semibold">
                검사 로그 ({filteredLogs.length}건)
              </h2>
            </div>

            {filteredLogs.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-gray-500">검사 로그가 없습니다.</p>
                <Link
                  href="/inspection"
                  className="mt-4 inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  첫 번째 검사 시작하기
                </Link>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        시간
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        파일명
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        결과
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        신뢰도
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        처리시간
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        액션
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {filteredLogs.map((log) => (
                      <tr key={log.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatTimestamp(log.timestamp)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900">
                            {log.original_filename}
                          </div>
                          <div className="text-sm text-gray-500">
                            {formatFileSize(log.file_size)}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span
                            className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                              log.is_defective
                                ? "bg-red-100 text-red-800"
                                : "bg-green-100 text-green-800"
                            }`}
                          >
                            {log.is_defective ? "❌ 불량" : "✅ 정상"}
                          </span>
                          {log.is_defective && log.defect_count > 0 && (
                            <div className="text-xs text-red-600 mt-1">
                              {log.defect_count}개 검출
                            </div>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {(log.confidence * 100).toFixed(1)}%
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {log.processing_time.toFixed(2)}s
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button
                            onClick={() => setSelectedLog(log)}
                            className="text-blue-600 hover:text-blue-900 mr-3"
                          >
                            상세보기
                          </button>
                          {log.result_image && (
                            <a
                              href={`http://localhost:5000/results/${log.result_image}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-green-600 hover:text-green-900"
                            >
                              결과이미지
                            </a>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>

      {/* 상세보기 모달 */}
      {selectedLog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl max-h-[90vh] overflow-y-auto text-black">
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-2xl font-bold">검사 상세 정보</h3>
                <button
                  onClick={() => setSelectedLog(null)}
                  className="text-gray-500 hover:text-gray-700 text-2xl"
                >
                  ✕
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-lg mb-4">기본 정보</h4>
                  <div className="space-y-3">
                    <div>
                      <span className="font-medium">검사 시간:</span>
                      <p className="text-gray-700">
                        {formatTimestamp(selectedLog.timestamp)}
                      </p>
                    </div>
                    <div>
                      <span className="font-medium">파일명:</span>
                      <p className="text-gray-700">
                        {selectedLog.original_filename}
                      </p>
                    </div>
                    <div>
                      <span className="font-medium">파일 크기:</span>
                      <p className="text-gray-700">
                        {formatFileSize(selectedLog.file_size)}
                      </p>
                    </div>
                    <div>
                      <span className="font-medium">모델:</span>
                      <p className="text-gray-700">{selectedLog.model_used}</p>
                    </div>
                    <div>
                      <span className="font-medium">처리 시간:</span>
                      <p className="text-gray-700">
                        {selectedLog.processing_time.toFixed(2)}초
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold text-lg mb-4">검사 결과</h4>
                  <div className="space-y-3">
                    <div>
                      <span className="font-medium">결과:</span>
                      <p
                        className={`text-lg font-bold ${
                          selectedLog.is_defective
                            ? "text-red-600"
                            : "text-green-600"
                        }`}
                      >
                        {selectedLog.is_defective ? "❌ 불량 검출" : "✅ 정상"}
                      </p>
                    </div>
                    <div>
                      <span className="font-medium">신뢰도:</span>
                      <p className="text-gray-700">
                        {(selectedLog.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                    {selectedLog.is_defective && (
                      <div>
                        <span className="font-medium">불량 개수:</span>
                        <p className="text-red-600 font-bold">
                          {selectedLog.defect_count}개
                        </p>
                      </div>
                    )}
                    <div>
                      <span className="font-medium">상세 분석:</span>
                      <p className="text-gray-700 mt-1">
                        {selectedLog.details}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* 결과 이미지 */}
              {selectedLog.result_image && (
                <div className="mt-6">
                  <h4 className="font-semibold text-lg mb-4">결과 이미지</h4>
                  <div className="text-center">
                    <img
                      src={`http://localhost:5000/results/${selectedLog.result_image}`}
                      alt="검사 결과"
                      className="max-w-full h-auto rounded-lg border border-gray-300"
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                      }}
                    />
                    <p className="text-sm text-gray-500 mt-2">
                      불량 영역이 빨간색 박스로 표시됩니다
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";

// Google Maps types
declare global {
  interface Window {
    google: any;
  }
}

export default function AboutPage() {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);

  useEffect(() => {
    const initializeMap = () => {
      if (!mapRef.current) return;

      // Company location (Gangnam Station area)
      const companyLocation = { lat: 37.4981, lng: 127.0276 };

      try {
        // Create map
        const map = new window.google.maps.Map(mapRef.current, {
          zoom: 17,
          center: companyLocation,
          mapTypeControl: false,
          streetViewControl: false,
          fullscreenControl: true,
          styles: [
            {
              featureType: "all",
              elementType: "geometry.fill",
              stylers: [{ color: "#f5f5f5" }],
            },
            {
              featureType: "water",
              elementType: "geometry",
              stylers: [{ color: "#e9e9e9" }, { lightness: 17 }],
            },
          ],
        });

        // Add marker
        const marker = new window.google.maps.Marker({
          position: companyLocation,
          map: map,
          title: "AI 검사 시스템 본사",
          animation: window.google.maps.Animation.DROP,
        });

        // Add info window
        const infoWindow = new window.google.maps.InfoWindow({
          content: `
            <div style="padding: 10px; font-family: Arial, sans-serif;">
              <h3 style="margin: 0 0 10px 0; color: #333;">AI 검사 시스템</h3>
              <p style="margin: 0; color: #666;">서울특별시 강남구 테헤란로 123</p>
              <p style="margin: 5px 0 0 0; color: #666;">7층</p>
            </div>
          `,
        });

        // Show info window on marker click
        marker.addListener("click", () => {
          infoWindow.open(map, marker);
        });

        // Auto-open info window
        setTimeout(() => {
          infoWindow.open(map, marker);
        }, 1000);

        mapInstanceRef.current = map;
      } catch (error) {
        console.error("Error initializing map:", error);
      }
    };

    const loadGoogleMaps = () => {
      // Check if Google Maps is already loaded
      if (window.google && window.google.maps) {
        initializeMap();
        return;
      }

      // Create script element
      const script = document.createElement("script");
      script.src = `https://maps.googleapis.com/maps/api/js?key=AIzaSyBUqQUwZlyZ3_zt0bxCqvy2jymt_r4hi4Y&libraries=places`;
      script.async = true;
      script.defer = true;

      script.onload = () => {
        initializeMap();
      };

      script.onerror = (error) => {
        console.error("Failed to load Google Maps:", error);
        // Show fallback content
        if (mapRef.current) {
          mapRef.current.innerHTML = `
            <div class="flex items-center justify-center h-full">
              <div class="text-center">
                <div class="text-4xl mb-2">🗺️</div>
                <p class="text-gray-600">지도를 불러올 수 없습니다</p>
                <p class="text-sm text-gray-500 mt-2">네트워크 연결을 확인해주세요</p>
              </div>
            </div>
          `;
        }
      };

      document.head.appendChild(script);
    };

    loadGoogleMaps();

    // Cleanup function
    return () => {
      mapInstanceRef.current = null;
    };
  }, []);
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
            <h1 className="text-xl font-bold">제작자 정보</h1>
            <div></div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <div className="w-24 h-24 bg-white rounded-full flex items-center justify-center mx-auto mb-6">
            <span className="text-black text-3xl font-bold">👨‍💻</span>
          </div>
          <h2 className="text-4xl font-bold mb-4">AI 검사 시스템</h2>
          <p className="text-xl text-gray-300">
            인공지능 기반 품질 검사 솔루션
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
          {/* Developer Info */}
          <div className="bg-white rounded-xl p-8 text-black">
            <h3 className="text-2xl font-bold mb-6">개발자 정보</h3>
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-gray-700">이름</h4>
                <p className="text-lg">AI 시스템 개발팀</p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-700">이메일</h4>
                <p className="text-lg">developer@ai-inspection.com</p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-700">개발 기간</h4>
                <p className="text-lg">2024.12 ~ 2025.01</p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-700">버전</h4>
                <p className="text-lg">v1.0.0</p>
              </div>
            </div>
          </div>

          {/* Technology Stack */}
          <div className="bg-white rounded-xl p-8 text-black">
            <h3 className="text-2xl font-bold mb-6">기술 스택</h3>
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-gray-700">Frontend</h4>
                <div className="flex flex-wrap gap-2 mt-2">
                  <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                    Next.js 14
                  </span>
                  <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                    TypeScript
                  </span>
                  <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                    Tailwind CSS
                  </span>
                </div>
              </div>
              <div>
                <h4 className="font-semibold text-gray-700">Backend</h4>
                <div className="flex flex-wrap gap-2 mt-2">
                  <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">
                    Python
                  </span>
                  <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">
                    FastAPI
                  </span>
                  <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">
                    TensorFlow
                  </span>
                </div>
              </div>
              <div>
                <h4 className="font-semibold text-gray-700">Database</h4>
                <div className="flex flex-wrap gap-2 mt-2">
                  <span className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm">
                    MongoDB
                  </span>
                  <span className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm">
                    Redis
                  </span>
                </div>
              </div>
              <div>
                <h4 className="font-semibold text-gray-700">AI/ML</h4>
                <div className="flex flex-wrap gap-2 mt-2">
                  <span className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm">
                    OpenCV
                  </span>
                  <span className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm">
                    PyTorch
                  </span>
                  <span className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm">
                    scikit-learn
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Features */}
        <div className="bg-white rounded-xl p-8 text-black mb-8">
          <h3 className="text-2xl font-bold mb-6">주요 기능</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <span className="text-green-500 text-xl">✅</span>
                <div>
                  <h4 className="font-semibold">실시간 AI 검사</h4>
                  <p className="text-gray-600 text-sm">
                    고성능 딥러닝 모델을 활용한 실시간 품질 검사
                  </p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <span className="text-green-500 text-xl">✅</span>
                <div>
                  <h4 className="font-semibold">높은 정확도</h4>
                  <p className="text-gray-600 text-sm">
                    99% 이상의 검사 정확도 보장
                  </p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <span className="text-green-500 text-xl">✅</span>
                <div>
                  <h4 className="font-semibold">사용자 친화적 UI</h4>
                  <p className="text-gray-600 text-sm">
                    직관적이고 모던한 사용자 인터페이스
                  </p>
                </div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <span className="text-green-500 text-xl">✅</span>
                <div>
                  <h4 className="font-semibold">상세한 로그 관리</h4>
                  <p className="text-gray-600 text-sm">
                    모든 검사 결과의 체계적 기록 및 분석
                  </p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <span className="text-green-500 text-xl">✅</span>
                <div>
                  <h4 className="font-semibold">확장 가능한 구조</h4>
                  <p className="text-gray-600 text-sm">
                    마이크로서비스 아키텍처 기반 확장성
                  </p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <span className="text-green-500 text-xl">✅</span>
                <div>
                  <h4 className="font-semibold">보안 강화</h4>
                  <p className="text-gray-600 text-sm">
                    Firebase 인증 및 데이터 암호화
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Location Section */}
        <div
          className="bg-white rounded-xl p-8 text-black mb-8"
          style={{ minHeight: "600px" }}
        >
          <h3 className="text-2xl font-bold mb-6">찾아오시는 곳</h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-full">
            {/* Address Information */}
            <div className="space-y-6">
              <div>
                <h4 className="font-semibold text-gray-700 mb-3">주소</h4>
                <p className="text-lg mb-2">서울특별시 강남구 테헤란로 123</p>
                <p className="text-gray-600">AI 검사 시스템 본사 7층</p>
              </div>

              <div>
                <h4 className="font-semibold text-gray-700 mb-3">교통편</h4>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 bg-blue-600 rounded-full"></span>
                    <span>지하철 2호선 강남역 3번 출구 도보 5분</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 bg-green-600 rounded-full"></span>
                    <span>지하철 신분당선 강남역 4번 출구 도보 3분</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 bg-orange-600 rounded-full"></span>
                    <span>버스 146, 360, 740번 강남역 정류장</span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-semibold text-gray-700 mb-3">운영시간</h4>
                <div className="space-y-1">
                  <p>평일: 09:00 ~ 18:00</p>
                  <p>토요일: 09:00 ~ 13:00</p>
                  <p className="text-red-600">일요일 및 공휴일: 휴무</p>
                </div>
              </div>

              <div>
                <h4 className="font-semibold text-gray-700 mb-3">연락처</h4>
                <div className="space-y-1">
                  <p>전화: 02-1234-5678</p>
                  <p>팩스: 02-1234-5679</p>
                  <p>이메일: info@ai-inspection.com</p>
                </div>
              </div>
            </div>

            {/* Map Container */}
            <div className="relative">
              <div
                ref={mapRef}
                className="w-full h-96 rounded-lg border border-gray-200"
                style={{ minHeight: "400px" }}
              >
                {/* Loading placeholder */}
                <div className="flex items-center justify-center h-full bg-gray-100 rounded-lg">
                  <div className="text-center">
                    <div className="text-4xl mb-2">🗺️</div>
                    <p className="text-gray-600">지도를 불러오는 중...</p>
                  </div>
                </div>
              </div>

              {/* External Map Links */}
              <div className="mt-4 flex flex-col space-y-2">
                <h5 className="font-semibold text-gray-700 mb-2">
                  다른 지도에서 보기
                </h5>
                <div className="grid grid-cols-1 gap-2">
                  <button
                    onClick={() =>
                      window.open(
                        "https://map.naver.com/v5/search/강남역",
                        "_blank"
                      )
                    }
                    className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors text-sm"
                  >
                    📍 네이버 지도에서 보기
                  </button>
                  <button
                    onClick={() =>
                      window.open(
                        "https://map.kakao.com/link/search/강남역",
                        "_blank"
                      )
                    }
                    className="bg-yellow-500 text-white px-4 py-2 rounded-lg hover:bg-yellow-600 transition-colors text-sm"
                  >
                    🗺️ 카카오맵에서 보기
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Contact */}
        <div className="bg-gray-900 rounded-xl p-8 border border-gray-800 text-center">
          <h3 className="text-2xl font-bold mb-4">문의사항</h3>
          <p className="text-gray-300 mb-6">
            시스템에 대한 문의사항이나 개선 제안이 있으시면 언제든 연락주세요.
          </p>
          <div className="flex justify-center space-x-4">
            <a
              href="mailto:developer@ai-inspection.com"
              className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
            >
              이메일 보내기
            </a>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-gray-600 text-white px-6 py-3 rounded-lg hover:bg-gray-700 transition-colors"
            >
              GitHub 방문
            </a>
          </div>
        </div>
      </main>
    </div>
  );
}

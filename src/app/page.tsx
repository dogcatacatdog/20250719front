'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { 
  signInWithEmailAndPassword, 
  signInWithPopup, 
  signOut, 
  onAuthStateChanged,
  User 
} from 'firebase/auth';
import { auth, googleProvider } from '@/lib/firebase';

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isSigningIn, setIsSigningIn] = useState(false);

  // Monitor authentication state
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  // Email/Password login
  const handleEmailLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('이메일과 비밀번호를 입력해주세요.');
      return;
    }

    setIsSigningIn(true);
    setError('');

    try {
      await signInWithEmailAndPassword(auth, email, password);
    } catch (error: any) {
      setError('로그인에 실패했습니다. 이메일과 비밀번호를 확인해주세요.');
      console.error('Login error:', error);
    } finally {
      setIsSigningIn(false);
    }
  };

  // Google login
  const handleGoogleLogin = async () => {
    setIsSigningIn(true);
    setError('');

    try {
      await signInWithPopup(auth, googleProvider);
    } catch (error: any) {
      setError('Google 로그인에 실패했습니다.');
      console.error('Google login error:', error);
    } finally {
      setIsSigningIn(false);
    }
  };

  // Logout
  const handleLogout = async () => {
    try {
      await signOut(auth);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  // Loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-white text-xl">로딩 중...</div>
      </div>
    );
  }

  // Login Component
  if (!user) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center p-4">
        <div className="bg-white rounded-xl shadow-2xl p-8 w-full max-w-md">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-black rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white text-2xl font-bold">AI</span>
            </div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              AI 검사 시스템
            </h1>
            <p className="text-gray-600">인공지능 품질 검사 플랫폼</p>
          </div>

          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
              {error}
            </div>
          )}

          <form onSubmit={handleEmailLogin} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                이메일
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                placeholder="이메일을 입력하세요"
                disabled={isSigningIn}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                비밀번호
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                placeholder="비밀번호를 입력하세요"
                disabled={isSigningIn}
              />
            </div>

            <button
              type="submit"
              disabled={isSigningIn}
              className={`w-full py-3 rounded-lg font-medium transition-colors ${
                isSigningIn 
                  ? 'bg-gray-400 text-gray-600 cursor-not-allowed' 
                  : 'bg-black text-white hover:bg-gray-800'
              }`}
            >
              {isSigningIn ? '로그인 중...' : '로그인'}
            </button>
          </form>

          <div className="relative mt-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white text-gray-500">또는</span>
            </div>
          </div>

          <button 
            onClick={handleGoogleLogin}
            disabled={isSigningIn}
            className={`w-full mt-6 border border-gray-300 text-gray-700 py-3 rounded-lg transition-colors font-medium flex items-center justify-center space-x-2 ${
              isSigningIn 
                ? 'cursor-not-allowed opacity-50' 
                : 'hover:bg-gray-50'
            }`}
          >
            <span className="text-red-500 font-bold">G</span>
            <span>{isSigningIn ? 'Google 로그인 중...' : 'Google로 로그인'}</span>
          </button>
        </div>
      </div>
    );
  }

  // Main Dashboard Component
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                <span className="text-black text-sm font-bold">AI</span>
              </div>
              <h1 className="text-xl font-bold">AI 검사 시스템</h1>
            </div>

            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-300">
                환영합니다, {user?.displayName || user?.email}
              </span>
              <button
                onClick={handleLogout}
                className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg text-sm transition-colors"
              >
                로그아웃
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center">
          <h2 className="text-4xl font-bold mb-4">
            인공지능 검사 시스템에 오신 것을 환영합니다
          </h2>
          <p className="text-xl text-gray-300 mb-8">
            최첨단 AI 기술로 품질 검사를 더 정확하고 빠르게
          </p>
        </div>

        {/* Navigation Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12">
          {/* Inspection Card */}
          <Link href="/inspection">
            <div className="bg-white rounded-xl p-8 text-black hover:shadow-2xl transition-all transform hover:scale-105 cursor-pointer">
              <div className="w-16 h-16 bg-blue-600 rounded-lg flex items-center justify-center mb-6">
                <span className="text-white text-2xl">🔍</span>
              </div>
              <h3 className="text-2xl font-bold mb-4">검사하기</h3>
              <p className="text-gray-600 mb-6">
                AI 알고리즘을 사용하여 제품의 품질을 자동으로 검사합니다.
              </p>
              <div className="flex items-center text-blue-600 font-medium">
                <span>검사 시작하기</span>
                <span className="ml-2">→</span>
              </div>
            </div>
          </Link>

          {/* Log Page Card */}
          <Link href="/logpage">
            <div className="bg-white rounded-xl p-8 text-black hover:shadow-2xl transition-all transform hover:scale-105 cursor-pointer">
              <div className="w-16 h-16 bg-green-600 rounded-lg flex items-center justify-center mb-6">
                <span className="text-white text-2xl">📊</span>
              </div>
              <h3 className="text-2xl font-bold mb-4">검사 로그</h3>
              <p className="text-gray-600 mb-6">
                이전 검사 결과와 통계를 확인하고 분석할 수 있습니다.
              </p>
              <div className="flex items-center text-green-600 font-medium">
                <span>로그 보기</span>
                <span className="ml-2">→</span>
              </div>
            </div>
          </Link>

          {/* About Card */}
          <Link href="/about">
            <div className="bg-white rounded-xl p-8 text-black hover:shadow-2xl transition-all transform hover:scale-105 cursor-pointer">
              <div className="w-16 h-16 bg-purple-600 rounded-lg flex items-center justify-center mb-6">
                <span className="text-white text-2xl">👨‍💻</span>
              </div>
              <h3 className="text-2xl font-bold mb-4">제작자 정보</h3>
              <p className="text-gray-600 mb-6">
                시스템 개발자 정보와 기술 스택을 확인할 수 있습니다.
              </p>
              <div className="flex items-center text-purple-600 font-medium">
                <span>정보 보기</span>
                <span className="ml-2">→</span>
              </div>
            </div>
          </Link>
        </div>

        {/* Stats Section */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-16">
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-400 mb-2">99.2%</div>
              <div className="text-sm text-gray-400">정확도</div>
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
            <div className="text-center">
              <div className="text-3xl font-bold text-green-400 mb-2">
                1,247
              </div>
              <div className="text-sm text-gray-400">총 검사 횟수</div>
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
            <div className="text-center">
              <div className="text-3xl font-bold text-yellow-400 mb-2">
                0.8s
              </div>
              <div className="text-sm text-gray-400">평균 검사 시간</div>
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-400 mb-2">
                24/7
              </div>
              <div className="text-sm text-gray-400">서비스 운영</div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 border-t border-gray-800 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-400">
            <p>&copy; 2025 AI 검사 시스템. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

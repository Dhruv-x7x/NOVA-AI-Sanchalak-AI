import { useState, useEffect } from 'react';

interface StartupStatus {
  ready: boolean;
  stage: string;
  percent: number;
  details: string;
  elapsed_seconds: number;
}

interface StartupLoaderProps {
  children: React.ReactNode;
}

export default function StartupLoader({ children }: StartupLoaderProps) {
  const [status, setStatus] = useState<StartupStatus | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [fadeOut, setFadeOut] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    let attempts = 0;
    const maxAttempts = 60; // 30 seconds max wait

    const checkStatus = async () => {
      try {
        const res = await fetch('/api/startup-status');
        if (res.ok) {
          const data: StartupStatus = await res.json();
          setStatus(data);
          setError(null);

          if (data.ready) {
            // Start fade out animation
            setFadeOut(true);
            setTimeout(() => setIsReady(true), 500);
            clearInterval(interval);
          }
        } else {
          attempts++;
          if (attempts > maxAttempts) {
            setError('Backend not responding');
            clearInterval(interval);
          }
        }
      } catch (err) {
        attempts++;
        // Backend might not be up yet, keep trying
        if (attempts > maxAttempts) {
          setError('Cannot connect to backend');
          clearInterval(interval);
        }
      }
    };

    // Start polling
    checkStatus();
    interval = setInterval(checkStatus, 500);

    return () => clearInterval(interval);
  }, []);

  // If already ready, just render children
  if (isReady) {
    return <>{children}</>;
  }

  const percent = status?.percent ?? 0;
  const details = status?.details ?? 'Connecting to server...';

  return (
    <div
      className={`fixed inset-0 z-[9999] flex flex-col items-center justify-center bg-[#0a0a0c] transition-opacity duration-500 ${fadeOut ? 'opacity-0' : 'opacity-100'}`}
    >
      {/* Background Video */}
      <video
        autoPlay
        loop
        muted
        playsInline
        className="absolute inset-0 w-full h-full object-cover opacity-20 z-0"
      >
        <source src="/clinical_trail.mp4" type="video/mp4" />
      </video>

      {/* Logo/Title */}
      <div className="relative z-10 mb-8 text-center">
        <div className="relative w-20 h-20 mx-auto mb-4">
          {/* CSS Spinner Fallback */}
          <div className="absolute inset-0 rounded-full border-2 border-indigo-500/20 border-t-indigo-500 animate-spin" />

          <video
            autoPlay
            loop
            muted
            playsInline
            className="relative z-10 w-20 h-20 object-cover rounded-full"
          >
            <source src="/clinical_trail.mp4" type="video/mp4" />
          </video>
        </div>
        <h1 className="relative z-10 text-3xl font-bold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
          a6on-i
        </h1>
        <p className="relative z-10 text-xs text-gray-500 mt-1 font-mono uppercase tracking-widest">
          Clinical Intelligence Platform
        </p>
      </div>

      {/* Progress Container */}
      <div className="relative z-10 w-80 space-y-4">
        {/* Progress Bar */}
        <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${percent}%` }}
          />
          {/* Shimmer effect */}
          <div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer"
            style={{
              width: `${percent}%`,
              animation: 'shimmer 1.5s infinite'
            }}
          />
        </div>

        {/* Status Text */}
        <div className="flex justify-between items-center text-xs">
          <span className="text-gray-400 font-mono truncate max-w-[200px]">
            {details}
          </span>
          <span className="text-indigo-400 font-mono font-bold">
            {percent}%
          </span>
        </div>

        {/* Error Message */}
        {error && (
          <div className="text-center text-red-400 text-xs mt-4 animate-pulse">
            {error}. Please check backend logs.
          </div>
        )}
      </div>

      {/* Animated dots */}
      <div className="relative z-10 mt-12 flex gap-2">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="w-2 h-2 rounded-full bg-indigo-500/50"
            style={{
              animation: `bounce 1s infinite ${i * 0.15}s`
            }}
          />
        ))}
      </div>

      {/* CSS Keyframes */}
      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        @keyframes bounce {
          0%, 100% { transform: translateY(0); opacity: 0.5; }
          50% { transform: translateY(-8px); opacity: 1; }
        }
      `}</style>
    </div>
  );
}

/**
 * a6on_iLoader — Branded loading animation component
 * Uses the loading.gif for a polished, lightweight loading experience.
 * Drop-in replacement for spinners anywhere in the app.
 */

import { cn } from '@/lib/utils';

interface A6onLoaderProps {
  /** Size of the animation: 'sm' (32px), 'md' (48px), 'lg' (64px), 'xl' (96px) */
  size?: 'sm' | 'md' | 'lg' | 'xl';
  /** Optional label text below the animation */
  label?: string;
  /** Whether to show the label with a pulse animation */
  pulse?: boolean;
  /** Full-page centered mode with min-height */
  fullPage?: boolean;
  /** Custom className for the outer container */
  className?: string;
}

const sizeMap = {
  sm: 'w-8 h-8',
  md: 'w-12 h-12',
  lg: 'w-16 h-16',
  xl: 'w-24 h-24',
};

export default function A6onLoader({
  size = 'md',
  label,
  pulse = true,
  fullPage = false,
  className = '',
}: A6onLoaderProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center gap-3 ${fullPage ? 'min-h-[60vh]' : ''
        } ${className}`}
    >
      <div className="relative">
        {/* CSS Spinner Fallback (shows while GIF loads) */}
        <div
          className={cn(
            "absolute inset-0 rounded-full border-2 border-primary/20 border-t-primary animate-spin",
            sizeMap[size]
          )}
        />

        {/* The Branded GIF */}
        <img
          src="/loading.gif?v=3"
          alt="Loading"
          className={cn(sizeMap[size], "relative z-10 object-contain")}
          // Optional: hide spinner when loaded
          onLoad={(e) => {
            const spinner = e.currentTarget.previousSibling as HTMLElement;
            if (spinner) (spinner as HTMLElement).style.display = 'none';
          }}
        />
      </div>

      {label && (
        <p
          className={cn(
            "text-indigo-400 font-mono text-xs uppercase tracking-widest",
            pulse && "animate-pulse"
          )}
        >
          {label}
        </p>
      )}
    </div>
  );
}

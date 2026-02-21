/**
 * SanchalakLoader — Branded loading animation component
 * Uses the loading.gif for a polished, lightweight loading experience.
 * Drop-in replacement for spinners anywhere in the app.
 */

interface SanchalakLoaderProps {
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

export default function SanchalakLoader({
  size = 'md',
  label,
  pulse = true,
  fullPage = false,
  className = '',
}: SanchalakLoaderProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center gap-3 ${
        fullPage ? 'min-h-[60vh]' : ''
      } ${className}`}
    >
      <img
        src="/loading.gif"
        alt="Loading"
        className={`${sizeMap[size]} object-contain`}
      />
      {label && (
        <p
          className={`text-indigo-400 font-mono text-xs uppercase tracking-widest ${
            pulse ? 'animate-pulse' : ''
          }`}
        >
          {label}
        </p>
      )}
    </div>
  );
}

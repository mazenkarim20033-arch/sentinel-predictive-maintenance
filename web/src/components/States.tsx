import type { ReactNode } from "react";

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="state" role="status" aria-live="polite">
      <div className="skeleton" aria-hidden="true" />
      <span className="sr-only">{label}</span>
      <p style={{ marginTop: "0.6rem" }}>{label}…</p>
    </div>
  );
}

export function ErrorBox({ title, error, onRetry }: { title: string; error: Error; onRetry?: () => void }) {
  return (
    <div className="state error" role="alert">
      <strong>{title}</strong>
      <p>{error.message}</p>
      {onRetry && <button onClick={onRetry}>Try again</button>}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="state">
      <strong>{title}</strong>
      {children && <p>{children}</p>}
    </div>
  );
}

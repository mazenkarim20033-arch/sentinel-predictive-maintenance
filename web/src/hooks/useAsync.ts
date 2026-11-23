import { useCallback, useEffect, useState } from "react";

export type AsyncState<T> =
  | { status: "loading"; data: undefined; error: undefined }
  | { status: "ready"; data: T; error: undefined }
  | { status: "error"; data: undefined; error: Error };

export function useAsync<T>(load: (signal: AbortSignal) => Promise<T>, deps: unknown[]): AsyncState<T> & { reload: () => void } {
  const [state, setState] = useState<AsyncState<T>>({ status: "loading", data: undefined, error: undefined });
  const [tick, setTick] = useState(0);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    setState({ status: "loading", data: undefined, error: undefined });
    load(controller.signal).then(
      (data) => !controller.signal.aborted && setState({ status: "ready", data, error: undefined }),
      (error: Error) => !controller.signal.aborted && error.name !== "AbortError" && setState({ status: "error", data: undefined, error }),
    );
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return { ...state, reload } as AsyncState<T> & { reload: () => void };
}

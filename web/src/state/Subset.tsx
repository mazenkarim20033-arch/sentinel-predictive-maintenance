import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { SubsetInfo } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { ErrorBox, Loading } from "../components/States";

interface Ctx {
  subset: string;
  setSubset: (s: string) => void;
  subsets: SubsetInfo[];
  info: SubsetInfo | undefined;
}

const SubsetContext = createContext<Ctx | null>(null);
const KEY = "sentinel.subset";

function remembered(): string | null {
  try {
    return window.localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function SubsetProvider({ children }: { children: ReactNode }) {
  const state = useAsync((signal) => api.subsets(signal), []);
  const [chosen, setChosen] = useState<string | null>(remembered);
  const usable = useMemo(() => (state.status === "ready" ? state.data.filter((s) => s.model_available && s.data_available) : []), [state]);
  const subset = chosen && usable.some((s) => s.subset === chosen) ? chosen : usable[0]?.subset ?? "";

  useEffect(() => {
    try {
      if (subset) window.localStorage.setItem(KEY, subset);
    } catch {
      /* storage can be unavailable; the choice just will not persist */
    }
  }, [subset]);

  if (state.status === "loading") return <main className="main"><Loading label="Connecting to the SENTINEL API" /></main>;
  if (state.status === "error") return <main className="main"><ErrorBox title="The API is not reachable" error={state.error} onRetry={state.reload} /></main>;
  if (usable.length === 0)
    return (
      <main className="main">
        <ErrorBox
          title="No dataset is ready to serve"
          error={new Error("Each dataset needs its model bundle and data files. Build the bundles with: python scripts/build_bundle.py (data files go in data/raw/extracted).")}
          onRetry={state.reload}
        />
      </main>
    );
  return <SubsetContext.Provider value={{ subset, setSubset: setChosen, subsets: usable, info: usable.find((s) => s.subset === subset) }}>{children}</SubsetContext.Provider>;
}

export function useSubset(): Ctx {
  const ctx = useContext(SubsetContext);
  if (!ctx) throw new Error("useSubset must be used inside SubsetProvider");
  return ctx;
}

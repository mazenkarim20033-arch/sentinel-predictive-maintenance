import type { ReactNode } from "react";
import type { Risk } from "../api/types";

export type ProvenanceKind = "prediction" | "ground_truth" | "dataset" | "measured_evaluation" | "illustrative";

const LABELS: Record<ProvenanceKind, string> = {
  prediction: "Model prediction",
  ground_truth: "Ground truth, evaluation only",
  dataset: "Dataset signal",
  measured_evaluation: "Measured benchmark",
  illustrative: "Illustrative",
};

const MEANING: Record<ProvenanceKind, string> = {
  prediction: "Produced by the model from the history available at that time.",
  ground_truth: "Known from the dataset. Shown for comparison; never an input to the model.",
  dataset: "Simulated measurement from the C-MAPSS data set.",
  measured_evaluation: "Result recorded from an experiment on held-out data.",
  illustrative: "Not real model output or data.",
};

export function Provenance({ kind }: { kind: ProvenanceKind }) {
  return (
    <span className={`chip ${kind}`} title={MEANING[kind]}>
      {LABELS[kind]}
    </span>
  );
}

const RISK_LABEL: Record<Risk, string> = { normal: "Normal", warning: "Warning", critical: "Critical" };

export function RiskBadge({ risk }: { risk: Risk }) {
  return <span className={`risk ${risk}`}>{RISK_LABEL[risk]}</span>;
}

export function Margin({ children }: { children: ReactNode }) {
  return <aside className="margin-notes" aria-label="Reading notes">{children}</aside>;
}

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, HashRouter } from "react-router-dom";
import { App } from "./App";
import { IS_SNAPSHOT } from "./api/client";
import "./styles.css";

// Static hosting cannot rewrite deep links, so the public demo routes with the URL hash.
const Router = IS_SNAPSHOT ? HashRouter : BrowserRouter;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Router>
      <App />
    </Router>
  </StrictMode>,
);

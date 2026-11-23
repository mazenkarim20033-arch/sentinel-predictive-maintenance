import { NavLink, Route, Routes } from "react-router-dom";
import { SubsetProvider, useSubset } from "./state/Subset";
import { Dashboard } from "./pages/Dashboard";
import { EngineDetail } from "./pages/EngineDetail";
import { Fleet } from "./pages/Fleet";
import { ModelLab } from "./pages/ModelLab";
import { Research } from "./pages/Research";
import { SimulationLab } from "./pages/SimulationLab";

function Shell() {
  const { subset, setSubset, subsets } = useSubset();
  return (
    <>
      <div className="banner">Simulated NASA C-MAPSS data. A research demonstration, not for maintenance decisions.</div>
      <header className="topbar">
        <NavLink to="/" className="brand">SENTINEL<small>Remaining-life monitoring</small></NavLink>
        <nav className="nav" aria-label="Main">
          <NavLink to="/" end>Dashboard</NavLink>
          <NavLink to="/fleet">Fleet</NavLink>
          <NavLink to="/lab">Simulation lab</NavLink>
          <NavLink to="/models">Model lab</NavLink>
          <NavLink to="/research">Research</NavLink>
        </nav>
        <label className="subset-picker">Dataset
          <select value={subset} onChange={(e) => setSubset(e.target.value)} aria-label="Dataset">
            {subsets.map((s) => <option key={s.subset} value={s.subset}>{s.subset}</option>)}
          </select>
        </label>
      </header>
      <main className="main" id="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/fleet" element={<Fleet />} />
          <Route path="/engines/:engineId" element={<EngineDetail />} />
          <Route path="/lab" element={<SimulationLab />} />
          <Route path="/models" element={<ModelLab />} />
          <Route path="/research" element={<Research />} />
          <Route path="*" element={<div className="state"><strong>Page not found.</strong><p>Use the menu above to go to the dashboard, fleet, simulation lab, model lab or research page.</p></div>} />
        </Routes>
      </main>
    </>
  );
}

export function App() {
  return (
    <SubsetProvider>
      <Shell />
    </SubsetProvider>
  );
}

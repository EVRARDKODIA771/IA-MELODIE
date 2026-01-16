import Recorder from "./Recorder.jsx";
import { useEffect } from "react";
useEffect(async() => {
  const pingBackend = () => {
    fetch("https://ia-melodie.onrender.com/ping").catch(() => {});
  };

  // Ping toutes les 5 minutes
  const interval = setInterval(pingBackend, 5 * 60 * 1000);

  // Ping immédiat au chargement
  await pingBackend();

  return () => clearInterval(interval);
}, []);

function App() {
  return (
    <div>
      <Recorder />
    </div>
  );
}

export default App;

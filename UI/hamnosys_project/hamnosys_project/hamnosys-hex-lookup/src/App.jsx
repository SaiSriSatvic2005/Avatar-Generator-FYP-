import { useState } from "react";
import InputBox from "./components/InputBox";
import OutputBox from "./components/OutputBox";
import hamMap from "./data/hamnosysMap.json";
import "./styles.css";

function App() {

  const [input, setInput] = useState("");
  const [result, setResult] = useState([]);
  const [fullSign, setFullSign] = useState("");

  const lookup = () => {

    const tokens = input
      .toLowerCase()
      .trim()
      .split(/[\s,]+/)
      .filter(Boolean);

    console.log(tokens);

    const results = tokens.map((key) => {

      const hex = hamMap[key];

      if (!hex) {
        return {
          word: key,
          hex: "----",
          char: "❌"
        };
      }

      return {
        word: key,
        hex,
        char: String.fromCodePoint(parseInt(hex, 16))
      };
    });

    // create full HamNoSys string
    const full = results.map(r => r.char).join("");

    setResult(results);
    setFullSign(full);
  };

  return (
    <div className="app">

      <h2>HamNoSys Hex Lookup</h2>

      <InputBox
        value={input}
        onChange={setInput}
        onSubmit={lookup}
      />

      {/* Full HamNoSys symbol sequence */}
      <div className="full-sign">
        <h3>Full HamNoSys Sign</h3>
        <div className="hamnosys-line">{fullSign}</div>
      </div>

      <OutputBox data={result} />

    </div>
  );
}

export default App;
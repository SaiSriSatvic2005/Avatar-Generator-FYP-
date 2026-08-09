function OutputBox({ data }) {
  if (!data || data.length === 0) return null;

  return (
    <div className="output">
      {data.map((item, index) => (
        <div className="result-row" key={index}>
          <div>
            <div><b>{item.word}</b></div>
            <div>0x{item.hex} · {item.category}</div>
          </div>
          <div className="symbol">{item.char}</div>
        </div>
      ))}
    </div>
  );
}

export default OutputBox;

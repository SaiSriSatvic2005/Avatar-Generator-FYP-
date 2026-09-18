function InputBox({ value, onChange, onSubmit }) {
  return (
    <div>
      <input
        placeholder="Enter HamNoSys word"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
      <button onClick={onSubmit}>Lookup</button>
    </div>
  );
}

export default InputBox;

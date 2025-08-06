import React, { useState } from 'react';

function MicInput({ onSpeechResult }) {
  const [listening, setListening] = useState(false);

  const handleMicClick = () => {
    const recognition = new window.webkitSpeechRecognition();
    recognition.lang = 'en-US';
    recognition.start();
    setListening(true);

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      onSpeechResult(transcript);
      setListening(false);
    };
  };

  return (
    <button onClick={handleMicClick}>
      {listening ? '🎙️ Listening...' : '🎤'}
    </button>
  );
}

export default MicInput;
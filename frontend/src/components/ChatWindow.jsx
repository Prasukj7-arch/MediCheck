import React, { useState } from 'react';
import MicInput from './MicInput';
import Message from './Message';

const ChatWindow = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);
    setInput("");

    const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input })
    });

    const data = await response.json();
    setMessages([...newMessages, { role: "assistant", content: data.response }]);
  };

  const handleMicResult = (transcript) => {
    setInput(transcript);
  };

  return (
    <div className="chat-window">
      <div className="messages">
        {messages.map((msg, index) => (
          <Message key={index} role={msg.role} content={msg.content} />
        ))}
      </div>
      <div className="input-container">
        <textarea
          rows="2"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask your medical question..."
        />
        <button onClick={sendMessage}>Send</button>
        <MicInput onSpeechResult={handleMicResult} />
      </div>
    </div>
  );
};

export default ChatWindow;
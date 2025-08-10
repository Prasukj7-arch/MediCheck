import React from 'react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import './styles/main.css';

function App() {
  return (
    <div className="app">
      <header className="header">MediCom</header>
      <div className="main">
        <Sidebar />
        <ChatWindow />
      </div>
    </div>
  );
}

export default App;
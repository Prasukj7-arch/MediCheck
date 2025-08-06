import React from "react";

const ChatHistory = ({ history }) => {
  return (
    <div>
      <h3>Query History</h3>
      {history.length === 0 ? (
        <p>No queries yet.</p>
      ) : (
        history.map((item, index) => (
          <div key={index}>
            <strong>{item.type} Query:</strong>
            <p>{item.query}</p>
            <strong>Response:</strong>
            <p>{item.response}</p>
            <hr />
          </div>
        ))
      )}
    </div>
  );
};

export default ChatHistory;

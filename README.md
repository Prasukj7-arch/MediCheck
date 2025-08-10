# MediCheck - AI-Powered Medical Assistant

MediCheck is a comprehensive AI-powered medical assistant that helps users understand medical reports, get health insights, and manage their healthcare needs.

## Features

### 🔐 Authentication System
- **User Registration**: Create new accounts with email and password
- **User Login**: Secure login with session management
- **User Dashboard**: Personalized homepage with user information

### 🏠 Homepage with Sidebar Navigation
- **Dashboard**: Overview of all available features
- **ChatBot**: AI-powered medical assistant with voice input
- **Book Appointment Details**: Schedule and manage medical appointments
- **Symptom Scanner**: Upload images or use live camera for symptom analysis
- **Report Analysis**: Upload and analyze medical reports with AI
- **Exercise & Home Diet Planner**: Get personalized health recommendations
- **SOS**: Emergency services and critical health information

### 🤖 AI ChatBot with Voice Input
- **Text Chat**: Type questions and get instant responses
- **Voice Input**: Speak to the chatbot using voice recognition
- **Medical Expertise**: Specialized in medical and health-related queries
- **24/7 Availability**: Always available for your health concerns

### 🔍 Symptom Scanner
- **Image Upload**: Upload images from your device for analysis
- **Live Camera**: Use your camera for real-time symptom scanning
- **AI Analysis**: Powered by PyTorch models for accurate diagnosis
- **Condition Information**: Detailed information about detected conditions
- **Confidence Scores**: Get confidence levels for predictions
- **Top Predictions**: View multiple possible conditions with probabilities
- **Medical Recommendations**: Get advice and next steps for detected conditions

### 📊 Medical Report Analysis
- **PDF Upload**: Drag and drop or browse to upload medical reports
- **AI Analysis**: Get detailed insights and explanations
- **Dual Perspectives**: 
  - **Patient View**: Easy-to-understand explanations in simple terms
  - **Doctor View**: Professional medical analysis with clinical terminology
- **Interactive Queries**: Ask specific questions about your reports
- **Normal Answers**: Conversational responses instead of formal letter format

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MediCheck
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   SECRET_KEY=your_secret_key_here
   ```

4. **Optional: Add custom symptom model**
   Place your `.pt` model file in one of these locations:
   - `models/symptom_model.pt`
   - `backend/models/symptom_model.pt`
   - `symptom_model.pt` (root directory)

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and go to `http://localhost:5000`

## Usage

### Getting Started
1. **Sign Up**: Create a new account with your email and password
2. **Login**: Sign in to access your personalized dashboard
3. **Explore Features**: Use the sidebar to navigate between different features

### Using the Symptom Scanner
1. **Navigate to Symptom Scanner**: Click on "Symptom Scanner" in the sidebar
2. **Choose Input Method**:
   - **Upload Image**: Click "Upload Image" and select an image file
   - **Live Camera**: Click "Live Camera" and use your device's camera
3. **Analyze Symptoms**: The AI will analyze the image and provide results
4. **View Results**: See the detected condition, confidence score, and recommendations
5. **Get Information**: Read detailed information about the detected condition

### Using the ChatBot
1. **Text Chat**: Click the chatbot icon in the bottom right corner
2. **Voice Input**: Click the microphone button to speak your question
3. **Ask Questions**: Ask about symptoms, medications, health concerns, etc.

### Medical Report Analysis
1. **Upload Report**: Go to Report Analysis section
2. **Choose Perspective**: Select Patient or Doctor view
3. **Ask Questions**: Get detailed analysis and explanations
4. **Get Insights**: Understand your medical reports better

## Technical Details

### Backend
- **Framework**: Flask (Python)
- **Database**: ChromaDB for vector storage
- **AI Models**: 
  - Sentence Transformers for embeddings
  - Mistral-7B-Instruct for text generation
  - PyTorch models for symptom analysis
- **Authentication**: Session-based with password hashing
- **Image Processing**: OpenCV and PIL for image handling

### Frontend
- **HTML/CSS/JavaScript**: Modern, responsive design
- **Voice Recognition**: Web Speech API
- **Real-time Chat**: AJAX-based communication
- **Drag & Drop**: File upload functionality
- **Camera Integration**: Live camera capture and processing
- **Image Preview**: Real-time image preview and analysis

### AI Features
- **RAG System**: Retrieval-Augmented Generation for medical reports
- **Vector Search**: Semantic search through medical documents
- **Context-Aware Responses**: Tailored responses based on uploaded reports
- **Symptom Analysis**: AI-powered image analysis for symptom detection
- **Condition Classification**: Multi-class classification for various medical conditions

## Security Features

- **Password Hashing**: Secure password storage using Werkzeug
- **Session Management**: Secure user sessions
- **Input Validation**: Proper validation of user inputs
- **File Upload Security**: Secure PDF file handling

## Browser Compatibility

- **Chrome**: Full support (including voice recognition)
- **Firefox**: Full support
- **Safari**: Full support
- **Edge**: Full support

## Voice Recognition Support

The chatbot's voice input feature requires:
- **HTTPS**: For production deployment (required by Web Speech API)
- **Modern Browser**: Chrome, Firefox, Safari, or Edge
- **Microphone Permission**: User must allow microphone access

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## Disclaimer

This application is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

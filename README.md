# MediCheck - AI-Powered Medical Assistant

MediCheck is a comprehensive AI-powered medical assistant that helps users understand medical reports, get health insights, and manage their healthcare needs.

## Features

### 🔐 Authentication System
- **User Registration**: Create new accounts with email and password
- **User Login**: Secure login with session management
- **User Dashboard**: Personalized homepage with user information
- **Database Integration**: Supabase database for persistent user data

### 🏠 Homepage with Sidebar Navigation
- **Dashboard**: Overview of all available features
- **Book Appointment Details**: Schedule and manage medical appointments
- **Symptom Scanner**: Upload images or use live camera for symptom analysis
- **Report Analysis**: Upload and analyze medical reports with AI
- **SOS**: Emergency services and critical health information

### 📅 Appointment Booking System
- **Manual Booking**: Complete appointment booking form with doctor selection, specialty, date, time, and reason
- **Chatbot Booking**: Book appointments through natural language conversation with the AI chatbot
- **Appointment Management**: View, edit, and delete existing appointments
- **Availability Checking**: Check available time slots for specific dates and specialties
- **Specialty Selection**: Choose from a comprehensive list of medical specialties
- **Real-time Updates**: Instant updates when appointments are booked or modified

### 🤖 AI ChatBot with Voice Input
- **Text Chat**: Type questions and get instant responses
- **Voice Input**: Speak to the chatbot using voice recognition
- **Medical Expertise**: Specialized in medical and health-related queries
- **Conversation Context**: Maintains conversation history for personalized responses
- **Follow-up Questions**: Generates relevant follow-up questions to better understand your situation
- **Medical Filtering**: Only responds to health-related questions
- **Appointment Booking**: Book appointments through natural language conversation
- **24/7 Availability**: Always available for your health concerns

### 🔍 Symptom Scanner
- **Image Upload**: Upload images from your device for analysis
- **Live Camera**: Use your camera for real-time symptom scanning
- **AI Analysis**: Powered by LLM for comprehensive medical analysis
- **LLM Integration**: Advanced image analysis using vision-capable models for medical descriptions
- **Medical Relevance Detection**: Automatically identifies if images are medically relevant
- **Condition Information**: Detailed information about detected conditions
- **Confidence Scores**: Get confidence levels for LLM analysis
- **Medical Recommendations**: Get advice and next steps for detected conditions
- **Comprehensive Analysis**: Detailed LLM medical descriptions and analysis

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

3. **Set up Supabase Database**
   
   a. **Create a Supabase project**:
   - Go to [supabase.com](https://supabase.com) and create a new project
   - Note down your project URL and anon key
   
   b. **Set up the database schema**:
   - Go to your Supabase project dashboard
   - Navigate to the SQL Editor
   - Run the SQL commands from `supabase_schema.sql`
   
   c. **Configure environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   SECRET_KEY=your_secret_key_here
   SUPABASE_URL=your_supabase_project_url_here
   SUPABASE_ANON_KEY=your_supabase_anon_key_here
   ```

4. **Optional: Add custom symptom model**
   Place your `.pt` model file in one of these locations:
   - `models/symptom_model.pt`
   - `backend/models/symptom_model.pt`
   - `symptom_model.pt` (root directory)

5. **Vision Model Requirements** (for LLM image analysis):
   - Ensure your OpenRouter API key has access to vision-capable models
   - Supported models: `openai/gpt-4o`, `openai/gpt-4o-mini`, `anthropic/claude-3-5-sonnet`, `anthropic/claude-3-haiku`, `google/gemini-pro-1.5`
   - Run `python test_vision_models.py` to check which models are available
   - If no vision models are available, the system will fall back to text-only analysis

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the application**
   Open your browser and go to `http://localhost:5000`

## Usage

### Getting Started
1. **Sign Up**: Create a new account with your email and password
2. **Login**: Sign in to access your personalized dashboard
3. **Explore Features**: Use the sidebar to navigate between different features

### Using Appointment Booking
1. **Manual Booking**:
   - Navigate to "Book Appointment Details" in the sidebar
   - Fill out the appointment form with doctor name, specialty, date, time, and reason
   - Click "Check Availability" to see available time slots
   - Click "Book Appointment" to confirm your booking
   
2. **Chatbot Booking**:
   - Click the chatbot icon (🤖) in the bottom right corner
   - Type or say: "I want to book an appointment with Dr. Smith for Cardiology on 2024-01-15 at 10:00 for chest pain"
   - The chatbot will automatically parse your request and book the appointment
   
3. **Managing Appointments**:
   - View all your appointments in the appointments section
   - Edit or delete appointments as needed
   - Check appointment status and details

### Using the Symptom Scanner
1. **Navigate to Symptom Scanner**: Click on "Symptom Scanner" in the sidebar
2. **Choose Input Method**:
   - **Upload Image**: Click "Upload Image" and select an image file
   - **Live Camera**: Click "Live Camera" and use your device's camera
3. **Analyze Symptoms**: The AI will analyze the image using LLM
4. **View Results**: See comprehensive analysis including:
   - **LLM Analysis**: Medical relevance, detailed descriptions, symptoms, severity, and recommendations
   - **Raw Analysis**: Full AI analysis text
5. **Get Information**: Read detailed information about the detected condition
6. **Medical Relevance**: The system automatically identifies if the image is medically relevant

### Using the ChatBot
1. **Access ChatBot**: Click the chatbot icon (🤖) in the bottom right corner
2. **Text Input**: Type your medical questions in the input field
3. **Voice Input**: Click the microphone button (🎤) to speak your question
   - Allow microphone access when prompted
   - Speak clearly for better recognition
   - Click the microphone again to stop recording
4. **Medical Questions**: Ask about:
   - Symptoms and conditions
   - Medications and treatments
   - Medical tests and procedures
   - Health concerns and advice
5. **Conversation Context**: The chatbot remembers your conversation history
6. **Follow-up Questions**: Click on suggested questions to continue the conversation
7. **Medical Focus**: Only health-related questions are accepted

### Medical Report Analysis
1. **Upload Report**: Go to Report Analysis section
2. **Choose Perspective**: Select Patient or Doctor view
3. **Ask Questions**: Get detailed analysis and explanations
4. **Get Insights**: Understand your medical reports better

## Testing

### Medical Chatbot Testing
Run the medical chatbot test to verify functionality:
```bash
python test_medical_chatbot.py
```

This test will:
- Check if the application is running
- Verify medical chatbot status
- Test conversation context and follow-up questions
- Validate medical query filtering
- Test voice input functionality (requires manual testing in browser)

### Symptom Scanner Testing
Test the symptom scanner functionality:
```bash
python test_llm_integration.py
```

### Vision Models Testing
Check available vision models for image analysis:
```bash
python test_vision_models.py
```

## Technical Details

### Backend
- **Framework**: Flask (Python)
- **Database**: ChromaDB for vector storage
- **AI Models**: 
  - Sentence Transformers for embeddings
  - Mistral-7B-Instruct for text generation
  - PyTorch models for symptom analysis
  - Vision-capable LLMs for image analysis
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
- **LLM Image Analysis**: Vision-capable LLM integration for comprehensive medical image analysis
- **Medical Relevance Detection**: Automatic identification of medically relevant images
- **Comprehensive Analysis**: Detailed LLM medical descriptions and analysis

## Troubleshooting

### Vision Model Issues
- **Error**: "No endpoints found for gpt-4-vision-preview"
  - **Solution**: The system now automatically tries multiple vision models
  - **Check**: Run `python test_vision_models.py` to see available models
  - **Fallback**: If no vision models work, the system will use text-only analysis

### LLM Integration Issues
- **Error**: "LLM analysis not available"
  - **Check**: Ensure `OPENROUTER_API_KEY` is set in your `.env` file
  - **Verify**: Test your API key with `python test_vision_models.py`
  - **Alternative**: The system will provide placeholder analysis if LLM is not available

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
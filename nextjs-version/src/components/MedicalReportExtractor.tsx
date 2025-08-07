'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, LazyMotion, domAnimation } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { 
  Upload, 
  Zap, 
  AlertCircle, 
  CheckCircle, 
  Info, 
  Lightbulb,
  FileText,
  User,
  Stethoscope,
  Send,
  Loader2
} from 'lucide-react';
import apiService from '@/services/api';
import { Perspective, FileUploadState, QueryState, SystemStatus } from '@/types';
import styles from '@/styles/MedicalReportExtractor.module.css';

const MedicalReportExtractor: React.FC = () => {
  const [perspective, setPerspective] = useState<Perspective>('patient');
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [uploadState, setUploadState] = useState<FileUploadState>({
    isUploading: false,
    isSuccess: false,
    isError: false,
    message: ''
  });
  const [queryState, setQueryState] = useState<QueryState>({
    isLoading: false,
    isError: false,
    response: '',
    error: ''
  });
  const [query, setQuery] = useState('');
  const queryInputRef = useRef<HTMLTextAreaElement>(null);

  // Check system status on mount
  useEffect(() => {
    checkSystemStatus();
  }, []);

  const checkSystemStatus = async () => {
    try {
      const status = await apiService.getSystemStatus();
      setSystemStatus(status);
    } catch (error) {
      setSystemStatus(null);
    }
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploadState({
      isUploading: true,
      isSuccess: false,
      isError: false,
      message: 'Uploading file...'
    });

    try {
      const result = await apiService.uploadFile(file);
      setUploadState({
        isUploading: false,
        isSuccess: true,
        isError: false,
        message: `Successfully uploaded ${result.filename}. Created ${result.chunks_created} text chunks.`
      });
      setQueryState(prev => ({ ...prev, response: '' }));
    } catch (error) {
      setUploadState({
        isUploading: false,
        isSuccess: false,
        isError: true,
        message: 'Failed to upload file. Please try again.'
      });
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxSize: 16 * 1024 * 1024, // 16MB
    multiple: false
  });

  const handleQuerySubmit = async () => {
    if (!query.trim()) {
      setQueryState({
        isLoading: false,
        isError: true,
        response: '',
        error: 'Please enter a question.'
      });
      return;
    }

    setQueryState({
      isLoading: true,
      isError: false,
      response: '',
      error: ''
    });

    try {
      const result = await apiService.sendQuery(query.trim(), perspective);
      setQueryState({
        isLoading: false,
        isError: false,
        response: result.response,
        error: ''
      });
    } catch (error) {
      setQueryState({
        isLoading: false,
        isError: true,
        response: '',
        error: 'Failed to get response. Please try again.'
      });
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.ctrlKey && e.key === 'Enter') {
      handleQuerySubmit();
    }
  };

  const handleExampleClick = (exampleQuery: string) => {
    setQuery(exampleQuery);
    queryInputRef.current?.focus();
  };

  const switchPerspective = (newPerspective: Perspective) => {
    setPerspective(newPerspective);
    setQuery('');
    setQueryState({
      isLoading: false,
      isError: false,
      response: '',
      error: ''
    });
  };

  const getExampleQueries = () => {
    if (perspective === 'patient') {
      return [
        'What does this medical report mean in simple terms?',
        'What medications am I currently taking and why?',
        'Are my test results normal?',
        'What should I do next based on this report?',
        'Can you explain my diagnosis in everyday language?'
      ];
    } else {
      return [
        'What are the key clinical findings in this report?',
        'What differential diagnoses should be considered?',
        'What are the recommended treatment protocols?',
        'Are there any concerning trends in the patient\'s data?',
        'What follow-up tests or monitoring are indicated?'
      ];
    }
  };

  const getPerspectiveInfo = () => {
    if (perspective === 'patient') {
      return {
        title: 'Patient View',
        description: 'Get easy-to-understand explanations of your medical report in simple, non-technical language. Perfect for understanding your health information without medical jargon.'
      };
    } else {
      return {
        title: 'Doctor View',
        description: 'Access detailed clinical analysis and professional medical insights. Get comprehensive medical interpretations suitable for healthcare professionals.'
      };
    }
  };

  return (
    <LazyMotion features={domAnimation}>
      <div className={styles.container}>
        {/* Header */}
        <motion.div 
          className={styles.header}
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1>Medical Report RAG Extractor</h1>
          <p>Upload your medical reports and get expert insights using AI-powered analysis</p>
        </motion.div>

        {/* Perspective Selector */}
        <motion.div 
          className={styles.perspectiveSelector}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4, delay: 0.2 }}
        >
          <motion.button
            className={`${styles.perspectiveBtn} ${perspective === 'patient' ? styles.active : ''}`}
            onClick={() => switchPerspective('patient')}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <User size={16} />
            Patient View
          </motion.button>
          <motion.button
            className={`${styles.perspectiveBtn} ${perspective === 'doctor' ? styles.active : ''}`}
            onClick={() => switchPerspective('doctor')}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Stethoscope size={16} />
            Doctor View
          </motion.button>
        </motion.div>

        {/* Main Content */}
        <div className={styles.mainContent}>
          {/* System Status */}
          {systemStatus && (
            <motion.div 
              className={styles.systemStatus}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Info size={16} />
              System Status: {systemStatus.status}
            </motion.div>
          )}

          {/* Perspective Info */}
          <motion.div 
            className={styles.perspectiveInfo}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <h3>
              {perspective === 'patient' ? <User size={18} /> : <Stethoscope size={18} />}
              {getPerspectiveInfo().title}
            </h3>
            <p>{getPerspectiveInfo().description}</p>
          </motion.div>

          {/* Upload Section */}
          <motion.div 
            className={styles.section}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
          >
            <h2>
              <Upload size={20} />
              Upload Medical Report
            </h2>
            
            <div 
              {...getRootProps()} 
              className={`${styles.uploadArea} ${isDragActive ? styles.dragover : ''}`}
            >
              <input {...getInputProps()} />
              <Upload className={styles.uploadIcon} />
              <h3>Drop your PDF here, or click to browse</h3>
              <p>Supports PDF files up to 16MB</p>
            </div>

            {/* Upload Status */}
            {uploadState.message && (
              <div className={`${styles.status} ${
                uploadState.isSuccess ? styles.success : 
                uploadState.isError ? styles.error : 
                styles.info
              }`}>
                {uploadState.isUploading && <Loader2 className={styles.loading} />}
                {uploadState.isSuccess && <CheckCircle size={16} />}
                {uploadState.isError && <AlertCircle size={16} />}
                {uploadState.message}
              </div>
            )}
          </motion.div>

          {/* Query Section */}
          <motion.div 
            className={styles.section}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.3 }}
          >
            <h2>
              <Zap size={20} />
              Ask Questions ({perspective === 'patient' ? 'Patient' : 'Doctor'} View)
            </h2>
            
            <div className={styles.querySection}>
              <div>
                <textarea
                  ref={queryInputRef}
                  className={styles.queryInput}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder={`Ask questions about your medical report${perspective === 'patient' ? ' in simple terms' : ' for clinical analysis'}...\n\nExamples:\n${perspective === 'patient' ? '- What does this report mean in simple terms?\n- What medications am I taking?\n- Are my test results normal?\n- What should I do next?' : '- What are the key clinical findings?\n- What differential diagnoses should be considered?\n- What treatment protocols are recommended?\n- Are there any concerning trends?'}`}
                  disabled={queryState.isLoading}
                />
                <div className={styles.inputTip}>
                  <Lightbulb size={12} />
                  Tip: Press Ctrl+Enter to submit
                </div>
                <motion.button
                  className={styles.queryBtn}
                  onClick={handleQuerySubmit}
                  disabled={queryState.isLoading || !query.trim()}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {queryState.isLoading ? (
                    <>
                      <Loader2 className={styles.loading} />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Send size={16} />
                      Ask Question
                    </>
                  )}
                </motion.button>
              </div>

              <div>
                <h3>Response:</h3>
                <div className={styles.responseArea}>
                  {!queryState.response && !queryState.isLoading && !queryState.isError && (
                    <div className={styles.errorMessage}>
                      <AlertCircle size={16} />
                      Please enter a question to get started.
                    </div>
                  )}
                  {queryState.isLoading && (
                    <div className={styles.status}>
                      <Loader2 className={styles.loading} />
                      Generating response...
                    </div>
                  )}
                  {queryState.response && (
                    <div>{queryState.response}</div>
                  )}
                  {queryState.isError && queryState.error && (
                    <div className={styles.errorMessage}>
                      <AlertCircle size={16} />
                      {queryState.error}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>

          {/* Example Queries */}
          <motion.div 
            className={styles.exampleQueries}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.4 }}
          >
            <h3>
              <FileText size={18} />
              {perspective === 'patient' ? 'Patient-Friendly' : 'Clinical'} Questions:
            </h3>
            {getExampleQueries().map((exampleQuery, index) => (
              <motion.div
                key={index}
                className={styles.exampleQuery}
                onClick={() => handleExampleClick(exampleQuery)}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
              >
                {exampleQuery}
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </LazyMotion>
  );
};

export default MedicalReportExtractor;

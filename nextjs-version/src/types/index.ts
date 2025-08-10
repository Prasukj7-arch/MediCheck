export interface UploadResponse {
  message: string;
  chunks_created: number;
  filename: string;
}

export interface QueryRequest {
  query: string;
  perspective: 'patient' | 'doctor';
}

export interface QueryResponse {
  response: string;
  query: string;
  perspective: 'patient' | 'doctor';
  chunks_used: number;
}

export interface SystemStatus {
  status: string;
  documents_stored: number;
  embedding_model: string;
  vector_db: string;
  llm_model: string;
}

export interface ApiError {
  error: string;
}

export type Perspective = 'patient' | 'doctor';

export interface FileUploadState {
  isUploading: boolean;
  isSuccess: boolean;
  isError: boolean;
  message: string;
}

export interface QueryState {
  isLoading: boolean;
  isError: boolean;
  response: string;
  error: string;
}

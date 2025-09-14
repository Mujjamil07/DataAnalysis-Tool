// Fixed API Service with proper session handling
const API_BASE_URL = 'http://127.0.0.1:8005';

class ApiService {
  // Health check
  static async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('✅ Health check successful:', data);
        return true;
      } else {
        console.error('❌ Health check failed:', response.status);
        return false;
      }
    } catch (error) {
      console.error('❌ Health check error:', error);
      return false;
    }
  }

  // Upload data
  static async uploadData(file) {
    try {
      console.log('📤 Uploading file:', file.name);
      
      // Convert file to base64
      const base64Data = await this.fileToBase64(file);
      
      const requestBody = {
        filename: file.name,
        data: base64Data,
        content_type: file.type || 'text/csv'
      };
      
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      console.log('✅ Upload successful:', result);
      return result;
      
    } catch (error) {
      console.error('❌ Upload error:', error);
      throw new Error(`Failed to upload data: ${error.message}`);
    }
  }

  // Get sessions - Fixed to handle various response formats
  static async getSessions() {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get sessions: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('✅ Get sessions successful:', result);
      
      // Ensure we always return an array
      if (Array.isArray(result)) {
        return result;
      } else if (result && Array.isArray(result.sessions)) {
        return result.sessions;
      } else if (result && typeof result === 'object') {
        // If result is an object with session data, convert to array
        return Object.keys(result).map(key => ({
          id: key,
          ...result[key]
        }));
      } else {
        console.warn('Sessions response is not in expected format:', result);
        return [];
      }
      
    } catch (error) {
      console.error('❌ Get sessions error:', error);
      // Return empty array instead of throwing to prevent crashes
      return [];
    }
  }

  // Get specific session - Fixed error handling
  static async getSession(sessionId) {
    try {
      if (!sessionId) {
        throw new Error('Session ID is required');
      }
      
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
      });
      
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(`Session ${sessionId} not found`);
        }
        throw new Error(`Failed to get session: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('✅ Get session successful:', result);
      return result;
      
    } catch (error) {
      console.error('❌ Get session error:', error);
      throw new Error(`Failed to get session: ${error.message}`);
    }
  }

  // Run EDA (Exploratory Data Analysis)
  static async runEDA(sessionId) {
    try {
      console.log('📊 Running EDA for session:', sessionId);
      
      if (!sessionId) {
        throw new Error('Session ID is required');
      }
      
      const response = await fetch(`${API_BASE_URL}/eda/${sessionId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to run EDA: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('✅ EDA completed:', result);
      return result;
      
    } catch (error) {
      console.error('❌ EDA error:', error);
      throw new Error(`Failed to run EDA: ${error.message}`);
    }
  }

  // Get available models
  static async getAvailableModels() {
    try {
      const response = await fetch(`${API_BASE_URL}/models`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get models: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('✅ Get models successful:', result);
      return result;
      
    } catch (error) {
      console.error('❌ Get models error:', error);
      throw new Error(`Failed to get models: ${error.message}`);
    }
  }

  // Enhanced analysis - Fixed with proper validation
  static async enhancedAnalysis(sessionId, targetColumn, selectedModels = []) {
    try {
      console.log('🚀 Starting enhanced analysis...');
      console.log('Session ID:', sessionId);
      console.log('Target Column:', targetColumn);
      console.log('Selected Models:', selectedModels);
      
      // Validate inputs
      if (!sessionId || sessionId.trim() === '') {
        throw new Error('Session ID is required and cannot be empty');
      }
      if (!targetColumn || targetColumn.trim() === '') {
        throw new Error('Target column is required and cannot be empty');
      }
      
      // Ensure selectedModels is an array
      const modelsArray = Array.isArray(selectedModels) ? selectedModels : [];
      
      // Check if backend is reachable
      const isHealthy = await this.healthCheck();
      if (!isHealthy) {
        throw new Error('Backend is not available. Please ensure the backend is running on port 8005.');
      }
      
      // Validate session exists (optional check)
      try {
        await this.getSession(sessionId);
      } catch (sessionError) {
        console.warn('Session validation failed:', sessionError.message);
        // Continue anyway - let the backend handle session validation
      }
      
      const requestBody = {
        session_id: sessionId.trim(),
        target_column: targetColumn.trim(),
        models: modelsArray
      };
      
      console.log('Request body:', requestBody);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
      
      let response;
      let lastError;
      
      // Use the correct endpoint that matches our backend
      console.log('🚀 Making enhanced analysis request to:', `${API_BASE_URL}/enhanced-analysis`);
      console.log('📤 Request body:', requestBody);
      
      response = await fetch(`${API_BASE_URL}/enhanced-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      // Check if we got a response
      if (!response) {
        throw new Error('Network error: No response received. This might be a CORS issue or the backend is not running.');
      }
      
      // Handle different response statuses
      if (!response.ok) {
        const errorText = await response.text().catch(() => 'No response body');
        
        if (response.status === 404) {
          throw new Error('Enhanced analysis endpoint not found. Please check your backend configuration.');
        } else if (response.status === 400) {
          throw new Error(`Bad request: ${errorText}`);
        } else if (response.status === 500) {
          throw new Error(`Server error: ${errorText}`);
        } else {
          throw new Error(`Enhanced analysis failed: ${response.status} - ${errorText}`);
        }
      }
      
      // Parse response
      let result;
      try {
        result = await response.json();
      } catch (parseError) {
        throw new Error('Failed to parse response as JSON. Backend may have returned invalid data.');
      }
      
      console.log('✅ Enhanced analysis successful:', result);
      return result;
      
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('Enhanced analysis request timed out (5 minutes). The analysis may be taking too long.');
      }
      
      console.error('❌ Enhanced analysis error:', error);
      
      // Provide more specific error messages based on error type
      if (error.message.includes('CORS')) {
        throw new Error('CORS error: Backend needs to allow cross-origin requests. Please check your backend CORS configuration.');
      } else if (error.message.includes('Failed to fetch') || error.message.includes('Network error')) {
        throw new Error('Network error: Cannot connect to backend. Please ensure the backend is running and accessible on port 8005.');
      } else if (error.message.includes('sessions.some is not a function')) {
        throw new Error('Session data format error. Please refresh the page and try again.');
      }
      
      throw new Error(`Failed to run enhanced analysis: ${error.message}`);
    }
  }

  // Train models - Fixed similar to enhanced analysis
  static async trainModels(sessionId, targetColumn, selectedModels = []) {
    try {
      console.log('🚀 Starting model training...');
      console.log('Session ID:', sessionId);
      console.log('Target Column:', targetColumn);
      console.log('Selected Models:', selectedModels);
      
      if (!sessionId || sessionId.trim() === '') {
        throw new Error('Session ID is required');
      }
      if (!targetColumn || targetColumn.trim() === '') {
        throw new Error('Target column is required');
      }
      
      const isHealthy = await this.healthCheck();
      if (!isHealthy) {
        throw new Error('Backend is not available. Please ensure the backend is running on port 8005.');
      }
      
      const requestBody = {
        session_id: sessionId.trim(),
        target_column: targetColumn.trim(),
        models: Array.isArray(selectedModels) ? selectedModels : []
      };
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout
      
      const response = await fetch(`${API_BASE_URL}/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorText = await response.text().catch(() => 'No response body');
        throw new Error(`Training failed: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      console.log('✅ Training successful:', result);
      return result;
      
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('Training request timed out. Please try again.');
      }
      console.error('❌ Training error:', error);
      throw new Error(`Failed to train models: ${error.message}`);
    }
  }

  // Delete session
  static async deleteSession(sessionId) {
    try {
      if (!sessionId) {
        throw new Error('Session ID is required');
      }
      
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to delete session: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('✅ Delete session successful:', result);
      return result;
      
    } catch (error) {
      console.error('❌ Delete session error:', error);
      throw new Error(`Failed to delete session: ${error.message}`);
    }
  }

  // Delete all sessions
  static async deleteAllSessions() {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to delete all sessions: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('✅ Delete all sessions successful:', result);
      return result;
      
    } catch (error) {
      console.error('❌ Delete all sessions error:', error);
      throw new Error(`Failed to delete all sessions: ${error.message}`);
    }
  }

  // Analyze data for cleaning insights
  static async analyzeData(sessionId) {
    try {
      console.log('🔍 Analyzing data for session:', sessionId);
      
      if (!sessionId) {
        throw new Error('Session ID is required');
      }
      
      const response = await fetch(`${API_BASE_URL}/analyze/${sessionId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Analysis failed: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      console.log('✅ Data analysis successful:', result);
      return result;
      
    } catch (error) {
      console.error('❌ Data analysis error:', error);
      throw new Error(`Failed to analyze data: ${error.message}`);
    }
  }

  // Clean data
  static async cleanData(sessionId, actions) {
    try {
      console.log('🧹 Cleaning data for session:', sessionId);
      console.log('Actions:', actions);
      
      if (!sessionId) {
        throw new Error('Session ID is required');
      }
      if (!actions || !Array.isArray(actions) || actions.length === 0) {
        throw new Error('Cleaning actions are required and must be an array');
      }
      
      const response = await fetch(`${API_BASE_URL}/clean/${sessionId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(actions)
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Cleaning failed: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      console.log('✅ Data cleaning successful:', result);
      return result;
      
    } catch (error) {
      console.error('❌ Data cleaning error:', error);
      throw new Error(`Failed to clean data: ${error.message}`);
    }
  }

  // Helper: Convert file to base64
  static fileToBase64(file) {
    return new Promise((resolve, reject) => {
      if (!file) {
        reject(new Error('No file provided'));
        return;
      }
      
      const reader = new FileReader();
      reader.onload = () => {
        try {
          // Remove the data URL prefix (data:text/csv;base64,)
          const base64 = reader.result.split(',')[1];
          resolve(base64);
        } catch (error) {
          reject(new Error('Failed to process file'));
        }
      };
      reader.onerror = () => {
        reject(new Error('Failed to read file'));
      };
      reader.readAsDataURL(file);
    });
  }

  // Test connection to backend
  static async testConnection() {
    try {
      console.log('🔍 Testing connection to backend...');
      
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        mode: 'cors',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
      });
      
      if (response.ok) {
        console.log('✅ Backend is reachable');
        return { success: true, message: 'Backend is reachable' };
      } else {
        console.log('❌ Backend returned error:', response.status);
        return { success: false, message: `Backend returned status: ${response.status}` };
      }
    } catch (error) {
      console.error('❌ Cannot connect to backend:', error);
      
      let message = 'Cannot connect to backend';
      if (error.message.includes('CORS')) {
        message = 'CORS issue detected. Backend needs CORS headers.';
      } else if (error.message.includes('Failed to fetch')) {
        message = 'Network issue. Backend might not be running.';
      }
      
      return { success: false, message, error: error.message };
    }
  }
}

export default ApiService;
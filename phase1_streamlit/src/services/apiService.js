// ApiService.js - Fixed version for backend integration

const API_BASE_URL = 'http://127.0.0.1:8005';

class ApiService {
  static async healthCheck() {
    try {
      console.log('Checking backend health...');
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Health check response:', data);
        return true;
      } else {
        console.error('Health check failed:', response.status);
        return false;
      }
    } catch (error) {
      console.error('Backend health check failed:', error);
      return false;
    }
  }

  static async uploadData(file) {
    try {
      console.log('Uploading file:', file.name, 'Size:', file.size);
      
      // Convert file to base64
      const base64Data = await this.fileToBase64(file);
      
      const requestBody = {
        filename: file.name,
        data: base64Data,
        content_type: file.type || 'text/csv'
      };
      
      console.log('Sending upload request...');
      
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Upload failed:', response.status, errorText);
        throw new Error(`Upload failed: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      console.log('Upload successful:', result);
      return result;
      
    } catch (error) {
      console.error('Upload error:', error);
      throw new Error(`Failed to upload data: ${error.message}`);
    }
  }

  static async trainModels(sessionId, targetColumn) {
    try {
      console.log('Training models for session:', sessionId, 'Target:', targetColumn);
      
      const requestBody = {
        session_id: sessionId,
        target_column: targetColumn
      };
      
      console.log('Sending train request with body:', requestBody);
      
      const response = await fetch(`${API_BASE_URL}/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });
      
      console.log('Training response status:', response.status);
      console.log('Training response ok:', response.ok);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Training failed:', response.status, errorText);
        throw new Error(`Training failed: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      console.log('Training successful:', result);
      return result;
      
    } catch (error) {
      console.error('Training error:', error);
      throw new Error(`Failed to train models: ${error.message}`);
    }
  }

  static async getResults(sessionId) {
    try {
      console.log('Getting results for session:', sessionId);
      
      const response = await fetch(`${API_BASE_URL}/results/${sessionId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Get results failed:', response.status, errorText);
        throw new Error(`Failed to get results: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      console.log('Results retrieved:', result);
      return result;
      
    } catch (error) {
      console.error('Get results error:', error);
      throw new Error(`Failed to get results: ${error.message}`);
    }
  }

  static async predictData(sessionId, modelName, file) {
    try {
      console.log('Making predictions:', sessionId, modelName);
      
      // Convert file to base64
      const base64Data = await this.fileToBase64(file);
      
      const requestBody = {
        session_id: sessionId,
        model_name: modelName,
        data: base64Data
      };
      
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Prediction failed:', response.status, errorText);
        throw new Error(`Prediction failed: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      console.log('Prediction successful:', result);
      return result;
      
    } catch (error) {
      console.error('Prediction error:', error);
      throw new Error(`Failed to make predictions: ${error.message}`);
    }
  }

  static async getSessions() {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get sessions: ${response.status}`);
      }
      
      return await response.json();
      
    } catch (error) {
      console.error('Get sessions error:', error);
      throw new Error(`Failed to get sessions: ${error.message}`);
    }
  }

  static async deleteSession(sessionId) {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to delete session: ${response.status}`);
      }
      
      return await response.json();
      
    } catch (error) {
      console.error('Delete session error:', error);
      throw new Error(`Failed to delete session: ${error.message}`);
    }
  }

  // Helper function to convert file to base64
  static fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        // Remove the data URL prefix (data:text/csv;base64,)
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  // Helper function to convert CSV data to file for upload
  static dataToFile(data, filename = 'data.csv') {
    try {
      console.log('Converting data to file:', data);
      
      let csvContent = '';
      let headers = [];
      let rows = [];
      
      // Get data from different possible structures
      if (data.rawData && Array.isArray(data.rawData) && data.rawData.length > 0) {
        headers = data.columns || Object.keys(data.rawData[0]);
        rows = data.rawData;
        console.log('Using rawData:', rows.length, 'rows');
      } else if (data.preview && Array.isArray(data.preview) && data.preview.length > 0) {
        headers = data.columns || Object.keys(data.preview[0]);
        rows = data.preview;
        console.log('Using preview data:', rows.length, 'rows');
      } else if (data.data && Array.isArray(data.data) && data.data.length > 0) {
        headers = data.columns || Object.keys(data.data[0]);
        rows = data.data;
        console.log('Using data.data:', rows.length, 'rows');
      } else {
        throw new Error('No valid data found. Check data structure.');
      }
      
      if (headers.length === 0 || rows.length === 0) {
        throw new Error('Empty data or headers not found');
      }
      
      console.log('Headers:', headers);
      console.log('Sample row:', rows[0]);
      
      // Create CSV content
      csvContent = headers.join(',') + '\n';
      
      for (const row of rows) {
        const rowValues = headers.map(header => {
          let value = row[header];
          
          // Handle different value types
          if (value === null || value === undefined || value === '') {
            return '';
          }
          
          // Convert to string
          value = String(value);
          
          // Escape commas and quotes in CSV
          if (value.includes(',') || value.includes('"') || value.includes('\n')) {
            value = `"${value.replace(/"/g, '""')}"`;
          }
          
          return value;
        });
        
        csvContent += rowValues.join(',') + '\n';
      }
      
      console.log('CSV Content length:', csvContent.length);
      console.log('CSV Preview:', csvContent.substring(0, 300));
      
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const file = new File([blob], filename, { type: 'text/csv' });
      
      console.log('File created:', file.name, file.size, 'bytes');
      return file;
      
    } catch (error) {
      console.error('Error converting data to file:', error);
      throw new Error(`Failed to convert data: ${error.message}`);
    }
  }

  // Debug helper
  static async debugBackend() {
    try {
      const response = await fetch(`${API_BASE_URL}/debug/sessions`);
      if (response.ok) {
        const debug = await response.json();
        console.log('Backend debug info:', debug);
        return debug;
      }
    } catch (error) {
      console.error('Debug request failed:', error);
    }
  }

  // Debug session data
  static async debugSession(sessionId) {
    try {
      console.log('Debugging session:', sessionId);
      const response = await fetch(`${API_BASE_URL}/debug/session/${sessionId}`);
      
      if (response.ok) {
        const debug = await response.json();
        console.log('Session debug info:', debug);
        return debug;
      } else {
        const errorText = await response.text();
        console.error('Session debug failed:', response.status, errorText);
        throw new Error(`Session debug failed: ${response.status} - ${errorText}`);
      }
    } catch (error) {
      console.error('Session debug error:', error);
      throw error;
    }
  }

  // Test training with detailed logging
  static async testTraining(sessionId, targetColumn) {
    try {
      console.log('=== TESTING TRAINING ===');
      console.log('Session ID:', sessionId);
      console.log('Target Column:', targetColumn);
      
      // First debug the session
      const sessionDebug = await this.debugSession(sessionId);
      console.log('Session Debug:', sessionDebug);
      
      // Check if target column exists
      if (sessionDebug.columns && !sessionDebug.columns.includes(targetColumn)) {
        console.error('Target column not found!');
        console.log('Available columns:', sessionDebug.columns);
        console.log('Suggested target columns:', sessionDebug.target_column_suggestions);
        throw new Error(`Target column '${targetColumn}' not found. Available: ${sessionDebug.columns.join(', ')}`);
      }
      
      // Now try training
      const requestBody = {
        session_id: sessionId,
        target_column: targetColumn
      };
      
      console.log('Sending training request:', requestBody);
      
      const response = await fetch(`${API_BASE_URL}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });
      
      console.log('Training response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Training failed:', response.status, errorText);
        throw new Error(`Training failed: ${response.status} - ${errorText}`);
      }
      
      const result = await response.json();
      console.log('Training successful:', result);
      return result;
      
    } catch (error) {
      console.error('Test training error:', error);
      throw error;
    }
  }

  // Enhanced trainModels with better error handling
  static async trainModels(sessionId, targetColumn) {
    try {
      console.log('🚀 Starting model training...');
      console.log('Session ID:', sessionId);
      console.log('Target Column:', targetColumn);
      
      // Validate inputs
      if (!sessionId) {
        throw new Error('Session ID is required');
      }
      if (!targetColumn) {
        throw new Error('Target column is required');
      }
      
      // Check backend health first
      const isHealthy = await this.healthCheck();
      if (!isHealthy) {
        throw new Error('Backend server is not available. Please ensure the backend is running.');
      }
      
      // Debug session before training
      console.log('🔍 Debugging session...');
      const sessionDebug = await this.debugSession(sessionId);
      console.log('Session debug info:', sessionDebug);
      
      // Check if target column exists
      if (sessionDebug.columns && !sessionDebug.columns.includes(targetColumn)) {
        console.error('❌ Target column not found!');
        console.log('Available columns:', sessionDebug.columns);
        console.log('Suggested target columns:', sessionDebug.target_column_suggestions);
        throw new Error(`Target column '${targetColumn}' not found. Available columns: ${sessionDebug.columns.join(', ')}`);
      }
      
      console.log('✅ Target column validation passed');
      
      // Prepare request
      const requestBody = {
        session_id: sessionId,
        target_column: targetColumn
      };
      
      console.log('📤 Sending training request:', requestBody);
      
      // Make request with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
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
      
      console.log('📥 Training response status:', response.status);
      console.log('📥 Training response headers:', Object.fromEntries(response.headers.entries()));
      
      if (!response.ok) {
        let errorText;
        try {
          errorText = await response.text();
          console.error('❌ Training failed:', response.status, errorText);
        } catch (e) {
          errorText = 'Failed to read error response';
          console.error('❌ Training failed:', response.status, 'Could not read error response');
        }
        
        // Provide specific error messages
        if (response.status === 404) {
          throw new Error(`Session not found: ${sessionId}. Please upload data first.`);
        } else if (response.status === 400) {
          throw new Error(`Training request failed: ${errorText}`);
        } else if (response.status === 500) {
          throw new Error(`Backend server error: ${errorText}`);
        } else {
          throw new Error(`Training failed: ${response.status} - ${errorText}`);
        }
      }
      
      const result = await response.json();
      console.log('✅ Training successful:', result);
      return result;
      
    } catch (error) {
      console.error('❌ Training error:', error);
      
      // Handle specific error types
      if (error.name === 'AbortError') {
        throw new Error('Training request timed out. Please try again.');
      } else if (error.message.includes('Failed to fetch')) {
        throw new Error('Backend server is not available. Please ensure the backend is running on port 8009.');
      } else {
        throw error;
      }
    }
  }
}

export default ApiService;
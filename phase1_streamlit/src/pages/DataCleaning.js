import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Database, 
  Scissors, 
  BarChart3, 
  PieChart, 
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Download,
  Eye,
  Filter,
  Target,
  Info,
  Zap,
  Trash2,
  Save,
  Upload
} from 'lucide-react';
import toast from 'react-hot-toast';
// import ApiService from '../services/apiService_fresh';

const DataCleaning = () => {
  // State
  const [backendData, setBackendData] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [cleanedData, setCleanedData] = useState(null);
  const [dataStats, setDataStats] = useState(null);
  const [missingData, setMissingData] = useState(null);
  const [outliers, setOutliers] = useState(null);
  const [duplicates, setDuplicates] = useState(null);
  const [dataTypes, setDataTypes] = useState(null);
  const [correlations, setCorrelations] = useState(null);
  const [distributions, setDistributions] = useState(null);
  const [cleaningActions, setCleaningActions] = useState([]);

  // Load data on component mount
  useEffect(() => {
    loadBackendData();
  }, []);

  // Auto-recover session ID if not found
  useEffect(() => {
    const checkSessionId = async () => {
      const storedSessionId = localStorage.getItem('session_id');
      if (!storedSessionId) {
        console.log('⚠️ No session ID in localStorage, trying to recover...');
        try {
          const response = await fetch('http://localhost:8005/sessions');
          if (response.ok) {
            const data = await response.json();
            if (data.sessions && data.sessions.length > 0) {
              const latestSessionId = data.sessions[0].session_id;
              localStorage.setItem('session_id', latestSessionId);
              console.log('✅ Session ID recovered:', latestSessionId);
              setSessionId(latestSessionId);
            }
          }
        } catch (error) {
          console.error('❌ Error recovering session ID:', error);
        }
      }
    };
    
    checkSessionId();
  }, []);

  // Load data from backend
  const loadBackendData = async () => {
    try {
      setIsLoading(true);
      console.log('🔄 Loading data for cleaning...');
      
      // Check backend health
      try {
        const healthResponse = await fetch('http://localhost:8005/health');
        if (!healthResponse.ok) {
          throw new Error('Backend not responding');
        }
      } catch (error) {
        console.log('❌ Backend not healthy');
        toast.error('Backend server is not responding');
        setIsLoading(false);
        return;
      }
      
      // Get session from localStorage
      const storedSessionId = localStorage.getItem('session_id');
      console.log('🔍 Stored session ID:', storedSessionId);
      
      if (storedSessionId) {
        try {
          const sessionResponse = await fetch(`http://localhost:8005/session/${storedSessionId}`);
          if (!sessionResponse.ok) {
            throw new Error('Session not found');
          }
          const sessionData = await sessionResponse.json();
          console.log('✅ Found session:', sessionData);
          
          setBackendData(sessionData);
          setSessionId(sessionData.session_id);
          
          // Auto-analyze data
          await analyzeData(sessionData.session_id);
          
          toast.success(`✅ Data loaded: ${sessionData.filename}`);
        } catch (error) {
          console.log('❌ Stored session not found:', error);
          localStorage.removeItem('session_id');
          toast.error('No data found. Please upload data first.');
        }
      } else {
        console.log('❌ No session ID found in localStorage');
        toast.error('No data found. Please upload data first.');
      }
    } catch (error) {
      console.error('❌ Error loading data:', error);
      toast.error('Failed to load data: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Analyze data for cleaning insights
  const analyzeData = async (sessionId) => {
    try {
      setIsProcessing(true);
      console.log('🔍 Analyzing data for cleaning insights...');
      
      // Get session ID from localStorage if not available
      const currentSessionId = sessionId || localStorage.getItem('session_id');
      console.log('🔍 Using session ID for analysis:', currentSessionId);
      
      if (!currentSessionId) {
        throw new Error('No session ID available');
      }
      
      // Get data analysis from backend
      const analysisResponse = await fetch(`http://localhost:8005/analyze/${currentSessionId}`);
      if (!analysisResponse.ok) {
        throw new Error(`Analysis failed: ${analysisResponse.status}`);
      }
      const analysisData = await analysisResponse.json();
      
      if (analysisData) {
        setDataStats(analysisData.stats);
        setMissingData(analysisData.missing_data);
        setOutliers(analysisData.outliers);
        setDuplicates(analysisData.duplicates);
        setDataTypes(analysisData.data_types);
        setCorrelations(analysisData.correlations);
        setDistributions(analysisData.distributions);
        
        console.log('✅ Data analysis completed');
        toast.success('Data analysis completed!');
      }
    } catch (error) {
      console.error('❌ Analysis error:', error);
      toast.error('Failed to analyze data: ' + error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  // Perform data cleaning
  const performCleaning = async (actions) => {
    try {
      setIsProcessing(true);
      console.log('🧹 Performing data cleaning...', actions);
      console.log('🔍 performCleaning function called successfully!');
      
      // Get session ID from localStorage if not available
      let currentSessionId = sessionId || localStorage.getItem('session_id');
      console.log('🔍 Using session ID:', currentSessionId);
      console.log('🔍 Actions to perform:', actions);
      
      // If no session ID, try to get the latest session from backend
      if (!currentSessionId) {
        console.log('⚠️ No session ID found, trying to get latest session...');
        try {
          const sessionsResponse = await fetch('http://localhost:8005/sessions');
          if (sessionsResponse.ok) {
            const sessionsData = await sessionsResponse.json();
            if (sessionsData.sessions && sessionsData.sessions.length > 0) {
              currentSessionId = sessionsData.sessions[0].session_id;
              console.log('✅ Using latest session ID:', currentSessionId);
              localStorage.setItem('session_id', currentSessionId);
            }
          }
        } catch (error) {
          console.error('❌ Error getting sessions:', error);
        }
      }
      
      console.log('🔍 Final session ID:', currentSessionId);
      console.log('🔍 Backend URL:', `http://localhost:8005/clean/${currentSessionId}`);
      
      if (!currentSessionId) {
        throw new Error('No session ID available');
      }
      
      const cleaningResponse = await fetch(`http://localhost:8005/clean/${currentSessionId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(actions)
      });
      
      console.log('🔍 Response status:', cleaningResponse.status);
      console.log('🔍 Response ok:', cleaningResponse.ok);
      
      if (!cleaningResponse.ok) {
        const errorText = await cleaningResponse.text();
        console.error('❌ Response error:', errorText);
        throw new Error(`Cleaning failed: ${cleaningResponse.status} - ${errorText}`);
      }
      
      const cleaningData = await cleaningResponse.json();
      console.log('✅ Cleaning response:', cleaningData);
      
      if (cleaningData) {
        setCleanedData(cleaningData.cleaned_data);
        setCleaningActions(cleaningData.applied_actions);
        
        // Re-analyze cleaned data
        await analyzeData(currentSessionId);
        
        console.log('✅ Data cleaning completed');
        
        // Special message for outlier removal
        if (cleaningData.outliers_removed > 0) {
          toast.success(`✅ Removed ${cleaningData.outliers_removed} outliers completely! Ready for visualizations.`, { id: 'outliers' });
        } else {
          toast.success('Data cleaning completed! Ready for visualizations.', { id: 'all' });
        }
        
        // Show specific success messages for each action
        if (actions.includes('remove_duplicates')) {
          toast.success('✅ Duplicates removed successfully!', { id: 'duplicates' });
        }
        if (actions.includes('fill_missing_values')) {
          toast.success('✅ Missing values filled successfully!', { id: 'missing' });
        }
        if (actions.includes('standardize_data')) {
          toast.success('✅ Data standardized successfully!', { id: 'standardize' });
        }
      }
    } catch (error) {
      console.error('❌ Cleaning error:', error);
      toast.error('Failed to clean data: ' + error.message);
      
      // Dismiss loading toasts
      toast.dismiss('duplicates');
      toast.dismiss('outliers');
      toast.dismiss('missing');
      toast.dismiss('standardize');
      toast.dismiss('all');
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle specific cleaning actions
  const handleRemoveDuplicates = () => {
    console.log('🧹 Starting duplicate removal...');
    console.log('🔍 Current session ID:', sessionId);
    console.log('🔍 localStorage session ID:', localStorage.getItem('session_id'));
    toast.loading('Removing duplicates...', { id: 'duplicates' });
    performCleaning(['remove_duplicates']);
  };

  const handleRemoveOutliers = () => {
    console.log('🧹 Starting outlier removal...');
    console.log('🔍 Current session ID:', sessionId);
    console.log('🔍 localStorage session ID:', localStorage.getItem('session_id'));
    console.log('🔍 Function called successfully!');
    toast.loading('Removing outliers...', { id: 'outliers' });
    performCleaning(['remove_outliers']);
  };

  const handleFillMissingValues = () => {
    console.log('🧹 Starting missing value filling...');
    toast.loading('Filling missing values...', { id: 'missing' });
    performCleaning(['fill_missing_values']);
  };

  const handleStandardizeData = () => {
    console.log('🧹 Starting data standardization...');
    toast.loading('Standardizing data...', { id: 'standardize' });
    performCleaning(['standardize_data']);
  };

  const handleAllCleaning = () => {
    console.log('🧹 Starting all cleaning actions...');
    console.log('🔍 Current session ID:', sessionId);
    console.log('🔍 localStorage session ID:', localStorage.getItem('session_id'));
    console.log('🔍 Function called successfully!');
    toast.loading('Applying all cleaning actions...', { id: 'all' });
    performCleaning(['remove_duplicates', 'remove_outliers', 'fill_missing_values', 'standardize_data']);
  };

  // Test outlier removal specifically
  const testOutlierRemoval = async () => {
    try {
      console.log('🔍 Testing outlier removal specifically...');
      const currentSessionId = sessionId || localStorage.getItem('session_id');
      
      if (!currentSessionId) {
        toast.error('❌ No session ID available for testing');
        return;
      }
      
      console.log('🔍 Testing with session ID:', currentSessionId);
      
      const response = await fetch(`http://localhost:8005/clean/${currentSessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(['remove_outliers'])
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('✅ Outlier removal test successful:', data);
        toast.success('✅ Outlier removal test successful!');
      } else {
        const errorText = await response.text();
        console.error('❌ Outlier removal test failed:', response.status, errorText);
        toast.error('❌ Outlier removal test failed!');
      }
    } catch (error) {
      console.error('❌ Outlier removal test error:', error);
      toast.error('❌ Outlier removal test error!');
    }
  };

  // Test backend connection
  const testBackendConnection = async () => {
    try {
      console.log('🔍 Testing backend connection...');
      const response = await fetch('http://localhost:8005/health');
      if (response.ok) {
        const data = await response.json();
        console.log('✅ Backend is healthy:', data);
        toast.success('✅ Backend connection successful!');
        
        // Test sessions endpoint
        const sessionsResponse = await fetch('http://localhost:8005/sessions');
        if (sessionsResponse.ok) {
          const sessionsData = await sessionsResponse.json();
          console.log('✅ Sessions available:', sessionsData.sessions.length);
          toast.success(`✅ ${sessionsData.sessions.length} sessions available!`);
          
            // Test cleaning endpoint with first session
            if (sessionsData.sessions.length > 0) {
              const testSessionId = sessionsData.sessions[0].session_id;
              console.log('🔍 Testing cleaning endpoint with session:', testSessionId);
              
              try {
                const testResponse = await fetch(`http://localhost:8005/clean/${testSessionId}`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(['remove_outliers'])
                });
                
                if (testResponse.ok) {
                  const testData = await testResponse.json();
                  console.log('✅ Cleaning endpoint working!', testData);
                  toast.success('✅ Cleaning endpoint working!');
                } else {
                  const errorText = await testResponse.text();
                  console.error('❌ Cleaning endpoint failed:', testResponse.status, errorText);
                  toast.error('❌ Cleaning endpoint failed!');
                }
              } catch (error) {
                console.error('❌ Cleaning endpoint error:', error);
                toast.error('❌ Cleaning endpoint error!');
              }
            }
        }
      } else {
        console.error('❌ Backend not healthy:', response.status);
        toast.error('❌ Backend connection failed!');
      }
    } catch (error) {
      console.error('❌ Backend connection error:', error);
      toast.error('❌ Backend connection error: ' + error.message);
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600">Loading data...</p>
        </div>
      </div>
    );
  }

  // No data state
  if (!backendData) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Data Cleaning & EDA</h1>
          <p className="text-gray-600">Clean and analyze your real dataset</p>
          
          {/* Session Status */}
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <p className="text-sm text-blue-700">
              <strong>Session ID:</strong> {sessionId || localStorage.getItem('session_id') || 'Not found'}
            </p>
            <p className="text-sm text-blue-600">
              <strong>Status:</strong> {sessionId || localStorage.getItem('session_id') ? '✅ Ready' : '⚠️ No session'}
            </p>
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-blue-50 border border-blue-200 rounded-lg p-6"
        >
          <div className="flex items-start">
            <Database className="w-6 h-6 text-blue-600 mt-1 mr-3 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-blue-800 mb-2">Upload Your Real Dataset</h3>
              <p className="text-blue-700 mb-4">
                Upload your actual dataset (CSV, Excel, etc.) to start real-time data cleaning and analysis.
                No sample data - only your real data will be processed.
              </p>
              
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
                onClick={() => window.location.href = '/upload'}
              >
                <Upload className="w-4 h-4 inline mr-2" />
                Upload Real Dataset
              </motion.button>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Data Cleaning & EDA</h1>
        <p className="text-gray-600">Clean and analyze your dataset for better model performance</p>
      </div>

      {/* Dataset Info */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-8"
      >
        <div className="flex items-start">
          <Database className="w-6 h-6 text-blue-600 mt-1 mr-3 flex-shrink-0" />
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-blue-800 mb-2">Dataset Information</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white p-3 rounded-lg">
                <div className="text-sm text-gray-600">Filename</div>
                <div className="font-medium text-gray-900">{backendData.filename}</div>
              </div>
              <div className="bg-white p-3 rounded-lg">
                <div className="text-sm text-gray-600">Rows</div>
                <div className="font-medium text-gray-900">{backendData.data_info.rows.toLocaleString()}</div>
              </div>
              <div className="bg-white p-3 rounded-lg">
                <div className="text-sm text-gray-600">Columns</div>
                <div className="font-medium text-gray-900">{backendData.data_info.columns}</div>
              </div>
              <div className="bg-white p-3 rounded-lg">
                <div className="text-sm text-gray-600">Status</div>
                <div className="font-medium text-green-600">
                  {cleanedData ? 'Cleaned' : 'Raw Data'}
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Data Analysis Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white border border-gray-200 rounded-lg p-6 mb-8"
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900">Data Analysis</h2>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => analyzeData(sessionId)}
            disabled={isProcessing}
            className="bg-green-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-green-700 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 inline mr-2 ${isProcessing ? 'animate-spin' : ''}`} />
            {isProcessing ? 'Analyzing...' : 'Analyze Data'}
          </motion.button>
        </div>

        {dataStats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Missing Data */}
            {missingData && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center mb-3">
                  <AlertTriangle className="w-5 h-5 text-red-600 mr-2" />
                  <h3 className="font-semibold text-red-800">Missing Data</h3>
                </div>
                <div className="space-y-2">
                  {Object.entries(missingData).map(([column, count]) => (
                    <div key={column} className="flex justify-between text-sm">
                      <span className="text-red-700">{column}:</span>
                      <span className="font-medium">{count} missing</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Duplicates */}
            {duplicates && (
              <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
                <div className="flex items-center mb-3">
                  <Trash2 className="w-5 h-5 text-orange-600 mr-2" />
                  <h3 className="font-semibold text-orange-800">Duplicates</h3>
                </div>
                <div className="text-sm">
                  <span className="text-orange-700">Total duplicates:</span>
                  <span className="font-medium ml-2">{duplicates.count}</span>
                </div>
              </div>
            )}

            {/* Outliers */}
            {outliers && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div className="flex items-center mb-3">
                  <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2" />
                  <h3 className="font-semibold text-yellow-800">Outliers</h3>
                </div>
                <div className="space-y-2">
                  {Object.entries(outliers).map(([column, count]) => (
                    <div key={column} className="flex justify-between text-sm">
                      <span className="text-yellow-700">{column}:</span>
                      <span className="font-medium">{count} outliers</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Data Types */}
            {dataTypes && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center mb-3">
                  <Info className="w-5 h-5 text-blue-600 mr-2" />
                  <h3 className="font-semibold text-blue-800">Data Types</h3>
                </div>
                <div className="space-y-2">
                  {Object.entries(dataTypes).map(([column, type]) => (
                    <div key={column} className="flex justify-between text-sm">
                      <span className="text-blue-700">{column}:</span>
                      <span className="font-medium">{type}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Basic Stats */}
            {dataStats && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center mb-3">
                  <BarChart3 className="w-5 h-5 text-green-600 mr-2" />
                  <h3 className="font-semibold text-green-800">Basic Statistics</h3>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-green-700">Mean:</span>
                    <span className="font-medium">{dataStats.mean?.toFixed(2) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-green-700">Std:</span>
                    <span className="font-medium">{dataStats.std?.toFixed(2) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-green-700">Min:</span>
                    <span className="font-medium">{dataStats.min?.toFixed(2) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-green-700">Max:</span>
                    <span className="font-medium">{dataStats.max?.toFixed(2) || 'N/A'}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </motion.div>

      {/* Data Cleaning Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white border border-gray-200 rounded-lg p-6 mb-8"
      >
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Data Cleaning Actions</h2>
        
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2" />
            <div>
              <h3 className="font-medium text-yellow-800">Outlier Removal Fixed!</h3>
              <p className="text-sm text-yellow-700 mt-1">
                Now all outliers from all numeric columns will be removed at once, not one by one.
                <br />
                <strong>BMI outliers will be removed completely in one click!</strong>
              </p>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleRemoveDuplicates}
            disabled={isProcessing}
            className="bg-red-600 text-white p-4 rounded-lg font-medium hover:bg-red-700 transition-colors disabled:opacity-50"
          >
            <Trash2 className="w-6 h-6 mx-auto mb-2" />
            Remove Duplicates
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleRemoveOutliers}
            disabled={isProcessing}
            className="bg-red-600 text-white p-4 rounded-lg font-medium hover:bg-red-700 transition-colors disabled:opacity-50 border-2 border-red-300"
          >
            <AlertTriangle className="w-6 h-6 mx-auto mb-2" />
            Remove All Outliers
            <div className="text-xs mt-1 opacity-90">(All columns at once)</div>
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleFillMissingValues}
            disabled={isProcessing}
            className="bg-blue-600 text-white p-4 rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            <Zap className="w-6 h-6 mx-auto mb-2" />
            Fill Missing Values
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleStandardizeData}
            disabled={isProcessing}
            className="bg-purple-600 text-white p-4 rounded-lg font-medium hover:bg-purple-700 transition-colors disabled:opacity-50"
          >
            <Filter className="w-6 h-6 mx-auto mb-2" />
            Standardize Data
          </motion.button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleAllCleaning}
            disabled={isProcessing}
            className="bg-green-600 text-white py-4 rounded-lg font-medium hover:bg-green-700 transition-colors disabled:opacity-50"
          >
            <Scissors className="w-5 h-5 inline mr-2" />
            {isProcessing ? 'Cleaning...' : 'Apply All Cleaning Actions'}
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => window.location.href = '/visualizations'}
            disabled={isProcessing}
            className="bg-blue-600 text-white py-4 rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            <BarChart3 className="w-5 h-5 inline mr-2" />
            Next: Visualizations
          </motion.button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={testBackendConnection}
            className="bg-yellow-600 text-white py-4 rounded-lg font-medium hover:bg-yellow-700 transition-colors"
          >
            <CheckCircle className="w-5 h-5 inline mr-2" />
            Test Backend
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={testOutlierRemoval}
            className="bg-orange-600 text-white py-4 rounded-lg font-medium hover:bg-orange-700 transition-colors"
          >
            <AlertTriangle className="w-5 h-5 inline mr-2" />
            Test Outlier Removal
          </motion.button>
        </div>
      </motion.div>

      {/* Applied Actions */}
      {cleaningActions.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-green-50 border border-green-200 rounded-lg p-6 mb-8"
        >
          <div className="flex items-center mb-4">
            <CheckCircle className="w-6 h-6 text-green-600 mr-2" />
            <h3 className="text-lg font-semibold text-green-800">Applied Cleaning Actions</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {cleaningActions.map((action, index) => (
              <div key={index} className="bg-white p-3 rounded-lg border">
                <div className="flex items-center">
                  <CheckCircle className="w-4 h-4 text-green-600 mr-2" />
                  <span className="text-sm font-medium text-gray-900">{action}</span>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-6 text-center">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => window.location.href = '/visualizations'}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              <BarChart3 className="w-4 h-4 inline mr-2" />
              View Data Visualizations
            </motion.button>
          </div>
        </motion.div>
      )}

      {/* Data Visualizations */}
      {distributions && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white border border-gray-200 rounded-lg p-6 mb-8"
        >
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Data Distributions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(distributions).map(([column, data]) => (
              <div key={column} className="bg-gray-50 rounded-lg p-4">
                <h3 className="font-medium text-gray-900 mb-3">{column}</h3>
                <div className="space-y-2">
                  {Object.entries(data).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-sm">
                      <span className="text-gray-600">{key}:</span>
                      <span className="font-medium">{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Correlation Matrix */}
      {correlations && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white border border-gray-200 rounded-lg p-6"
        >
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Correlation Matrix</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  <th className="text-left p-2">Feature</th>
                  {Object.keys(correlations).map(column => (
                    <th key={column} className="p-2 text-center">{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(correlations).map(([row, cols]) => (
                  <tr key={row}>
                    <td className="font-medium p-2">{row}</td>
                    {Object.entries(cols).map(([col, value]) => (
                      <td key={col} className={`p-2 text-center ${
                        Math.abs(value) > 0.7 ? 'bg-red-100' :
                        Math.abs(value) > 0.5 ? 'bg-yellow-100' :
                        Math.abs(value) > 0.3 ? 'bg-blue-100' : 'bg-gray-50'
                      }`}>
                        {value.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default DataCleaning;













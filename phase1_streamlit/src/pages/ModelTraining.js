import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, 
  CheckCircle, 
  Activity,
  TrendingUp,
  BarChart3,
  Target,
  Settings,
  Upload,
  Database,
  RefreshCw,
  X
} from 'lucide-react';
import { useAppStore } from '../store/appStore';
import toast from 'react-hot-toast';
import ApiService from '../services/apiService';

const ModelTraining = () => {
  // Store state
  const { data, targetColumn, addModel, updateResults, models, results, setSessionId, setTargetColumn, setResults } = useAppStore();
  
  // Local state
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingSteps, setTrainingSteps] = useState([]);
  const [liveResults, setLiveResults] = useState([]);
  const [backendData, setBackendData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [sessionId, setLocalSessionId] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [selectedModels, setSelectedModels] = useState(['logistic_regression', 'xgboost']);
  const [trainingError, setTrainingError] = useState(null);

  // Available models
  const availableModels = [
    {
      id: 'logistic_regression',
      name: 'Logistic Regression',
      description: 'Linear model for binary classification',
      time: '1-2 minutes',
      color: 'purple'
    },
    {
      id: 'random_forest',
      name: 'Random Forest',
      description: 'Ensemble method using multiple decision trees',
      time: '2-3 minutes',
      color: 'blue'
    },
    {
      id: 'svm',
      name: 'Support Vector Machine',
      description: 'Kernel-based classification algorithm',
      time: '2-3 minutes',
      color: 'orange'
    },
    {
      id: 'xgboost',
      name: 'XGBoost',
      description: 'Gradient boosting with extreme gradient descent',
      time: '3-5 minutes',
      color: 'green'
    },
    {
      id: 'lightgbm',
      name: 'LightGBM',
      description: 'Light gradient boosting machine',
      time: '2-4 minutes',
      color: 'teal'
    }
  ];

  // Load data from backend on component mount
  useEffect(() => {
    loadBackendData();
  }, []);

  // Function to load data from backend
  const loadBackendData = async () => {
    try {
      setIsLoading(true);
      console.log('🔄 Loading backend data...');
      
      // Check backend health
      const isHealthy = await ApiService.healthCheck();
      if (!isHealthy) {
        console.log('❌ Backend not healthy');
        toast.error('Backend server is not responding');
        setIsLoading(false);
        return;
      }
      
      // Get sessions from backend
      const sessionsResponse = await ApiService.getSessions();
      console.log('📊 Sessions response:', sessionsResponse);
      
      if (sessionsResponse && sessionsResponse.sessions && sessionsResponse.sessions.length > 0) {
        const latestSession = sessionsResponse.sessions[0];
        console.log('✅ Found session:', latestSession);
        
        // Update state
        setBackendData(latestSession);
        setLocalSessionId(latestSession.session_id);
        setSessionId(latestSession.session_id);
        
        // Auto-select target column if available
        if (latestSession.data_info && latestSession.data_info.column_names) {
          const columns = latestSession.data_info.column_names;
          console.log('📋 Available columns:', columns);
          
          if (!targetColumn && columns.length > 0) {
            const autoTargetColumn = getBestTargetColumn(columns);
            console.log('🎯 Auto-selected target column:', autoTargetColumn);
            if (autoTargetColumn) {
              setTargetColumn(autoTargetColumn);
            }
          }
        }
        
        toast.success(`✅ Data loaded: ${latestSession.filename}`);
      } else {
        console.log('⚠️ No sessions found in backend');
        toast.info('No data found. Please upload data first.');
      }
    } catch (error) {
      console.error('❌ Error loading backend data:', error);
      toast.error('Failed to load data: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Manual refresh function
  const handleRefresh = async () => {
    setIsRefreshing(true);
    await loadBackendData();
    setIsRefreshing(false);
  };

  // Helper function to get the best target column
  const getBestTargetColumn = (columns) => {
    if (!columns || columns.length === 0) return null;
    
    // Common target column names (case-insensitive)
    const targetNames = [
      'target', 'label', 'class', 'outcome', 'result', 'prediction', 
      'price', 'sales', 'profit', 'revenue', 'quantity', 'discount',
      'diabetes', 'target_column', 'y', 'dependent', 'output'
    ];
    
    // First, try exact matches
    for (const name of targetNames) {
      if (columns.includes(name)) {
        return name;
      }
    }
    
    // Then try case-insensitive matches
    for (const name of targetNames) {
      const found = columns.find(col => col.toLowerCase() === name.toLowerCase());
      if (found) {
        return found;
      }
    }
    
    // If no common target names found, return the last column
    return columns[columns.length - 1];
  };

  // Handle model selection
  const toggleModel = (modelId) => {
    setSelectedModels(prev => 
      prev.includes(modelId) 
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  // Select all models
  const selectAllModels = () => {
    setSelectedModels(availableModels.map(model => model.id));
  };

  // Clear all models
  const clearAllModels = () => {
    setSelectedModels([]);
  };

  // Start training function
  const startTraining = async () => {
    if (!sessionId) {
      toast.error('❌ No session found. Please upload data first.');
      return;
    }

    if (!targetColumn) {
      toast.error('❌ Please select a target column.');
      return;
    }

    if (selectedModels.length === 0) {
      toast.error('❌ Please select at least one model to train.');
      return;
    }

    setTrainingStatus('training');
    setTrainingProgress(0);
    setTrainingSteps(['Initializing training...']);
    setTrainingError(null);

    try {
      console.log('🚀 Starting training with session:', sessionId, 'target:', targetColumn, 'models:', selectedModels);
      
      const response = await ApiService.trainModels(sessionId, targetColumn);
      
      if (response && response.results) {
        setTrainingStatus('completed');
        setTrainingProgress(100);
        setTrainingSteps(['Training completed successfully!']);
        
        // Update results - store all results at once
        setResults(response.results);
        setLiveResults(response.results);
        
        toast.success('🎉 Model training completed successfully!');
        console.log('✅ Training results:', response.results);
      } else {
        throw new Error('Invalid response from training');
      }
    } catch (error) {
      console.error('❌ Training error:', error);
      setTrainingStatus('error');
      setTrainingSteps(['Training failed: ' + error.message]);
      setTrainingError(error.message);
      toast.error('❌ Training failed: ' + error.message);
    }
  };

  // Handle target column change
  const handleTargetColumnChange = (e) => {
    const newTargetColumn = e.target.value;
    setTargetColumn(newTargetColumn);
    console.log('🎯 Target column changed to:', newTargetColumn);
  };

  // Check if we have data to work with
  const hasData = backendData && backendData.data_info;
  const hasColumns = hasData && backendData.data_info.column_names && backendData.data_info.column_names.length > 0;

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Activity className="w-8 h-8 animate-spin mx-auto mb-4 text-watson-600" />
          <p className="text-gray-600">Loading data...</p>
        </div>
      </div>
    );
  }

  // No data state
  if (!hasData) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Training</h1>
          <p className="text-gray-600">Train machine learning models on your dataset</p>
        </div>

        {/* Data Required Alert */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 mb-8"
        >
          <div className="flex items-start">
            <Target className="w-6 h-6 text-yellow-600 mt-1 mr-3 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-yellow-800 mb-2">Data Required</h3>
              <p className="text-yellow-700 mb-4">
                Please upload and process a dataset first to start model training.
              </p>
              
              {/* Debug Info */}
              <div className="bg-yellow-100 p-4 rounded-lg mb-4 text-left">
                <h4 className="font-semibold text-yellow-800 mb-2">Debug Info:</h4>
                <div className="text-sm text-yellow-700 space-y-1">
                  <div>Frontend data exists: {data ? 'Yes' : 'No'}</div>
                  <div>Backend data exists: {backendData ? 'Yes' : 'No'}</div>
                  <div>Session ID: {sessionId || 'None'}</div>
                  <div>Loading: {isLoading ? 'Yes' : 'No'}</div>
                  <div>Has Data: {hasData ? 'Yes' : 'No'}</div>
                  <div>Has Columns: {hasColumns ? 'Yes' : 'No'}</div>
                </div>
              </div>
              
              {/* Action Buttons */}
              <div className="space-y-3">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="bg-watson-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-watson-700 transition-colors"
                  onClick={() => window.location.href = '/upload'}
                >
                  <Upload className="w-4 h-4 inline mr-2" />
                  Go to Data Upload
                </motion.button>
                
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="bg-gray-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-gray-600 transition-colors"
                  onClick={handleRefresh}
                  disabled={isRefreshing}
                >
                  <RefreshCw className={`w-4 h-4 inline mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
                  {isRefreshing ? 'Refreshing...' : 'Refresh Data'}
                </motion.button>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  // Main content with data
  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Training</h1>
        <p className="text-gray-600">Train machine learning models on your dataset</p>
      </div>

      {/* Training Error Banner */}
      {trainingError && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <X className="w-5 h-5 text-red-600 mr-2" />
              <span className="text-red-800 font-medium">Training failed: {trainingError}</span>
            </div>
            <button
              onClick={() => setTrainingError(null)}
              className="text-red-600 hover:text-red-800"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </motion.div>
      )}

      {/* Data Status Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-green-50 border border-green-200 rounded-lg p-6 mb-8"
      >
        <div className="flex items-start">
          <Database className="w-6 h-6 text-green-600 mt-1 mr-3 flex-shrink-0" />
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-green-800 mb-2">Data Status</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="bg-white p-3 rounded-lg">
                <div className="text-sm text-gray-600">Dataset</div>
                <div className="font-medium text-gray-900">{backendData.filename}</div>
              </div>
              <div className="bg-white p-3 rounded-lg">
                <div className="text-sm text-gray-600">Rows</div>
                <div className="font-medium text-gray-900">{backendData.data_info.rows}</div>
              </div>
              <div className="bg-white p-3 rounded-lg">
                <div className="text-sm text-gray-600">Columns</div>
                <div className="font-medium text-gray-900">{backendData.data_info.columns}</div>
              </div>
            </div>
            
            {/* Target Column Selection */}
            {hasColumns && (
              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Target className="w-4 h-4 inline mr-2" />
                  Select Target Column
                </label>
                <select
                  value={targetColumn || ''}
                  onChange={handleTargetColumnChange}
                  className="w-full md:w-64 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-watson-500 focus:border-watson-500"
                >
                  <option value="">Select a target column...</option>
                  {backendData.data_info.column_names.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </div>
      </motion.div>

      {/* Model Selection Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white border border-gray-200 rounded-lg p-6 mb-8"
      >
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Select Models to Train</h2>
        
        {/* Selection Controls */}
        <div className="flex items-center justify-between mb-6">
          <div className="space-x-3">
            <button
              onClick={selectAllModels}
              className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              Select All
            </button>
            <button
              onClick={clearAllModels}
              className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              Clear All
            </button>
          </div>
          <div className="text-sm text-gray-600">
            {selectedModels.length} of {availableModels.length} models selected
          </div>
        </div>

        {/* Model Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          {availableModels.map((model) => (
            <motion.div
              key={model.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`relative border-2 rounded-lg p-4 cursor-pointer transition-all ${
                selectedModels.includes(model.id)
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 bg-white hover:border-gray-300'
              }`}
              onClick={() => toggleModel(model.id)}
            >
              {/* Selection Checkmark */}
              {selectedModels.includes(model.id) && (
                <div className="absolute top-2 right-2">
                  <CheckCircle className="w-5 h-5 text-blue-500" />
                </div>
              )}

              {/* Model Icon */}
              <div className={`w-8 h-8 rounded-lg mb-3 flex items-center justify-center ${
                model.color === 'purple' ? 'bg-purple-100' :
                model.color === 'blue' ? 'bg-blue-100' :
                model.color === 'orange' ? 'bg-orange-100' :
                model.color === 'green' ? 'bg-green-100' :
                model.color === 'teal' ? 'bg-teal-100' : 'bg-gray-100'
              }`}>
                <Brain className={`w-5 h-5 ${
                  model.color === 'purple' ? 'text-purple-600' :
                  model.color === 'blue' ? 'text-blue-600' :
                  model.color === 'orange' ? 'text-orange-600' :
                  model.color === 'green' ? 'text-green-600' :
                  model.color === 'teal' ? 'text-teal-600' : 'text-gray-600'
                }`} />
              </div>

              {/* Model Info */}
              <h3 className={`font-semibold mb-1 ${
                model.color === 'purple' ? 'text-purple-800' :
                model.color === 'blue' ? 'text-blue-800' :
                model.color === 'orange' ? 'text-orange-800' :
                model.color === 'green' ? 'text-green-800' :
                model.color === 'teal' ? 'text-teal-800' : 'text-gray-800'
              }`}>{model.name}</h3>
              <p className="text-sm text-gray-600 mb-2">{model.description}</p>
              <p className="text-xs text-gray-500">Estimated time: {model.time}</p>
            </motion.div>
          ))}
        </div>

        {/* Training Progress */}
        {trainingStatus === 'training' && (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">Training Progress</span>
              <span className="text-sm text-gray-500">{trainingProgress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-watson-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${trainingProgress}%` }}
              ></div>
            </div>
            <div className="mt-2 text-sm text-gray-600">
              {trainingSteps[trainingSteps.length - 1]}
            </div>
          </div>
        )}

        {/* Train Button */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={startTraining}
          disabled={trainingStatus === 'training' || !targetColumn || selectedModels.length === 0}
          className={`w-full px-6 py-4 rounded-lg font-medium transition-colors ${
            trainingStatus === 'training' || !targetColumn || selectedModels.length === 0
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-watson-600 text-white hover:bg-watson-700'
          }`}
        >
          <Brain className="w-5 h-5 inline mr-2" />
          {trainingStatus === 'training' 
            ? 'Training...' 
            : `Train ${selectedModels.length} Selected Model${selectedModels.length !== 1 ? 's' : ''}`
          }
        </motion.button>
      </motion.div>

      {/* Training Results */}
      {trainingStatus === 'completed' && liveResults && Object.keys(liveResults).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white border border-gray-200 rounded-lg p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Training Results</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(liveResults).map(([modelName, metrics]) => (
              <div key={modelName} className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-2">{modelName}</h4>
                {metrics.error ? (
                  <p className="text-red-600 text-sm">{metrics.error}</p>
                ) : (
                  <div className="space-y-1 text-sm">
                    {metrics.r2_score !== undefined && (
                      <div className="flex justify-between">
                        <span className="text-gray-600">R² Score:</span>
                        <span className="font-medium">{(metrics.r2_score * 100).toFixed(2)}%</span>
                      </div>
                    )}
                    {metrics.rmse !== undefined && (
                      <div className="flex justify-between">
                        <span className="text-gray-600">RMSE:</span>
                        <span className="font-medium">{metrics.rmse.toFixed(4)}</span>
                      </div>
                    )}
                    {metrics.training_time !== undefined && (
                      <div className="flex justify-between">
                        <span className="text-gray-600">Training Time:</span>
                        <span className="font-medium">{metrics.training_time}s</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ModelTraining;

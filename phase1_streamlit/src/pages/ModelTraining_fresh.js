import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { 
  Brain, 
  CheckCircle, 
  Activity,
  Upload,
  Database,
  Target,
  RefreshCw,
  X
} from 'lucide-react';
import toast from 'react-hot-toast';
import ApiService from '../services/apiService_fresh';
import { useAppStore } from '../store/appStore';

const ModelTraining = () => {
  // Navigation
  const navigate = useNavigate();
  
  // Global store
  const { 
    results: globalResults, 
    trainingStatus: globalTrainingStatus,
    bestModel: globalBestModel,
    targetColumn: globalTargetColumn,
    sessionId: globalSessionId,
    setResults, 
    setTrainingStatus: setGlobalTrainingStatus, 
    setBestModel,
    setSessionId: setGlobalSessionId,
    setTargetColumn: setGlobalTargetColumn
  } = useAppStore();
  
  // State
  const [backendData, setBackendData] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingResults, setTrainingResults] = useState(null);
  const [trainingError, setTrainingError] = useState(null);

  // Available models - will be loaded from backend
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [isLoadingModels, setIsLoadingModels] = useState(true);

  // Load data on component mount
  useEffect(() => {
    loadBackendData();
    loadAvailableModels();
    
    // Restore training results from global store if available
    if (globalResults && Object.keys(globalResults).length > 0) {
      console.log('🔄 Restoring training results from global store:', globalResults);
      setTrainingResults(globalResults);
      setTrainingStatus(globalTrainingStatus || 'completed');
      setTrainingProgress(100);
    }
    
    // Restore session ID and target column from global store if available
    if (globalSessionId) {
      console.log('🔄 Restoring session ID from global store:', globalSessionId);
      setSessionId(globalSessionId);
    }
    
    if (globalTargetColumn) {
      console.log('🔄 Restoring target column from global store:', globalTargetColumn);
      setTargetColumn(globalTargetColumn);
    }
  }, []);

  // Load data from backend
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
      
      // First try to get session from localStorage
      const storedSessionId = localStorage.getItem('session_id');
      console.log('🔍 Stored session ID:', storedSessionId);
      
      if (storedSessionId) {
        try {
          // Get specific session from backend
          const sessionData = await ApiService.getSession(storedSessionId);
          console.log('✅ Found session from localStorage:', sessionData);
          
          setBackendData(sessionData);
          setSessionId(sessionData.session_id);
          setGlobalSessionId(sessionData.session_id);
          
          // Auto-select target column
          if (sessionData.data_info && sessionData.data_info.column_names) {
            const columns = sessionData.data_info.column_names;
            console.log('📋 Available columns:', columns);
            
            if (!targetColumn && columns.length > 0) {
              const autoTargetColumn = getBestTargetColumn(columns);
              console.log('🎯 Auto-selected target column:', autoTargetColumn);
              if (autoTargetColumn) {
                setTargetColumn(autoTargetColumn);
                setGlobalTargetColumn(autoTargetColumn);
              }
            }
          }
          
          toast.success(`✅ Data loaded: ${sessionData.filename}`);
          return;
        } catch (error) {
          console.log('❌ Stored session not found, clearing localStorage');
          localStorage.removeItem('session_id');
        }
      }
      
      // Fallback: Get all sessions
      const sessionsResponse = await ApiService.getSessions();
      console.log('📊 Sessions response:', sessionsResponse);
      
      if (sessionsResponse && sessionsResponse.sessions && sessionsResponse.sessions.length > 0) {
        const latestSession = sessionsResponse.sessions[0];
        console.log('✅ Found latest session:', latestSession);
        
        setBackendData(latestSession);
        setSessionId(latestSession.session_id);
        setGlobalSessionId(latestSession.session_id);
        
        // Save to localStorage
        localStorage.setItem('session_id', latestSession.session_id);
        
        // Auto-select target column
        if (latestSession.data_info && latestSession.data_info.column_names) {
          const columns = latestSession.data_info.column_names;
          console.log('📋 Available columns:', columns);
          
          if (!targetColumn && columns.length > 0) {
            const autoTargetColumn = getBestTargetColumn(columns);
            console.log('🎯 Auto-selected target column:', autoTargetColumn);
            if (autoTargetColumn) {
              setTargetColumn(autoTargetColumn);
              setGlobalTargetColumn(autoTargetColumn);
            }
          }
        }
        
        toast.success(`✅ Data loaded: ${latestSession.filename}`);
      } else {
        console.log('⚠️ No sessions found');
        toast.info('No data found. Please upload data first.');
      }
    } catch (error) {
      console.error('❌ Error loading data:', error);
      toast.error('Failed to load data: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Manual refresh
  const handleRefresh = async () => {
    setIsRefreshing(true);
    await loadBackendData();
    setIsRefreshing(false);
  };

  // Get best target column
  const getBestTargetColumn = (columns) => {
    if (!columns || columns.length === 0) return null;
    
    const targetNames = [
      'target', 'label', 'class', 'outcome', 'result', 'prediction', 
      'price', 'sales', 'profit', 'revenue', 'quantity', 'discount',
      'diabetes', 'target_column', 'y', 'dependent', 'output'
    ];
    
    // Exact matches
    for (const name of targetNames) {
      if (columns.includes(name)) {
        return name;
      }
    }
    
    // Case-insensitive matches
    for (const name of targetNames) {
      const found = columns.find(col => col.toLowerCase() === name.toLowerCase());
      if (found) {
        return found;
      }
    }
    
    // Return last column
    return columns[columns.length - 1];
  };

  // Load available models from backend
  const loadAvailableModels = async () => {
    try {
      setIsLoadingModels(true);
      console.log('🔄 Loading available models...');
      
      const modelsResponse = await ApiService.getAvailableModels();
      if (modelsResponse && modelsResponse.models) {
        const modelsList = Object.entries(modelsResponse.models).map(([name, config]) => ({
          id: name,
          name: name,
          description: config.description,
          type: config.type,
          formula: config.formula,
          available: config.available,
          time: getModelTime(name),
          color: getModelColor(name)
        })).filter(model => model.available);
        
        setAvailableModels(modelsList);
        console.log('✅ Loaded models:', modelsList);
        
        // Auto-select first few models
        if (modelsList.length > 0) {
          setSelectedModels(modelsList.slice(0, 3).map(model => model.id));
        }
      }
    } catch (error) {
      console.error('❌ Error loading models:', error);
      toast.error('Failed to load models: ' + error.message);
    } finally {
      setIsLoadingModels(false);
    }
  };

  // Get model training time
  const getModelTime = (modelName) => {
    const timeMap = {
      'Linear Regression': '1-2 minutes',
      'Logistic Regression': '1-2 minutes',
      'Random Forest': '2-3 minutes',
      'Support Vector Machine': '2-3 minutes',
      'Decision Tree': '1-2 minutes',
      'K-Nearest Neighbors': '1-2 minutes',
      'Naive Bayes': '1-2 minutes',
      'XGBoost': '3-5 minutes',
      'LightGBM': '2-4 minutes',
      'Gradient Boosting': '3-4 minutes',
      'AdaBoost': '2-3 minutes',
      'Ridge Regression': '1-2 minutes',
      'Lasso Regression': '1-2 minutes'
    };
    return timeMap[modelName] || '2-3 minutes';
  };

  // Get model color
  const getModelColor = (modelName) => {
    const colorMap = {
      'Linear Regression': 'blue',
      'Logistic Regression': 'purple',
      'Random Forest': 'green',
      'Support Vector Machine': 'orange',
      'Decision Tree': 'teal',
      'K-Nearest Neighbors': 'indigo',
      'Naive Bayes': 'pink',
      'XGBoost': 'emerald',
      'LightGBM': 'cyan',
      'Gradient Boosting': 'amber',
      'AdaBoost': 'rose',
      'Ridge Regression': 'sky',
      'Lasso Regression': 'violet'
    };
    return colorMap[modelName] || 'gray';
  };

  // Toggle model selection
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

  // Start training
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
    setGlobalTrainingStatus('training');
    setTrainingProgress(0);
    setTrainingError(null);
    setTrainingResults(null);

    try {
      console.log('🚀 Starting training:', { sessionId, targetColumn, selectedModels });
      
      const response = await ApiService.trainModels(sessionId, targetColumn, selectedModels);
      
      if (response && response.results) {
        setTrainingStatus('completed');
        setGlobalTrainingStatus('completed');
        setTrainingProgress(100);
        setTrainingResults(response.results);
        
        // Store results in global store for Model Results page
        setResults(response.results);
        
        // Set best model if available
        if (response.best_model) {
          setBestModel({
            name: response.best_model,
            score: response.best_score,
            type: response.problem_type
          });
        }
        
        toast.success('🎉 Training completed successfully!');
        console.log('✅ Training results:', response.results);
      } else {
        throw new Error('Invalid response from training');
      }
    } catch (error) {
      console.error('❌ Training error:', error);
      setTrainingStatus('error');
      setGlobalTrainingStatus('error');
      setTrainingError(error.message);
      toast.error('❌ Training failed: ' + error.message);
    }
  };

  // Handle target column change
  const handleTargetColumnChange = (e) => {
    setTargetColumn(e.target.value);
    setGlobalTargetColumn(e.target.value);
  };

  // Check if we have data
  const hasData = backendData && backendData.data_info;
  const hasColumns = hasData && backendData.data_info.column_names && backendData.data_info.column_names.length > 0;

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Activity className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600">Loading data...</p>
        </div>
      </div>
    );
  }

  // No data state
  if (!hasData) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Training</h1>
          <p className="text-gray-600">Train machine learning models on your dataset</p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-yellow-50 border border-yellow-200 rounded-lg p-6"
        >
          <div className="flex items-start">
            <Target className="w-6 h-6 text-yellow-600 mt-1 mr-3 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-yellow-800 mb-2">Data Required</h3>
              <p className="text-yellow-700 mb-4">
                Please upload and process a dataset first to start model training.
              </p>
              
              <div className="space-y-3">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
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

  // Main content
  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Training</h1>
            <p className="text-gray-600">Train machine learning models on your cleaned dataset</p>
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => window.location.href = '/visualizations'}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors"
          >
            <Activity className="w-4 h-4 inline mr-2" />
            Back to Visualizations
          </motion.button>
        </div>
      </div>

      {/* Training Error */}
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

             {/* Data Status */}
       <motion.div
         initial={{ opacity: 0, y: 20 }}
         animate={{ opacity: 1, y: 0 }}
         className="bg-green-50 border border-green-200 rounded-lg p-6 mb-8"
       >
         <div className="flex items-start">
           <Database className="w-6 h-6 text-green-600 mt-1 mr-3 flex-shrink-0" />
           <div className="flex-1">
             <h3 className="text-lg font-semibold text-green-800 mb-2">Your Dataset Status</h3>
             <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
               <div className="bg-white p-3 rounded-lg">
                 <div className="text-sm text-gray-600">Your Dataset</div>
                 <div className="font-medium text-gray-900">{backendData.filename}</div>
               </div>
               <div className="bg-white p-3 rounded-lg">
                 <div className="text-sm text-gray-600">Data Rows</div>
                 <div className="font-medium text-gray-900">{backendData.data_info.rows.toLocaleString()}</div>
               </div>
               <div className="bg-white p-3 rounded-lg">
                 <div className="text-sm text-gray-600">Features</div>
                 <div className="font-medium text-gray-900">{backendData.data_info.columns}</div>
               </div>
             </div>
             
             <div className="bg-white p-3 rounded-lg mb-4">
               <div className="text-sm text-gray-600 mb-2">Available Features:</div>
               <div className="flex flex-wrap gap-2">
                 {backendData.data_info.column_names.map((column, index) => (
                   <span 
                     key={index}
                     className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                   >
                     {column}
                   </span>
                 ))}
               </div>
             </div>
            
            {hasColumns && (
              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Target className="w-4 h-4 inline mr-2" />
                  Select Target Column
                </label>
                <select
                  value={targetColumn}
                  onChange={handleTargetColumnChange}
                  className="w-full md:w-64 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
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

      {/* Model Selection */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white border border-gray-200 rounded-lg p-6 mb-8"
      >
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Select Models to Train</h2>
        
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
              {selectedModels.includes(model.id) && (
                <div className="absolute top-2 right-2">
                  <CheckCircle className="w-5 h-5 text-blue-500" />
                </div>
              )}

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
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${trainingProgress}%` }}
              ></div>
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
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          <Brain className="w-5 h-5 inline mr-2" />
          {trainingStatus === 'training' 
            ? 'Training...' 
            : `Train ${selectedModels.length} Selected Model${selectedModels.length !== 1 ? 's' : ''}`
          }
        </motion.button>
      </motion.div>

      {/* Debug Panel */}
      <div className="mt-4 p-4 bg-gray-100 rounded-lg text-sm">
        <h4 className="font-medium mb-2">Debug Info:</h4>
        <div className="space-y-1">
          <div>Training Status: <span className="font-mono">{trainingStatus}</span></div>
          <div>Has Results: <span className="font-mono">{trainingResults ? 'Yes' : 'No'}</span></div>
          <div>Results Count: <span className="font-mono">{trainingResults ? Object.keys(trainingResults).length : 0}</span></div>
          <div>Global Results: <span className="font-mono">{globalResults ? Object.keys(globalResults).length : 0}</span></div>
        </div>
        
        {/* Always Visible Test Button */}
        <div className="mt-3">
          <button
            onClick={() => {
              console.log('🧪 Always visible test button clicked');
              navigate('/results');
            }}
            className="bg-purple-500 text-white px-4 py-2 rounded text-sm hover:bg-purple-600"
          >
            Test Navigation (Always Visible)
          </button>
        </div>
      </div>

      {/* Training Results */}
      {trainingStatus === 'completed' && trainingResults && (() => {
        console.log('🎯 Rendering training results:', trainingResults);
        return (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white border border-gray-200 rounded-lg p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Training Results</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(trainingResults).map(([modelName, metrics]) => (
              <div key={modelName} className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-2">{modelName}</h4>
                {metrics.error ? (
                  <p className="text-red-600 text-sm">{metrics.error}</p>
                ) : (
                  <div className="space-y-2">
                    {/* Classification Metrics */}
                    {metrics.accuracy !== undefined && (
                      <div className="bg-white p-3 rounded-lg border">
                        <h5 className="font-medium text-gray-800 mb-2 text-sm">Classification Metrics</h5>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Accuracy:</span>
                            <span className="font-medium text-green-600">{(metrics.accuracy * 100).toFixed(2)}%</span>
                          </div>
                          {metrics.precision !== undefined && (
                            <div className="flex justify-between">
                              <span className="text-gray-600">Precision:</span>
                              <span className="font-medium text-blue-600">{(metrics.precision * 100).toFixed(2)}%</span>
                            </div>
                          )}
                          {metrics.recall !== undefined && (
                            <div className="flex justify-between">
                              <span className="text-gray-600">Recall:</span>
                              <span className="font-medium text-purple-600">{(metrics.recall * 100).toFixed(2)}%</span>
                            </div>
                          )}
                          {metrics.f1_score !== undefined && (
                            <div className="flex justify-between">
                              <span className="text-gray-600">F1-Score:</span>
                              <span className="font-medium text-orange-600">{(metrics.f1_score * 100).toFixed(2)}%</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {/* Regression Metrics */}
                    {metrics.r2_score !== undefined && (
                      <div className="bg-white p-3 rounded-lg border">
                        <h5 className="font-medium text-gray-800 mb-2 text-sm">Regression Metrics</h5>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-gray-600">R² Score:</span>
                            <span className="font-medium text-green-600">{(metrics.r2_score * 100).toFixed(2)}%</span>
                          </div>
                          {metrics.rmse !== undefined && (
                            <div className="flex justify-between">
                              <span className="text-gray-600">RMSE:</span>
                              <span className="font-medium text-red-600">{metrics.rmse.toFixed(4)}</span>
                            </div>
                          )}
                          {metrics.mse !== undefined && (
                            <div className="flex justify-between">
                              <span className="text-gray-600">MSE:</span>
                              <span className="font-medium text-blue-600">{metrics.mse.toFixed(4)}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {/* Model Info */}
                    <div className="bg-white p-3 rounded-lg border">
                      <h5 className="font-medium text-gray-800 mb-2 text-sm">Model Information</h5>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        {metrics.training_time !== undefined && (
                          <div className="flex justify-between">
                            <span className="text-gray-600">Training Time:</span>
                            <span className="font-medium">{metrics.training_time}s</span>
                          </div>
                        )}
                        {metrics.type && (
                          <div className="flex justify-between">
                            <span className="text-gray-600">Type:</span>
                            <span className="font-medium capitalize">{metrics.type}</span>
                          </div>
                        )}
                        {metrics.formula && (
                          <div className="col-span-2">
                            <div className="text-gray-600 text-xs mb-1">Formula:</div>
                            <div className="text-xs bg-gray-100 p-2 rounded font-mono">{metrics.formula}</div>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {/* Confusion Matrix (if available) */}
                    {metrics.confusion_matrix && (
                      <div className="bg-white p-3 rounded-lg border">
                        <h5 className="font-medium text-gray-800 mb-2 text-sm">Confusion Matrix</h5>
                        <div className="text-xs">
                          <div className="grid grid-cols-2 gap-1">
                            {metrics.confusion_matrix.map((row, i) => (
                              row.map((cell, j) => (
                                <div key={`${i}-${j}`} className="bg-gray-100 p-1 text-center rounded">
                                  {cell}
                                </div>
                              ))
                            ))}
                          </div>
                          <div className="text-gray-500 text-xs mt-1">
                            [Predicted] × [Actual]
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
          
          {/* Success Message and Next Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="mt-6 text-center"
          >
            <div className="mb-4">
              <div className="inline-flex items-center px-4 py-2 bg-green-100 text-green-800 rounded-lg mb-3">
                <CheckCircle className="w-5 h-5 mr-2" />
                <span className="font-medium">Training Completed Successfully!</span>
              </div>
              <p className="text-gray-600 text-sm">
                Your models have been trained and are ready for analysis. Click below to view detailed results and comparisons.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => {
                  console.log('🚀 Navigating to Model Results page...');
                  navigate('/results');
                }}
                className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-8 py-3 rounded-lg font-medium shadow-lg hover:shadow-xl transition-all duration-300 flex items-center space-x-2"
              >
                <span>View Detailed Results</span>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </motion.button>
              
              {/* Test Button */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => {
                  console.log('🧪 Test button clicked - navigating to results');
                  navigate('/results');
                }}
                className="bg-blue-500 text-white px-6 py-3 rounded-lg font-medium shadow-lg hover:shadow-xl transition-all duration-300"
              >
                Test Navigation
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
        );
      })()}
    </div>
  );
};

export default ModelTraining;













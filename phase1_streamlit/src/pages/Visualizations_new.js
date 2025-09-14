import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Database, 
  BarChart3, 
  TrendingUp, 
  Target,
  RefreshCw,
  AlertTriangle
} from 'lucide-react';
import toast from 'react-hot-toast';
import ApiService from '../services/apiService_fresh';

const Visualizations = () => {
  const [sessionData, setSessionData] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadSessionData();
  }, []);

  const loadSessionData = async () => {
    try {
      setIsLoading(true);
      
      const sessionId = localStorage.getItem('session_id');
      if (!sessionId) {
        console.log('No session ID found');
        setIsLoading(false);
        return;
      }

      const session = await ApiService.getSession(sessionId);
      setSessionData(session);

      const analysis = await ApiService.analyzeData(sessionId);
      setAnalysisData(analysis);

      toast.success('Data loaded successfully!');
    } catch (error) {
      console.error('Error loading data:', error);
      toast.error('Failed to load data');
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600">Loading visualizations...</p>
        </div>
      </div>
    );
  }

  if (!sessionData) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-yellow-600" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">No Data Found</h2>
          <p className="text-gray-600 mb-4">Please upload and process a dataset first</p>
          <button
            onClick={() => window.location.href = '/upload'}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
          >
            Go to Data Upload
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Data Visualizations</h1>
        <p className="text-gray-600">Interactive charts and analysis of your dataset</p>
      </motion.div>

      {/* Dataset Information */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white border border-gray-200 rounded-lg p-6 mb-8"
      >
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Dataset Information</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="text-sm text-gray-600">Rows</div>
            <div className="text-2xl font-bold text-blue-600">{sessionData.data_info?.rows || 0}</div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="text-sm text-gray-600">Columns</div>
            <div className="text-2xl font-bold text-green-600">{sessionData.data_info?.columns || 0}</div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="text-sm text-gray-600">Filename</div>
            <div className="text-lg font-medium text-gray-900">{sessionData.filename}</div>
          </div>
        </div>
      </motion.div>

      {/* Next to Model Training */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-green-50 border border-green-200 rounded-lg p-6"
      >
        <div className="text-center">
          <h3 className="text-lg font-semibold text-green-800 mb-4">Ready for Model Training!</h3>
          <p className="text-green-700 mb-6">
            Your data has been cleaned and analyzed. Now you can train machine learning models on your dataset.
          </p>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => window.location.href = '/training'}
            className="bg-green-600 text-white px-8 py-4 rounded-lg font-medium hover:bg-green-700 transition-colors text-lg"
          >
            <Target className="w-5 h-5 inline mr-2" />
            Start Model Training
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
};

export default Visualizations;


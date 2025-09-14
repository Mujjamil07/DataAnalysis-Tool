import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, 
  CheckCircle, 
  AlertTriangle,
  TrendingUp,
  BarChart3,
  Target,
  Zap,
  Shield,
  Activity,
  Database,
  Filter,
  Award
} from 'lucide-react';
import toast from 'react-hot-toast';
import ApiService from '../services/apiService_fresh';
import { useAppStore } from '../store/appStore';

const EnhancedAnalysis = () => {
  const { 
    sessionId: globalSessionId, 
    targetColumn: globalTargetColumn,
    setResults: setGlobalResults,
    setBestModel: setGlobalBestModel,
    setTrainingStatus: setGlobalTrainingStatus
  } = useAppStore();

  const [sessionId, setSessionId] = useState(globalSessionId);
  const [targetColumn, setTargetColumn] = useState(globalTargetColumn);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [dataQuality, setDataQuality] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [selectedFeatures, setSelectedFeatures] = useState([]);

  // Available analysis options
  const analysisOptions = [
    {
      id: 'outlier_detection',
      name: 'Outlier Detection & Treatment',
      description: 'Detect and handle outliers using IQR and Z-score methods',
      icon: AlertTriangle,
      color: 'text-orange-500'
    },
    {
      id: 'class_imbalance',
      name: 'Class Imbalance Handling',
      description: 'Apply undersampling, oversampling, SMOTE techniques',
      icon: Target,
      color: 'text-blue-500'
    },
    {
      id: 'feature_selection',
      name: 'Advanced Feature Selection',
      description: 'RFE, univariate selection, mutual information',
      icon: Filter,
      color: 'text-green-500'
    },
    {
      id: 'cross_validation',
      name: 'Cross-Validation Analysis',
      description: '5-fold cross-validation for robust evaluation',
      icon: Shield,
      color: 'text-purple-500'
    }
  ];

  useEffect(() => {
    // Restore from global store
    if (globalSessionId) setSessionId(globalSessionId);
    if (globalTargetColumn) setTargetColumn(globalTargetColumn);
  }, [globalSessionId, globalTargetColumn]);

  const startEnhancedAnalysis = async () => {
    if (!sessionId || !targetColumn) {
      toast.error('Please select session and target column');
      return;
    }

    setIsAnalyzing(true);
    try {
      const response = await ApiService.enhancedAnalysis(sessionId, targetColumn);
      
      if (response.success) {
        setAnalysisResults(response.results);
        setDataQuality(response.data_quality_report);
        setRecommendations(response.recommendations);
        setSelectedFeatures(response.selected_features);
        
        // Update global store
        setGlobalResults(response.results);
        setGlobalBestModel(response.best_model);
        setGlobalTrainingStatus('completed');
        
        toast.success('Enhanced analysis completed successfully!');
      } else {
        toast.error('Analysis failed');
      }
    } catch (error) {
      console.error('Enhanced analysis error:', error);
      toast.error('Enhanced analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getMetricColor = (value, type = 'accuracy') => {
    if (type === 'accuracy' || type === 'r2_score') {
      if (value >= 0.9) return 'text-green-600';
      if (value >= 0.8) return 'text-yellow-600';
      return 'text-red-600';
    }
    return 'text-gray-600';
  };

  const formatMetric = (value) => {
    if (typeof value === 'number') {
      return value.toFixed(4);
    }
    return value;
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6"
        >
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2 bg-gradient-to-r from-purple-500 to-blue-600 rounded-lg">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Enhanced ML Analysis</h1>
              <p className="text-gray-600">Advanced preprocessing, feature selection, and model comparison</p>
            </div>
          </div>

          {/* Session and Target Selection */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Session ID
              </label>
              <input
                type="text"
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter session ID"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Target Column
              </label>
              <input
                type="text"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter target column name"
              />
            </div>
          </div>

          {/* Analysis Options */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Analysis Features</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {analysisOptions.map((option) => (
                <motion.div
                  key={option.id}
                  whileHover={{ scale: 1.02 }}
                  className="bg-gray-50 rounded-lg p-4 border border-gray-200"
                >
                  <div className="flex items-center space-x-3 mb-2">
                    <option.icon className={`w-5 h-5 ${option.color}`} />
                    <h4 className="font-medium text-gray-900">{option.name}</h4>
                  </div>
                  <p className="text-sm text-gray-600">{option.description}</p>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Start Analysis Button */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={startEnhancedAnalysis}
            disabled={isAnalyzing || !sessionId || !targetColumn}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 px-6 rounded-lg font-medium shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {isAnalyzing ? (
              <>
                <Activity className="w-5 h-5 animate-spin" />
                <span>Running Enhanced Analysis...</span>
              </>
            ) : (
              <>
                <Brain className="w-5 h-5" />
                <span>Start Enhanced Analysis</span>
              </>
            )}
          </motion.button>
        </motion.div>

        {/* Data Quality Report */}
        {dataQuality && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6"
          >
            <div className="flex items-center space-x-3 mb-4">
              <Database className="w-6 h-6 text-blue-600" />
              <h2 className="text-xl font-bold text-gray-900">Data Quality Report</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-blue-50 rounded-lg p-4">
                <h3 className="font-semibold text-blue-900 mb-2">Dataset Shape</h3>
                <p className="text-2xl font-bold text-blue-600">
                  {dataQuality.shape[0]} × {dataQuality.shape[1]}
                </p>
                <p className="text-sm text-blue-700">Rows × Columns</p>
              </div>
              
              <div className="bg-orange-50 rounded-lg p-4">
                <h3 className="font-semibold text-orange-900 mb-2">Missing Values</h3>
                <p className="text-2xl font-bold text-orange-600">
                  {Object.values(dataQuality.missing_values).reduce((a, b) => a + b, 0)}
                </p>
                <p className="text-sm text-orange-700">Total missing</p>
              </div>
              
              <div className="bg-red-50 rounded-lg p-4">
                <h3 className="font-semibold text-red-900 mb-2">Duplicates</h3>
                <p className="text-2xl font-bold text-red-600">
                  {dataQuality.duplicates}
                </p>
                <p className="text-sm text-red-700">Duplicate records</p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Recommendations */}
        {recommendations && recommendations.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6"
          >
            <div className="flex items-center space-x-3 mb-4">
              <AlertTriangle className="w-6 h-6 text-yellow-600" />
              <h2 className="text-xl font-bold text-gray-900">Recommendations</h2>
            </div>
            
            <div className="space-y-3">
              {recommendations.map((recommendation, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 bg-yellow-50 rounded-lg">
                  <CheckCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
                  <p className="text-gray-700">{recommendation}</p>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Selected Features */}
        {selectedFeatures && selectedFeatures.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6"
          >
            <div className="flex items-center space-x-3 mb-4">
              <Filter className="w-6 h-6 text-green-600" />
              <h2 className="text-xl font-bold text-gray-900">Selected Features</h2>
            </div>
            
            <div className="flex flex-wrap gap-2">
              {selectedFeatures.map((feature, index) => (
                <span
                  key={index}
                  className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium"
                >
                  {feature}
                </span>
              ))}
            </div>
          </motion.div>
        )}

        {/* Analysis Results */}
        {analysisResults && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
          >
            <div className="flex items-center space-x-3 mb-6">
              <BarChart3 className="w-6 h-6 text-purple-600" />
              <h2 className="text-xl font-bold text-gray-900">Model Performance Comparison</h2>
            </div>

            {/* Results Table */}
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Model
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Accuracy
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Precision
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Recall
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      F1-Score
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      CV Mean
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      CV Std
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {Object.entries(analysisResults).map(([modelName, metrics]) => (
                    <tr key={modelName} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <Brain className="w-4 h-4 text-gray-400 mr-2" />
                          <span className="text-sm font-medium text-gray-900">{modelName}</span>
                        </div>
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getMetricColor(metrics.accuracy)}`}>
                        {formatMetric(metrics.accuracy)}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getMetricColor(metrics.precision)}`}>
                        {formatMetric(metrics.precision)}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getMetricColor(metrics.recall)}`}>
                        {formatMetric(metrics.recall)}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getMetricColor(metrics.f1_score)}`}>
                        {formatMetric(metrics.f1_score)}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getMetricColor(metrics.cv_mean)}`}>
                        {formatMetric(metrics.cv_mean)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-600">
                        {formatMetric(metrics.cv_std)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Best Model Highlight */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mt-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200"
            >
              <div className="flex items-center space-x-3">
                <Award className="w-6 h-6 text-green-600" />
                <div>
                  <h3 className="text-lg font-semibold text-green-900">Best Performing Model</h3>
                  <p className="text-green-700">
                    The enhanced analysis identified the optimal model with advanced preprocessing and feature selection techniques.
                  </p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default EnhancedAnalysis;

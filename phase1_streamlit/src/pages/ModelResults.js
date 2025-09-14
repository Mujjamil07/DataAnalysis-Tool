import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  Download, 
  TrendingUp, 
  Award,
  Eye,
  FileText,
  Share2,
  Filter
} from 'lucide-react';
import { useAppStore } from '../store/appStore';

const ModelResults = () => {
  const { models, results } = useAppStore();
  const [selectedModel, setSelectedModel] = useState(null);
  const [viewMode, setViewMode] = useState('overview');

  // Mock results data
  const mockResults = {
    random_forest: {
      accuracy: 94.2,
      precision: 92.8,
      recall: 95.1,
      f1_score: 93.9,
      confusion_matrix: [[85, 15], [8, 92]],
      feature_importance: [
        { feature: 'income', importance: 0.35 },
        { feature: 'age', importance: 0.28 },
        { feature: 'education', importance: 0.22 },
        { feature: 'location', importance: 0.15 }
      ]
    },
    xgboost: {
      accuracy: 93.8,
      precision: 91.5,
      recall: 94.2,
      f1_score: 92.8,
      confusion_matrix: [[83, 17], [9, 91]],
      feature_importance: [
        { feature: 'income', importance: 0.32 },
        { feature: 'age', importance: 0.30 },
        { feature: 'education', importance: 0.25 },
        { feature: 'location', importance: 0.13 }
      ]
    },
    lightgbm: {
      accuracy: 93.5,
      precision: 90.8,
      recall: 93.7,
      f1_score: 92.2,
      confusion_matrix: [[82, 18], [10, 90]],
      feature_importance: [
        { feature: 'income', importance: 0.33 },
        { feature: 'age', importance: 0.29 },
        { feature: 'education', importance: 0.24 },
        { feature: 'location', importance: 0.14 }
      ]
    }
  };

  // Use real results instead of mock data
  const realResults = results || {};
  const modelEntries = Object.entries(realResults);
  const hasResults = modelEntries.length > 0;

  if (!hasResults) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6 text-center"
      >
        <div className="card max-w-md mx-auto">
          <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            No Results Available
          </h3>
          <p className="text-gray-600 mb-4">
            Train some models first to see their performance results.
          </p>
          <button className="btn-primary">
            Train Models
          </button>
        </div>
      </motion.div>
    );
  }

  const bestModel = modelEntries.reduce((best, [id, model]) => {
    const result = mockResults[id];
    return result && result.accuracy > (best?.accuracy || 0) ? { id, ...result } : best;
  }, null);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 space-y-6"
    >
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gradient font-ibm mb-2">
          Model Results & Analysis
        </h1>
        <p className="text-lg text-gray-600 font-inter">
          Compare model performance and download detailed reports
        </p>
      </div>

      {/* Best Model Highlight */}
      {bestModel && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="card bg-gradient-to-r from-green-50 to-emerald-50 border-green-200"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-green-100 rounded-lg">
                <Award className="w-8 h-8 text-green-600" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-gray-900">
                  Best Performing Model
                </h3>
                <p className="text-gray-600">
                  {models[bestModel.id]?.name} - {bestModel.accuracy}% accuracy
                </p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-green-600">
                {bestModel.accuracy}%
              </div>
              <div className="text-sm text-gray-500">Accuracy</div>
            </div>
          </div>
        </motion.div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Comparison */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-gray-900 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-watson-600" />
                Model Performance Comparison
              </h3>
              <div className="flex space-x-2">
                <button
                  onClick={() => setViewMode('overview')}
                  className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                    viewMode === 'overview'
                      ? 'bg-watson-100 text-watson-700'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Overview
                </button>
                <button
                  onClick={() => setViewMode('detailed')}
                  className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                    viewMode === 'detailed'
                      ? 'bg-watson-100 text-watson-700'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Detailed
                </button>
              </div>
            </div>

            {/* Performance Metrics Table */}
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
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {modelEntries.map(([modelName, result]) => {
                    if (!result || result.error) return null;
                    
                    return (
                      <tr key={modelName} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="flex-shrink-0 h-10 w-10">
                              <div className="h-10 w-10 rounded-lg bg-watson-100 flex items-center justify-center">
                                <TrendingUp className="w-5 h-5 text-watson-600" />
                              </div>
                            </div>
                            <div className="ml-4">
                              <div className="text-sm font-medium text-gray-900">
                                {modelName}
                              </div>
                              <div className="text-sm text-gray-500">
                                {result.type || 'Classification'}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900">
                            {result.accuracy ? `${(result.accuracy * 100).toFixed(1)}%` : 
                             result.r2_score ? `${(result.r2_score * 100).toFixed(1)}%` : 'N/A'}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900">
                            {result.precision ? `${(result.precision * 100).toFixed(1)}%` : 'N/A'}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900">
                            {result.recall ? `${(result.recall * 100).toFixed(1)}%` : 'N/A'}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900">
                            {result.f1_score ? `${(result.f1_score * 100).toFixed(1)}%` : 'N/A'}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button
                            onClick={() => setSelectedModel(modelName)}
                            className="text-watson-600 hover:text-watson-900 mr-3"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button className="text-gray-600 hover:text-gray-900">
                            <Download className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="space-y-6">
          {/* Export Options */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Download className="w-5 h-5 mr-2 text-watson-600" />
              Export Results
            </h3>
            
            <div className="space-y-3">
              <button className="w-full btn-secondary flex items-center justify-center">
                <FileText className="w-4 h-4 mr-2" />
                Download PDF Report
              </button>
              <button className="w-full btn-secondary flex items-center justify-center">
                <Share2 className="w-4 h-4 mr-2" />
                Share Results
              </button>
              <button className="w-full btn-secondary flex items-center justify-center">
                <Filter className="w-4 h-4 mr-2" />
                Filter Models
              </button>
            </div>
          </div>

          {/* Summary Stats */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Summary Statistics
            </h3>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Total Models:</span>
                <span className="text-sm font-medium">{modelEntries.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Best Accuracy:</span>
                <span className="text-sm font-medium text-green-600">
                  {bestModel?.accuracy}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Average Accuracy:</span>
                <span className="text-sm font-medium">
                  {(modelEntries.reduce((sum, [id]) => sum + (mockResults[id]?.accuracy || 0), 0) / modelEntries.length).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Model View */}
      {selectedModel && realResults[selectedModel] && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-gray-900">
              {models[selectedModel]?.name} - Detailed Analysis
            </h3>
            <button
              onClick={() => setSelectedModel(null)}
              className="text-gray-500 hover:text-gray-700"
            >
              ×
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Confusion Matrix */}
            <div>
              <h4 className="text-lg font-medium text-gray-900 mb-4">Confusion Matrix</h4>
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="grid grid-cols-2 gap-2 text-center">
                  <div className="p-3 bg-green-100 rounded">
                    <div className="text-2xl font-bold text-green-800">
                      {realResults[selectedModel].confusion_matrix?.[0]?.[0] || 'N/A'}
                    </div>
                    <div className="text-xs text-green-600">True Negatives</div>
                  </div>
                  <div className="p-3 bg-red-100 rounded">
                    <div className="text-2xl font-bold text-red-800">
                      {realResults[selectedModel].confusion_matrix?.[0]?.[1] || 'N/A'}
                    </div>
                    <div className="text-xs text-red-600">False Positives</div>
                  </div>
                  <div className="p-3 bg-red-100 rounded">
                    <div className="text-2xl font-bold text-red-800">
                      {realResults[selectedModel].confusion_matrix?.[1]?.[0] || 'N/A'}
                    </div>
                    <div className="text-xs text-red-600">False Negatives</div>
                  </div>
                  <div className="p-3 bg-green-100 rounded">
                    <div className="text-2xl font-bold text-green-800">
                      {realResults[selectedModel].confusion_matrix?.[1]?.[1] || 'N/A'}
                    </div>
                    <div className="text-xs text-green-600">True Positives</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Feature Importance */}
            <div>
              <h4 className="text-lg font-medium text-gray-900 mb-4">Feature Importance</h4>
              <div className="space-y-3">
                {(realResults[selectedModel]?.feature_importance || []).map((feature, index) => (
                  <div key={feature.feature} className="flex items-center justify-between">
                    <span className="text-sm text-gray-700">{feature.feature}</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-watson-600 h-2 rounded-full"
                          style={{ width: `${feature.importance * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium text-gray-900">
                        {(feature.importance * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default ModelResults;

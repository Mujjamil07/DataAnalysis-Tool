import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Zap, 
  Activity,
  TrendingUp,
  AlertTriangle,
  CheckCircle
} from 'lucide-react';

const PerformanceOptimizer = ({ dataSize, onOptimize }) => {
  const [optimizationLevel, setOptimizationLevel] = useState('auto');
  const [performance, setPerformance] = useState({
    displayRows: 1000,
    analysisSpeed: 'fast',
    memoryUsage: 'low'
  });

  useEffect(() => {
    if (dataSize > 50000) {
      setOptimizationLevel('high');
      setPerformance({
        displayRows: 1000,
        analysisSpeed: 'optimized',
        memoryUsage: 'optimized'
      });
    } else if (dataSize > 10000) {
      setOptimizationLevel('medium');
      setPerformance({
        displayRows: 5000,
        analysisSpeed: 'fast',
        memoryUsage: 'low'
      });
    } else {
      setOptimizationLevel('low');
      setPerformance({
        displayRows: 10000,
        analysisSpeed: 'instant',
        memoryUsage: 'minimal'
      });
    }
  }, [dataSize]);

  const getOptimizationMessage = () => {
    if (dataSize > 100000) {
      return {
        title: '🚀 Ultra Performance Mode',
        message: 'Large dataset detected. Using advanced optimizations for smooth experience.',
        color: 'red'
      };
    } else if (dataSize > 50000) {
      return {
        title: '⚡ High Performance Mode',
        message: 'Optimizing display and analysis for large datasets.',
        color: 'orange'
      };
    } else if (dataSize > 10000) {
      return {
        title: '📊 Balanced Performance',
        message: 'Good balance between display and performance.',
        color: 'yellow'
      };
    } else {
      return {
        title: '✅ Standard Mode',
        message: 'Standard performance mode for smaller datasets.',
        color: 'green'
      };
    }
  };

  const optimizationInfo = getOptimizationMessage();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-lg border border-blue-200"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center">
          <Zap className="w-5 h-5 text-blue-600 mr-2" />
          <h3 className="text-sm font-semibold text-gray-900">
            {optimizationInfo.title}
          </h3>
        </div>
        <div className={`px-2 py-1 text-xs rounded-full ${
          optimizationInfo.color === 'red' ? 'bg-red-100 text-red-700' :
          optimizationInfo.color === 'orange' ? 'bg-orange-100 text-orange-700' :
          optimizationInfo.color === 'yellow' ? 'bg-yellow-100 text-yellow-700' :
          'bg-green-100 text-green-700'
        }`}>
          {optimizationLevel.toUpperCase()}
        </div>
      </div>

      <p className="text-xs text-gray-600 mb-3">
        {optimizationInfo.message}
      </p>

      <div className="grid grid-cols-3 gap-3 text-xs">
        <div className="text-center">
          <div className="font-medium text-gray-900">{performance.displayRows.toLocaleString()}</div>
          <div className="text-gray-500">Display Rows</div>
        </div>
        <div className="text-center">
          <div className="font-medium text-gray-900 capitalize">{performance.analysisSpeed}</div>
          <div className="text-gray-500">Analysis Speed</div>
        </div>
        <div className="text-center">
          <div className="font-medium text-gray-900 capitalize">{performance.memoryUsage}</div>
          <div className="text-gray-500">Memory Usage</div>
        </div>
      </div>

      {dataSize > 50000 && (
        <div className="mt-3 p-2 bg-yellow-50 rounded border border-yellow-200">
          <div className="flex items-center">
            <AlertTriangle className="w-4 h-4 text-yellow-600 mr-2" />
            <p className="text-xs text-yellow-700">
              <strong>Performance Tip:</strong> For datasets with {dataSize.toLocaleString()}+ rows, 
              we optimize display while maintaining full data for ML training.
            </p>
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default PerformanceOptimizer;




import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Database, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  BarChart3,
  PieChart,
  Activity,
  Target
} from 'lucide-react';
import { useAppStore } from '../store/appStore';

const DataAnalysis = () => {
  const { data, targetColumn } = useAppStore();
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (data && data.rawData) {
      performRealTimeAnalysis();
    }
  }, [data, targetColumn]);

  const performRealTimeAnalysis = () => {
    setLoading(true);
    
    // Use requestAnimationFrame for better performance
    requestAnimationFrame(() => {
      const rawData = data.rawData;
      const columns = data.columns;
      
      // Real-time statistical analysis
      const analysis = {
        // Basic stats
        totalRows: rawData.length,
        totalColumns: columns.length,
        memoryUsage: (JSON.stringify(rawData).length / 1024 / 1024).toFixed(2),
        
        // Data quality metrics
        missingValues: {},
        duplicates: 0,
        outliers: {},
        dataTypes: {},
        
        // Statistical analysis
        numericalStats: {},
        categoricalStats: {},
        
        // Target analysis
        targetAnalysis: null,
        
        // Data quality score
        qualityScore: 0
      };

      // Analyze each column
      columns.forEach(column => {
        const values = rawData.map(row => row[column]);
        
        // Missing values
        const missing = values.filter(v => v === null || v === undefined || v === '').length;
        analysis.missingValues[column] = missing;
        
        // Data types
        const sampleValue = values.find(v => v !== null && v !== undefined && v !== '');
        if (sampleValue !== undefined) {
          if (!isNaN(sampleValue) && sampleValue !== '') {
            analysis.dataTypes[column] = 'numerical';
            
            // Numerical statistics
            const numValues = values.filter(v => !isNaN(v) && v !== '').map(v => parseFloat(v));
            if (numValues.length > 0) {
              const mean = numValues.reduce((a, b) => a + b, 0) / numValues.length;
              const sorted = numValues.sort((a, b) => a - b);
              const median = sorted[Math.floor(sorted.length / 2)];
              const min = Math.min(...numValues);
              const max = Math.max(...numValues);
              const std = Math.sqrt(numValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / numValues.length);
              
              analysis.numericalStats[column] = {
                mean: mean.toFixed(2),
                median: median.toFixed(2),
                min: min.toFixed(2),
                max: max.toFixed(2),
                std: std.toFixed(2),
                count: numValues.length
              };
            }
          } else {
            analysis.dataTypes[column] = 'categorical';
            
            // Categorical statistics
            const valueCounts = {};
            values.forEach(v => {
              if (v !== null && v !== undefined && v !== '') {
                valueCounts[v] = (valueCounts[v] || 0) + 1;
              }
            });
            
            analysis.categoricalStats[column] = {
              uniqueValues: Object.keys(valueCounts).length,
              valueCounts: valueCounts,
              mostCommon: Object.entries(valueCounts).sort((a, b) => b[1] - a[1])[0]
            };
          }
        }
      });

      // Duplicate detection
      const uniqueRows = new Set();
      rawData.forEach(row => {
        uniqueRows.add(JSON.stringify(row));
      });
      analysis.duplicates = rawData.length - uniqueRows.size;

      // Target analysis
      if (targetColumn && analysis.dataTypes[targetColumn]) {
        if (analysis.dataTypes[targetColumn] === 'categorical') {
          analysis.targetAnalysis = {
            type: 'classification',
            classes: analysis.categoricalStats[targetColumn].uniqueValues,
            distribution: analysis.categoricalStats[targetColumn].valueCounts,
            balanced: analysis.categoricalStats[targetColumn].uniqueValues <= 10
          };
        } else {
          analysis.targetAnalysis = {
            type: 'regression',
            range: `${analysis.numericalStats[targetColumn].min} - ${analysis.numericalStats[targetColumn].max}`,
            mean: analysis.numericalStats[targetColumn].mean
          };
        }
      }

      // Calculate quality score
      const totalCells = rawData.length * columns.length;
      const missingTotal = Object.values(analysis.missingValues).reduce((a, b) => a + b, 0);
      analysis.qualityScore = Math.round(((totalCells - missingTotal - analysis.duplicates) / totalCells) * 100);

             setAnalysis(analysis);
       setLoading(false);
     });
  };

  if (!data) {
    return (
      <div className="text-center py-8">
        <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-600">Upload data to see real-time analysis</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Real-Time Data Analysis</h2>
        <p className="text-gray-600">Live statistical analysis and data quality assessment</p>
      </div>

      {loading ? (
        <div className="text-center py-8">
          <Activity className="w-8 h-8 text-blue-600 mx-auto mb-4 animate-spin" />
          <p className="text-gray-600">Analyzing data in real-time...</p>
        </div>
      ) : analysis ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Basic Information */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Database className="w-5 h-5 mr-2 text-blue-600" />
              Dataset Overview
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span>Total Rows:</span>
                <span className="font-medium">{analysis.totalRows.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Total Columns:</span>
                <span className="font-medium">{analysis.totalColumns}</span>
              </div>
              <div className="flex justify-between">
                <span>Memory Usage:</span>
                <span className="font-medium">{analysis.memoryUsage} MB</span>
              </div>
              <div className="flex justify-between">
                <span>Data Quality Score:</span>
                <span className={`font-medium ${
                  analysis.qualityScore >= 90 ? 'text-green-600' :
                  analysis.qualityScore >= 70 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {analysis.qualityScore}%
                </span>
              </div>
            </div>
          </div>

          {/* Data Quality */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <CheckCircle className="w-5 h-5 mr-2 text-green-600" />
              Data Quality
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span>Missing Values:</span>
                <span className="font-medium text-red-600">
                  {Object.values(analysis.missingValues).reduce((a, b) => a + b, 0).toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Duplicate Rows:</span>
                <span className="font-medium text-orange-600">{analysis.duplicates.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Numerical Columns:</span>
                <span className="font-medium">
                  {Object.values(analysis.dataTypes).filter(t => t === 'numerical').length}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Categorical Columns:</span>
                <span className="font-medium">
                  {Object.values(analysis.dataTypes).filter(t => t === 'categorical').length}
                </span>
              </div>
            </div>
          </div>

          {/* Target Analysis */}
          {analysis.targetAnalysis && (
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Target className="w-5 h-5 mr-2 text-purple-600" />
                Target Variable Analysis
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span>Type:</span>
                  <span className="font-medium capitalize">{analysis.targetAnalysis.type}</span>
                </div>
                {analysis.targetAnalysis.type === 'classification' && (
                  <>
                    <div className="flex justify-between">
                      <span>Classes:</span>
                      <span className="font-medium">{analysis.targetAnalysis.classes}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Balanced:</span>
                      <span className={`font-medium ${
                        analysis.targetAnalysis.balanced ? 'text-green-600' : 'text-yellow-600'
                      }`}>
                        {analysis.targetAnalysis.balanced ? 'Yes' : 'No'}
                      </span>
                    </div>
                  </>
                )}
                {analysis.targetAnalysis.type === 'regression' && (
                  <>
                    <div className="flex justify-between">
                      <span>Range:</span>
                      <span className="font-medium">{analysis.targetAnalysis.range}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Mean:</span>
                      <span className="font-medium">{analysis.targetAnalysis.mean}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Column Analysis */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <BarChart3 className="w-5 h-5 mr-2 text-indigo-600" />
              Column Analysis
            </h3>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {Object.entries(analysis.dataTypes).map(([column, type]) => (
                <div key={column} className="border-b border-gray-200 pb-2">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">{column}</span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      type === 'numerical' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                    }`}>
                      {type}
                    </span>
                  </div>
                  <div className="text-xs text-gray-600 mt-1">
                    Missing: {analysis.missingValues[column]} 
                    {type === 'numerical' && analysis.numericalStats[column] && (
                      <span> | Mean: {analysis.numericalStats[column].mean}</span>
                    )}
                    {type === 'categorical' && analysis.categoricalStats[column] && (
                      <span> | Unique: {analysis.categoricalStats[column].uniqueValues}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </motion.div>
  );
};

export default DataAnalysis;

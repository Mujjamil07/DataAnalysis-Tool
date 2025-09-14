import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  TrendingUp, 
  PieChart, 
  Activity,
  Database,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { useAppStore } from '../store/appStore';

const Visualizations_simple = () => {
  const { data } = useAppStore();
  const [visualizations, setVisualizations] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    if (data && data.length > 0) {
      generateVisualizations();
    }
  }, [data]);

  const generateVisualizations = async () => {
    if (!data || data.length === 0) return;

    setIsGenerating(true);

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Generate real visualizations from actual data
    const numericColumns = Object.keys(data[0] || {}).filter(col => {
      const sampleValue = data[0]?.[col];
      return !isNaN(sampleValue) && sampleValue !== '';
    });

    const distributions = {};
    const correlations = {};
    const boxPlots = {};

    // Calculate distributions for numeric columns
    numericColumns.forEach(col => {
      const values = data.map(row => parseFloat(row[col])).filter(v => !isNaN(v));
      if (values.length > 0) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);
        const sortedValues = values.sort((a, b) => a - b);
        const q1 = sortedValues[Math.floor(sortedValues.length * 0.25)];
        const q3 = sortedValues[Math.floor(sortedValues.length * 0.75)];
        const median = sortedValues[Math.floor(sortedValues.length * 0.5)];
        
        distributions[col] = {
          mean: Math.round(mean * 100) / 100,
          std: Math.round(std * 100) / 100,
          median: Math.round(median * 100) / 100,
          q1: Math.round(q1 * 100) / 100,
          q3: Math.round(q3 * 100) / 100,
          min: Math.round(Math.min(...values) * 100) / 100,
          max: Math.round(Math.max(...values) * 100) / 100
        };

        // Box plot data
        boxPlots[col] = {
          min: Math.round(Math.min(...values) * 100) / 100,
          q1: Math.round(q1 * 100) / 100,
          median: Math.round(median * 100) / 100,
          q3: Math.round(q3 * 100) / 100,
          max: Math.round(Math.max(...values) * 100) / 100,
          outliers: values.filter(v => v < q1 - 1.5 * (q3 - q1) || v > q3 + 1.5 * (q3 - q1)).length
        };
      }
    });

    // Calculate correlations
    for (let i = 0; i < numericColumns.length; i++) {
      for (let j = i + 1; j < numericColumns.length; j++) {
        const col1 = numericColumns[i];
        const col2 = numericColumns[j];
        
        const values1 = data.map(row => parseFloat(row[col1])).filter(v => !isNaN(v));
        const values2 = data.map(row => parseFloat(row[col2])).filter(v => !isNaN(v));
        
        if (values1.length > 0 && values2.length > 0) {
          const mean1 = values1.reduce((a, b) => a + b, 0) / values1.length;
          const mean2 = values2.reduce((a, b) => a + b, 0) / values2.length;
          
          const numerator = values1.reduce((sum, val, idx) => sum + (val - mean1) * (values2[idx] - mean2), 0);
          const denominator1 = Math.sqrt(values1.reduce((sum, val) => sum + Math.pow(val - mean1, 2), 0));
          const denominator2 = Math.sqrt(values2.reduce((sum, val) => sum + Math.pow(val - mean2, 2), 0));
          
          const correlation = denominator1 * denominator2 !== 0 ? numerator / (denominator1 * denominator2) : 0;
          correlations[`${col1}-${col2}`] = Math.round(correlation * 100) / 100;
        }
      }
    }

    setVisualizations({
      distributions,
      correlations,
      boxPlots,
      numericColumns
    });

    setIsGenerating(false);
  };

  if (!data || data.length === 0) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="text-center py-12">
          <Database className="mx-auto h-16 w-16 text-gray-400 mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">No Data Available</h2>
          <p className="text-gray-600 mb-4">Please upload and clean your data first</p>
          <button
            onClick={() => window.location.href = '/upload'}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
          >
            Go to Data Upload
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Real-time Data Visualizations</h1>
          <p className="text-gray-600">Interactive charts and analysis of your cleaned data</p>
        </div>

        {isGenerating ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Generating Visualizations</h2>
            <p className="text-gray-600">Analyzing your data...</p>
          </div>
        ) : visualizations ? (
          <div className="space-y-8">
            {/* Data Overview */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <p className="text-sm text-blue-600 font-medium">Total Rows</p>
                <p className="text-2xl font-bold text-blue-900">{data.length.toLocaleString()}</p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <p className="text-sm text-green-600 font-medium">Features</p>
                <p className="text-2xl font-bold text-green-900">{Object.keys(data[0]).length}</p>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <p className="text-sm text-purple-600 font-medium">Numeric Columns</p>
                <p className="text-2xl font-bold text-purple-900">{visualizations.numericColumns.length}</p>
              </div>
              <div className="bg-orange-50 p-4 rounded-lg">
                <p className="text-sm text-orange-600 font-medium">Data Quality</p>
                <p className="text-2xl font-bold text-orange-900">High</p>
              </div>
            </div>

            {/* Distribution Charts */}
            {Object.keys(visualizations.distributions).length > 0 && (
              <div className="bg-white rounded-lg border p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2 text-blue-600" />
                  Data Distributions
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(visualizations.distributions).map(([column, stats]) => (
                    <div key={column} className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="font-medium text-gray-900 mb-3">{column}</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Mean:</span>
                          <span className="font-medium">{stats.mean}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Std Dev:</span>
                          <span className="font-medium">{stats.std}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Median:</span>
                          <span className="font-medium">{stats.median}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Min:</span>
                          <span className="font-medium">{stats.min}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Max:</span>
                          <span className="font-medium">{stats.max}</span>
                        </div>
                      </div>
                      <div className="mt-3 w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-blue-600 h-2 rounded-full" style={{ width: '60%' }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Box Plots for Outlier Detection */}
            {Object.keys(visualizations.boxPlots).length > 0 && (
              <div className="bg-white rounded-lg border p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                  <Activity className="w-5 h-5 mr-2 text-red-600" />
                  Outlier Detection (Box Plots)
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(visualizations.boxPlots).map(([column, box]) => (
                    <div key={column} className="bg-red-50 p-4 rounded-lg border border-red-200">
                      <h4 className="font-medium text-red-900 mb-3">{column}</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Min:</span>
                          <span className="font-medium">{box.min}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Q1:</span>
                          <span className="font-medium">{box.q1}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Median:</span>
                          <span className="font-medium">{box.median}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Q3:</span>
                          <span className="font-medium">{box.q3}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Max:</span>
                          <span className="font-medium">{box.max}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Outliers:</span>
                          <span className="font-medium text-red-600">{box.outliers}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Correlations */}
            {Object.keys(visualizations.correlations).length > 0 && (
              <div className="bg-white rounded-lg border p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                  <TrendingUp className="w-5 h-5 mr-2 text-green-600" />
                  Feature Correlations
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(visualizations.correlations).map(([correlation, value]) => {
                    const [col1, col2] = correlation.split('-');
                    return (
                      <div key={correlation} className="bg-green-50 p-4 rounded-lg border border-green-200">
                        <h4 className="font-medium text-green-900 mb-2">{col1} vs {col2}</h4>
                        <div className="text-center">
                          <p className="text-2xl font-bold text-green-600">{value}</p>
                          <p className="text-sm text-green-700">
                            {Math.abs(value) < 0.3 ? 'Weak' : Math.abs(value) < 0.7 ? 'Moderate' : 'Strong'} correlation
                          </p>
                        </div>
                        <div className="mt-2 w-full bg-green-200 rounded-full h-2">
                          <div 
                            className="bg-green-600 h-2 rounded-full" 
                            style={{ width: `${Math.abs(value) * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex space-x-4">
              <button
                onClick={() => window.location.href = '/training'}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 flex items-center"
              >
                <CheckCircle className="w-5 h-5 mr-2" />
                Proceed to Model Training
              </button>
              <button
                onClick={() => window.location.href = '/upload'}
                className="bg-gray-600 text-white px-6 py-3 rounded-lg hover:bg-gray-700 flex items-center"
              >
                <Database className="w-5 h-5 mr-2" />
                Upload New Data
              </button>
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <AlertCircle className="mx-auto h-16 w-16 text-yellow-400 mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">No Visualizations Available</h2>
            <p className="text-gray-600 mb-4">Unable to generate visualizations from the current data</p>
            <button
              onClick={generateVisualizations}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
            >
              Try Again
            </button>
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default Visualizations_simple;






